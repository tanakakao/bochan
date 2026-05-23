from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence, Union

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import t_batch_mode_transform

from ..hetero_utils import (
    _normal_cdf,
    _normal_pdf,
    aggregate_objectives,
    make_weight_tensor,
    stack_multi_summaries,
)


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum", "max"]
OutputMode = Literal["mean", "sum", "max", "min", "weighted_mean", "product"]
NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "inverse_exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract"]


# =========================================================
# Generic helpers
# =========================================================
def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _reduce_q(score: Tensor, reduction: ReductionType) -> Tensor:
    if score.ndim == 0:
        return score
    if score.shape[-1] == 1:
        return score.squeeze(-1)
    if reduction == "mean":
        return score.mean(dim=-1)
    if reduction == "sum":
        return score.sum(dim=-1)
    if reduction == "max":
        return score.max(dim=-1).values
    raise ValueError(f"Unknown reduction: {reduction!r}.")


def _coerce_reference_to_tensor(X_ref, *, ref: Optional[Tensor] = None) -> Optional[Tensor]:
    if X_ref is None:
        return None

    if torch.is_tensor(X_ref):
        out = X_ref
    elif isinstance(X_ref, (list, tuple)):
        tensors = []
        for item in X_ref:
            if item is None:
                continue
            t = _coerce_reference_to_tensor(item, ref=ref)
            if t is not None and t.numel() > 0:
                tensors.append(t)

        if len(tensors) == 0:
            return None
        if len(tensors) == 1:
            out = tensors[0]
        else:
            try:
                out = torch.cat(tensors, dim=-2)
            except RuntimeError:
                out = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=-2)
    else:
        raise TypeError(
            "X_pending must be None, Tensor, list, or tuple. "
            f"Got {type(X_ref)}."
        )

    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)

    return out


def _align_pointwise_score_to_X(
    score: Tensor,
    Xt: Tensor,
    *,
    name: str,
    reduce_extra: Literal["mean", "sum"] = "mean",
) -> Tensor:
    target = Xt.shape[:-1]
    out = score

    if out.shape == target:
        return out

    if out.ndim >= 1 and out.shape[-1] == 1 and len(target) >= 1 and target[-1] != 1:
        out = out.squeeze(-1)
        if out.shape == target:
            return out

    if out.numel() == int(torch.tensor(target).prod().item()):
        return out.reshape(target)

    while out.ndim > len(target):
        out = out.mean(dim=0) if reduce_extra == "mean" else out.sum(dim=0)
        if out.shape == target:
            return out

    if out.shape == target:
        return out

    if out.ndim == len(target) and out.shape[-1] == target[-1]:
        try:
            return out.expand(target)
        except RuntimeError:
            pass

    raise RuntimeError(
        f"{name}: score shape mismatch. "
        f"score.shape={tuple(score.shape)}, expected={tuple(target)}, Xt.shape={tuple(Xt.shape)}."
    )


def _expand_scalar_or_list(value, m: int, name: str):
    if isinstance(value, (float, int)):
        return [float(value)] * m
    if torch.is_tensor(value):
        v = value.reshape(-1)
        if v.numel() == 1:
            return [float(v.item())] * m
        if v.numel() != m:
            raise ValueError(f"{name} must have {m} elements. Got {v.numel()}.")
        return v
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return list(value) * m
        if len(value) != m:
            raise ValueError(f"{name} must have length {m}. Got {len(value)}.")
        return list(value)
    raise ValueError(f"Unsupported {name}: {type(value)}")


def _objective_call(objective, score: Tensor, X: Optional[Tensor]):
    try:
        return objective(score, X=X)
    except TypeError:
        return objective(score)


def _apply_score_objective(owner, score: Tensor, X: Optional[Tensor], name: str) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score
    out = _objective_call(objective, score, X)
    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return Tensor. Got {type(out)}.")
    return out


def _align_mo_samples_to_X(
    Y: Tensor,
    X: Tensor,
    *,
    m: int,
    name: str,
) -> Tensor:
    """MC multi-output objective 用の shape に正規化する。

    Expected:
        Y: sample_shape x batch_shape x q x m
        X:                batch_shape x q x d

    Some custom / ModelList / hetero posteriors may return samples where the
    MC sample dimension is not the leading dimension.  BoTorch's
    MCMultiOutputObjective validates that ``Y.shape[-2] == X.shape[-2]``.
    This helper moves a q-sized axis to ``-2`` when necessary.
    """
    Xq = _ensure_q_batch(X)
    q = int(Xq.shape[-2])
    batch_shape = tuple(Xq.shape[:-2])

    if Y.ndim < 2:
        raise RuntimeError(f"{name}: expected at least 2D tensor, got {tuple(Y.shape)}.")

    if Y.shape[-1] != m:
        if m == 1 and Y.shape[-1] != 1:
            Y = Y.unsqueeze(-1)
        else:
            raise RuntimeError(
                f"{name}: last dim must be output dimension m={m}. "
                f"Got Y.shape={tuple(Y.shape)}."
            )

    # Already valid for BoTorch MCMultiOutputObjective.
    if Y.ndim >= 2 and Y.shape[-2] == q:
        return Y

    # Move the last q-sized non-output axis to -2.
    candidate_axes = [
        ax for ax in range(Y.ndim - 1)
        if Y.shape[ax] == q
    ]
    if candidate_axes:
        ax = candidate_axes[-1]
        Y = Y.movedim(ax, -2)
        if Y.shape[-2] == q:
            return Y

    # If q=1, the q axis may have been squeezed. Insert it before m.
    if q == 1:
        # Prefer keeping leading MC / batch dims unchanged.
        Y = Y.unsqueeze(-2)
        return Y

    raise RuntimeError(
        f"{name}: could not align objective samples to X q-batch. "
        f"Y.shape={tuple(Y.shape)}, X.shape={tuple(Xq.shape)}, q={q}, m={m}."
    )


def _align_pointwise_to_X_q_m(
    Y: Tensor,
    X: Tensor,
    *,
    m: int,
    name: str,
) -> Tensor:
    """summary tensorsを batch_shape x q x m に揃える。"""
    Xq = _ensure_q_batch(X)
    q = int(Xq.shape[-2])
    target = tuple(Xq.shape[:-2]) + (q, m)

    if Y.shape == target:
        return Y

    if Y.shape[-1] != m:
        raise RuntimeError(
            f"{name}: expected last dim m={m}, got shape={tuple(Y.shape)}."
        )

    if Y.ndim >= 2 and Y.shape[-2] == q:
        # Extra leading dimensions, e.g. fully Bayesian batch. Average them.
        out = Y
        while out.ndim > len(target):
            out = out.mean(dim=0)
        if out.shape == target:
            return out

    if Y.numel() == int(torch.tensor(target).prod().item()):
        return Y.reshape(target)

    if q == 1 and Y.shape == tuple(Xq.shape[:-2]) + (m,):
        return Y.unsqueeze(-2)

    raise RuntimeError(
        f"{name}: could not align tensor to batch_shape x q x m. "
        f"Y.shape={tuple(Y.shape)}, target={target}, X.shape={tuple(Xq.shape)}."
    )


def _maybe_concat_pending_points(owner, X: Tensor) -> Tensor:
    """BoTorch の concatenate_pending_points を使わずに X_pending を連結する。

    parent qEHVI / qNEHVI の forward は decorator 内で shape assertion するため、
    custom objective の MC sample dim が残るケースでは super().forward を呼べない。
    そのため、pending 連結だけを手動で再現する。
    """
    Xq = _ensure_q_batch(X)
    X_pending = getattr(owner, "X_pending", None)
    if X_pending is None:
        return Xq

    Xp = _ensure_q_batch(X_pending).to(device=Xq.device, dtype=Xq.dtype)
    target_batch = tuple(Xq.shape[:-2])

    if tuple(Xp.shape[:-2]) != target_batch:
        # Try broadcasting singleton / no-batch pending points to current t-batch.
        if Xp.ndim == 2:
            Xp = Xp.unsqueeze(0)
        try:
            Xp = Xp.expand(*target_batch, Xp.shape[-2], Xp.shape[-1])
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not broadcast X_pending to candidate batch shape. "
                f"X.shape={tuple(Xq.shape)}, X_pending.shape={tuple(X_pending.shape)}."
            ) from exc

    return torch.cat([Xq, Xp], dim=-2)


def _finalize_acq_output_to_batch(
    value: Tensor,
    X: Tensor,
    *,
    name: str,
    reduce_extra: Literal["mean", "sum"] = "mean",
) -> Tensor:
    """acquisition output を BoTorch が期待する t-batch shape に揃える。

    t_batch_mode_transform は acquisition output が X.shape[:-2] と一致することを
    期待する。custom MC acquisition で sample_shape が残ると [128] のような
    出力になり assertion に落ちるため、余分な leading dims を平均で潰す。
    """
    Xq = _ensure_q_batch(X)
    target = tuple(Xq.shape[:-2])
    out = value

    if out.shape == target:
        return out

    if len(target) == 0:
        # No explicit t-batch. All remaining dims are MC / fantasy / extra batch dims.
        if out.ndim == 0:
            return out
        return out.mean() if reduce_extra == "mean" else out.sum()

    # If trailing dims exactly match target, reduce all leading extra dims.
    while out.ndim > len(target):
        out = out.mean(dim=0) if reduce_extra == "mean" else out.sum(dim=0)
        if out.shape == target:
            return out

    if out.shape == target:
        return out

    if out.numel() == int(torch.tensor(target).prod().item()):
        return out.reshape(target)

    # Common case: out=[mc] and target=[1] or similar. Average extra MC dim
    # then expand/reshape if possible.
    if out.ndim == 1 and len(target) == 1:
        return out.mean().expand(*target)

    raise RuntimeError(
        f"{name}: could not align acquisition output to t-batch shape. "
        f"value.shape={tuple(value.shape)}, target={target}, X.shape={tuple(Xq.shape)}."
    )


# =========================================================
# Ordinal likelihood / utility helpers
# =========================================================
def _default_ordinal_likelihood(model: Model):
    lik = getattr(model, "ordinal_likelihood", None)
    if lik is not None:
        return lik
    lik = getattr(model, "likelihood", None)
    if lik is not None:
        return lik
    return None


def _extract_ordinal_likelihoods(model: Model, ordinal_likelihoods=None) -> list:
    if ordinal_likelihoods is not None:
        if isinstance(ordinal_likelihoods, (list, tuple)):
            return list(ordinal_likelihoods)
        return [ordinal_likelihoods]

    if hasattr(model, "ordinal_likelihoods"):
        likes = getattr(model, "ordinal_likelihoods")
        likes = likes() if callable(likes) else likes
        return list(likes)

    if hasattr(model, "likelihoods"):
        likes = getattr(model, "likelihoods")
        likes = likes() if callable(likes) else likes
        return list(likes)

    if hasattr(model, "models"):
        out = []
        for i, sm in enumerate(model.models):
            lik = _default_ordinal_likelihood(sm)
            if lik is None:
                raise ValueError(
                    f"Could not infer ordinal likelihood for submodel index {i}. "
                    "Pass ordinal_likelihoods explicitly."
                )
            out.append(lik)
        return out

    lik = _default_ordinal_likelihood(model)
    if lik is not None:
        return [lik]

    raise ValueError(
        "Could not infer ordinal likelihoods from model. "
        "Expected model.ordinal_likelihoods, model.likelihoods, "
        "model.models[i].ordinal_likelihood / likelihood, or pass ordinal_likelihoods."
    )


def _get_cutpoints(ordinal_likelihood) -> Tensor:
    for name in ("cutpoints", "transformed_cutpoints", "thresholds", "_cutpoints"):
        obj = getattr(ordinal_likelihood, name, None)
        if obj is not None:
            return torch.as_tensor(obj() if callable(obj) else obj).detach().flatten()

    raw_cutpoints = getattr(ordinal_likelihood, "raw_cutpoints", None)
    if raw_cutpoints is not None:
        if hasattr(ordinal_likelihood, "_ordered_cutpoints"):
            return torch.as_tensor(ordinal_likelihood._ordered_cutpoints()).detach().flatten()
        if hasattr(ordinal_likelihood, "transform_cutpoints"):
            return torch.as_tensor(
                ordinal_likelihood.transform_cutpoints(raw_cutpoints)
            ).detach().flatten()
        return torch.sort(torch.as_tensor(raw_cutpoints).detach().flatten()).values

    raise ValueError("Could not obtain cutpoints from ordinal_likelihood.")


def _ordinal_logit_latent_to_probs(
    latent: Tensor,
    ordinal_likelihood,
    eps: float = 1e-12,
) -> Tensor:
    f = latent
    if f.ndim >= 1 and f.shape[-1] == 1:
        f = f.squeeze(-1)

    if ordinal_likelihood is None:
        raise ValueError("ordinal_likelihood is required to convert latent values to probabilities.")

    num_classes = int(getattr(ordinal_likelihood, "num_classes", 0))
    cutpoints = _get_cutpoints(ordinal_likelihood).to(device=f.device, dtype=f.dtype)

    if num_classes <= 0:
        num_classes = int(cutpoints.numel() + 1)

    z = cutpoints.view(*((1,) * f.ndim), -1) - f.unsqueeze(-1)
    cdf = torch.sigmoid(z)

    p0 = cdf[..., :1]
    plast = 1.0 - cdf[..., -1:]
    if num_classes == 2:
        probs = torch.cat([p0, plast], dim=-1)
    else:
        pmid = cdf[..., 1:] - cdf[..., :-1]
        probs = torch.cat([p0, pmid, plast], dim=-1)

    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _latent_to_probs(
    latent: Tensor,
    *,
    ordinal_likelihood,
    latent_to_probs: Optional[Callable[[Tensor], Tensor]] = None,
    eps: float = 1e-12,
) -> Tensor:
    latent_in = latent.unsqueeze(-1) if latent.ndim >= 1 and latent.shape[-1] != 1 else latent

    if latent_to_probs is not None:
        probs = latent_to_probs(latent_in)
    else:
        probs = None
        if ordinal_likelihood is not None:
            for name in (
                "probs_from_latent",
                "class_probs_from_latent",
                "class_probs_from_f",
                "latent_to_probs",
                "probs",
            ):
                fn = getattr(ordinal_likelihood, name, None)
                if callable(fn):
                    probs = fn(latent_in)
                    break

            if probs is None:
                probs = _ordinal_logit_latent_to_probs(
                    latent_in,
                    ordinal_likelihood=ordinal_likelihood,
                    eps=eps,
                )

        if probs is None:
            raise ValueError(
                "Could not convert latent values to class probabilities. "
                "Pass latent_to_probs explicitly or expose a probability conversion method."
            )

    if probs.ndim >= 2 and probs.shape[-2] == 1:
        probs = probs.squeeze(-2)

    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _utility_values_tensor(
    utility_values_list,
    *,
    ref: Tensor,
    m: int,
    ordinal_likelihoods: Sequence,
) -> list[Tensor]:
    if utility_values_list is None:
        out = []
        for lik in ordinal_likelihoods:
            num_classes = int(getattr(lik, "num_classes", 0))
            if num_classes <= 0:
                num_classes = int(_get_cutpoints(lik).numel() + 1)
            out.append(torch.arange(num_classes, device=ref.device, dtype=ref.dtype))
        return out

    if torch.is_tensor(utility_values_list):
        uv = utility_values_list.to(device=ref.device, dtype=ref.dtype)
        if uv.ndim == 1:
            return [uv] * m
        if uv.ndim == 2 and uv.shape[0] == m:
            return [uv[i] for i in range(m)]
        raise ValueError(
            "utility_values_list tensor must be [K] or [m, K]. "
            f"Got shape={tuple(uv.shape)}."
        )

    if isinstance(utility_values_list, (list, tuple)):
        if len(utility_values_list) == 0:
            raise ValueError("utility_values_list is empty.")
        if all(not isinstance(v, (list, tuple)) and not torch.is_tensor(v) for v in utility_values_list):
            uv = torch.as_tensor(utility_values_list, device=ref.device, dtype=ref.dtype)
            return [uv] * m
        if len(utility_values_list) != m:
            raise ValueError(f"utility_values_list must have length {m}. Got {len(utility_values_list)}.")
        return [
            torch.as_tensor(v, device=ref.device, dtype=ref.dtype)
            for v in utility_values_list
        ]

    raise TypeError(f"Unsupported utility_values_list: {type(utility_values_list)}.")


def ordinal_latent_samples_to_expected_utility(
    samples: Tensor,
    *,
    model: Model,
    utility_values_list=None,
    ordinal_likelihoods=None,
    latent_to_probs_list: Optional[Sequence[Optional[Callable[[Tensor], Tensor]]]] = None,
    objective_signs: Optional[Sequence[float] | Tensor] = None,
    eps: float = 1e-12,
) -> Tensor:
    """
    latent ordinal samples を expected utility objective samples に変換する。

    Args:
        samples:
            sample_shape x batch_shape x q x m または ... x q x m。
            最後の次元を output dimension とみなす。
    """
    likes = _extract_ordinal_likelihoods(model, ordinal_likelihoods)
    m = len(likes)

    if samples.shape[-1] != m:
        if m == 1 and samples.ndim >= 1:
            samples = samples.unsqueeze(-1)
        else:
            raise RuntimeError(
                "samples last dimension must match number of ordinal likelihoods. "
                f"samples.shape={tuple(samples.shape)}, m={m}."
            )

    if latent_to_probs_list is None:
        latent_to_probs_list = [None] * m
    elif len(latent_to_probs_list) != m:
        raise ValueError(f"latent_to_probs_list must have length {m}.")

    utility_list = _utility_values_tensor(
        utility_values_list,
        ref=samples,
        m=m,
        ordinal_likelihoods=likes,
    )

    cols = []
    for i in range(m):
        probs_i = _latent_to_probs(
            samples[..., i],
            ordinal_likelihood=likes[i],
            latent_to_probs=latent_to_probs_list[i],
            eps=eps,
        )
        u = utility_list[i].to(device=samples.device, dtype=samples.dtype).view(
            *((1,) * (probs_i.ndim - 1)),
            -1,
        )
        if probs_i.shape[-1] != u.shape[-1]:
            raise RuntimeError(
                f"utility_values length mismatch for output {i}: "
                f"probs classes={probs_i.shape[-1]}, utility length={u.shape[-1]}."
            )
        cols.append((probs_i * u).sum(dim=-1))

    values = torch.stack(cols, dim=-1)

    if objective_signs is not None:
        signs = torch.as_tensor(
            objective_signs,
            device=values.device,
            dtype=values.dtype,
        ).reshape(-1)
        if signs.numel() != m:
            raise ValueError(f"objective_signs must have length {m}. Got {signs.numel()}.")
        values = values * signs.view(*((1,) * (values.ndim - 1)), m)

    return values


def compute_hetero_multi_output_ordinal_train_y(
    model: Model,
    train_X: Tensor,
    *,
    utility_values_list=None,
    objective_signs: Optional[Sequence[float] | Tensor] = None,
    noise_penalty: float | Sequence[float] | Tensor = 0.0,
    variance_scale: float | Sequence[float] | Tensor = 1.0,
    tau: float | Sequence[float] | Tensor = 1e-6,
    default_sigma: float | Sequence[float] | Tensor = 0.0,
    eps: float = 1e-12,
) -> Tensor:
    """
    observed / baseline 用の hetero-adjusted utility 目的値を返す。

    既存の hetero summary が返す robust_mean を使うため、
    ordinal label そのものではなく acquisition と同じ utility 空間になる。
    """
    with torch.no_grad():
        X = _ensure_q_batch(train_X)
        summary = stack_multi_summaries(
            model,
            X,
            utility_values_list=utility_values_list,
            noise_penalties=noise_penalty,
            variance_scales=variance_scale,
            taus=tau,
            default_sigmas=default_sigma,
            eps=eps,
        )
        y = summary["robust_mean"]
        if objective_signs is not None:
            signs = torch.as_tensor(objective_signs, device=y.device, dtype=y.dtype).reshape(-1)
            y = y * signs.view(*((1,) * (y.ndim - 1)), -1)
        return y.reshape(-1, y.shape[-1])


# =========================================================
# Score objective for pointwise normal acquisitions
# =========================================================
class qHeteroMultiOutputOrdinalNormalScoreObjective(torch.nn.Module):
    """
    multi-output ordinal BO 系の計算済み pointwise score に作用する objective。

    EHVI / NEHVI 用の MCMultiOutputObjective ではなく、
    ExpectedUtility / PI / EI / UCB などの scalar score に作用する。
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        weight: float = 1.0,
        sign: float = 1.0,
        aggregated_risk_mode: Literal["ignore", "error"] = "ignore",
    ) -> None:
        super().__init__()
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.weight = float(weight)
        self.sign = float(sign)
        self.aggregated_risk_mode = aggregated_risk_mode

        if self.n_w is not None and self.n_w <= 0:
            raise ValueError("n_w must be positive or None.")
        if self.risk_type not in (None, "var", "cvar"):
            raise ValueError(f"Unknown risk_type: {self.risk_type!r}.")
        if self.risk_type is not None and self.n_w is None:
            raise ValueError("risk_type is specified, but n_w is None.")
        if self.risk_type is not None and not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        if self.aggregated_risk_mode not in ("ignore", "error"):
            raise ValueError("aggregated_risk_mode must be 'ignore' or 'error'.")

    @staticmethod
    def _ensure_q_batch(X: Tensor) -> Tensor:
        return X if X.dim() > 2 else X.unsqueeze(0)

    def _is_aggregated_score(self, score: Tensor, X: Optional[Tensor]) -> bool:
        if X is None or score.ndim == 0:
            return False
        Xq = self._ensure_q_batch(X)
        return tuple(score.shape) == tuple(Xq.shape[:-2])

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(score):
            raise TypeError(f"score must be a Tensor. Got {type(score)}.")

        score = score * self.sign * self.weight

        if score.ndim == 0:
            return score
        if self.n_w is None or self.n_w <= 1:
            return score

        if self._is_aggregated_score(score, X):
            if self.aggregated_risk_mode == "error":
                raise RuntimeError(
                    "qHeteroMultiOutputOrdinalNormalScoreObjective received an aggregated score. "
                    "InputPerturbation aggregation requires pointwise score."
                )
            return score

        q_like = int(score.shape[-1])
        if q_like % int(self.n_w) != 0:
            raise RuntimeError(f"score last dim must be divisible by n_w={self.n_w}.")
        q = q_like // int(self.n_w)
        score_w = score.reshape(*score.shape[:-1], q, int(self.n_w))

        if self.risk_type is None:
            return score_w.mean(dim=-1)

        descending = not self.maximize
        sorted_score = torch.sort(score_w, dim=-1, descending=descending).values
        k = max(1, int(math.ceil(int(self.n_w) * self.alpha)))
        tail = sorted_score[..., :k]

        if self.risk_type == "var":
            return tail[..., -1]
        if self.risk_type == "cvar":
            return tail.mean(dim=-1)
        raise ValueError(f"Unknown risk_type: {self.risk_type!r}.")


# =========================================================
# Hetero MC objective for EHVI / NEHVI / NParEGO
# =========================================================
class qHeteroMultiOutputOrdinalUtilityObjective(MCMultiOutputObjective):
    """
    BoTorch qEHVI / qNEHVI 内で使う hetero-adjusted ordinal utility objective。

    posterior latent samples を expected utility に変換し、必要に応じて
    `robust mean + beta * (sample utility - robust mean)` として hetero 調整し、
    noise penalty を引く。
    """

    def __init__(
        self,
        *,
        model: Model,
        utility_values_list=None,
        ordinal_likelihoods=None,
        latent_to_probs_list: Optional[Sequence[Optional[Callable[[Tensor], Tensor]]]] = None,
        base_objective: Optional[MCMultiOutputObjective] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        beta: float = 1.0,
        noise_penalty: float | Sequence[float] | Tensor = 0.3,
        variance_scale: float | Sequence[float] | Tensor = 1.0,
        tau: float | Sequence[float] | Tensor = 1e-6,
        default_sigma: float | Sequence[float] | Tensor = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.model = model
        self.utility_values_list = utility_values_list
        self.ordinal_likelihoods = ordinal_likelihoods
        self.latent_to_probs_list = latent_to_probs_list
        self.base_objective = base_objective
        self.objective_signs = objective_signs
        self.beta = float(beta)
        self.noise_penalty = noise_penalty
        self.variance_scale = variance_scale
        self.tau = tau
        self.default_sigma = default_sigma
        self.eps = float(eps)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if X is None:
            raise ValueError("X must be provided for qHeteroMultiOutputOrdinalUtilityObjective.")

        Xq = _ensure_q_batch(X)
        likes = _extract_ordinal_likelihoods(self.model, self.ordinal_likelihoods)
        m = len(likes)

        utilities = ordinal_latent_samples_to_expected_utility(
            samples,
            model=self.model,
            utility_values_list=self.utility_values_list,
            ordinal_likelihoods=likes,
            latent_to_probs_list=self.latent_to_probs_list,
            objective_signs=self.objective_signs,
            eps=self.eps,
        )
        utilities = _align_mo_samples_to_X(
            utilities,
            Xq,
            m=m,
            name="qHeteroMultiOutputOrdinalUtilityObjective.utilities",
        )

        with torch.no_grad():
            summary = stack_multi_summaries(
                self.model,
                Xq,
                utility_values_list=self.utility_values_list,
                noise_penalties=0.0,
                variance_scales=self.variance_scale,
                taus=self.tau,
                default_sigmas=self.default_sigma,
                eps=self.eps,
            )
            robust_mean = _align_pointwise_to_X_q_m(
                summary["robust_mean"],
                Xq,
                m=m,
                name="qHeteroMultiOutputOrdinalUtilityObjective.robust_mean",
            )
            sigma = summary.get("sigma", summary.get("total_std", None))
            if sigma is not None:
                sigma = _align_pointwise_to_X_q_m(
                    sigma,
                    Xq,
                    m=m,
                    name="qHeteroMultiOutputOrdinalUtilityObjective.sigma",
                )

            if self.objective_signs is not None:
                signs = torch.as_tensor(
                    self.objective_signs,
                    device=robust_mean.device,
                    dtype=robust_mean.dtype,
                ).reshape(-1)
                if signs.numel() != m:
                    raise ValueError(f"objective_signs must have length {m}. Got {signs.numel()}.")
                robust_mean = robust_mean * signs.view(*((1,) * (robust_mean.ndim - 1)), m)

        # robust_mean / sigma: batch_shape x q x m
        # utilities: sample_shape x batch_shape x q x m
        adjusted = robust_mean.unsqueeze(0) + self.beta * (utilities - robust_mean.unsqueeze(0))

        if sigma is not None:
            penalties = _expand_scalar_or_list(
                self.noise_penalty,
                m,
                "noise_penalty",
            )
            penalties = torch.as_tensor(
                penalties,
                device=utilities.device,
                dtype=utilities.dtype,
            ).reshape(-1)
            adjusted = adjusted - sigma.unsqueeze(0) * penalties.view(
                *((1,) * (adjusted.ndim - 1)),
                m,
            )

        adjusted = _align_mo_samples_to_X(
            adjusted,
            Xq,
            m=m,
            name="qHeteroMultiOutputOrdinalUtilityObjective.adjusted",
        )

        if self.base_objective is None:
            return adjusted

        # IdentityMCMultiOutputObjective and other BoTorch MO objectives require
        # adjusted.shape[-2] == X.shape[-2].  The alignment above protects against
        # custom posteriors that return sample / q dimensions in a different order.
        return self.base_objective(adjusted, X=Xq)


# =========================================================
# Base for scalar / normal acquisitions
# =========================================================
class _BaseHeteroMultiOutputOrdinalNormalAcquisition(AcquisitionFunction):
    """
    ExpectedUtility / PI / EI / UCB などの scalar BO acquisition base。

    Standard order:
        per-output pointwise score
        -> output aggregation
        -> pending penalty
        -> objective
        -> q reduction
    """

    def __init__(
        self,
        model: Model,
        *,
        utility_values_list: Optional[Sequence[Optional[Sequence[float] | Tensor]]] = None,
        objective_weights: Optional[Sequence[float] | Tensor] = None,
        noise_penalty: float | Sequence[float] | Tensor = 0.0,
        variance_scale: float | Sequence[float] | Tensor = 1.0,
        tau: float | Sequence[float] | Tensor = 1e-6,
        default_sigma: float | Sequence[float] | Tensor = 0.0,
        reduction: ReductionType = "max",
        # backward-compatible aliases
        reduce: Optional[str] = None,
        output_mode: OutputMode = "weighted_mean",
        aggregate: Optional[str] = None,
        eps: float = 1e-12,
        # pending penalty
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        self.submodels = list(getattr(model, "models", []))
        self.m = len(self.submodels) if len(self.submodels) > 0 else 1

        if reduce is not None:
            reduction = str(reduce)
        if aggregate is not None:
            output_mode = str(aggregate)

        if reduction not in ("mean", "sum", "max"):
            raise ValueError("reduction must be 'mean', 'sum', or 'max'.")
        if output_mode == "weighted_sum":
            output_mode = "weighted_mean"
        if output_mode not in ("mean", "sum", "max", "min", "weighted_mean", "product"):
            raise ValueError("output_mode must be mean/sum/max/min/weighted_mean/product.")

        self.utility_values_list = utility_values_list
        self.objective_weights = objective_weights
        self.noise_penalty = noise_penalty
        self.variance_scale = variance_scale
        self.tau = tau
        self.default_sigma = default_sigma
        self.reduction = reduction
        self.output_mode = output_mode
        self.eps = float(eps)
        self.objective = objective

        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)

        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def _set_eval_mode(self) -> None:
        self.model.eval()
        for sm in self.submodels:
            sm.eval()
            lik = getattr(sm, "likelihood", None)
            if lik is not None and hasattr(lik, "eval"):
                lik.eval()
            olik = getattr(sm, "ordinal_likelihood", None)
            if olik is not None and hasattr(olik, "eval"):
                olik.eval()

    def _apply_input_transform(self, X: Tensor) -> Tensor:
        X = _ensure_q_batch(X)

        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch(Xt)

        if self.submodels:
            it = getattr(self.submodels[0], "input_transform", None)
            if it is not None:
                Xt = it(X)
                if isinstance(Xt, tuple):
                    Xt = Xt[0]
                return _ensure_q_batch(Xt)

        return X

    def _transform_reference_like_candidate(self, X_ref, *, ref: Tensor) -> Optional[Tensor]:
        Xr = _coerce_reference_to_tensor(X_ref, ref=ref)
        if Xr is None or Xr.numel() == 0:
            return None
        Xr_t = self._apply_input_transform(_ensure_q_batch(Xr))
        return Xr_t.to(device=ref.device, dtype=ref.dtype)

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)

        if self.pending_penalty_weight <= 0.0:
            return Xt.new_zeros(Xt.shape[:-1])

        Xp_t = self._transform_reference_like_candidate(self.X_pending, ref=Xt)
        if Xp_t is None or Xp_t.numel() == 0:
            return Xt.new_zeros(Xt.shape[:-1])

        d = Xt.shape[-1]
        X2d = Xt.reshape(-1, d)
        Xp2d = Xp_t.reshape(-1, Xp_t.shape[-1])

        if Xp2d.shape[-1] != d:
            raise RuntimeError(
                "X_pending feature dimension mismatch after transform: "
                f"Xt.shape={tuple(Xt.shape)}, X_pending_transformed.shape={tuple(Xp_t.shape)}."
            )

        dist = torch.cdist(X2d, Xp2d).min(dim=-1).values.reshape(*Xt.shape[:-1])
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist)

    def _summary(self, X: Tensor) -> dict[str, Tensor]:
        return stack_multi_summaries(
            self.model,
            X,
            utility_values_list=self.utility_values_list,
            noise_penalties=self.noise_penalty,
            variance_scales=self.variance_scale,
            taus=self.tau,
            default_sigmas=self.default_sigma,
            eps=self.eps,
        )

    def _weights(self, ref: Tensor) -> Optional[Tensor]:
        return make_weight_tensor(self.objective_weights, ref=ref, m=self.m)

    def _aggregate_outputs(self, values: Tensor) -> Tensor:
        weights = self._weights(values)
        return aggregate_objectives(values, method=self.output_mode, weights=weights)

    def _threshold_vector(
        self,
        values: float | Sequence[float] | Tensor,
        *,
        ref: Tensor,
        name: str,
    ) -> Tensor:
        expanded = _expand_scalar_or_list(values, self.m, name)
        return torch.as_tensor(expanded, device=ref.device, dtype=ref.dtype).view(
            *((1,) * (ref.ndim - 1)),
            self.m,
        )

    def _finalize_pointwise_score(
        self,
        score_per_output: Tensor,
        X: Tensor,
        *,
        summary: dict[str, Tensor],
        name: str,
    ) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)

        if score_per_output.shape[-1] != self.m:
            raise RuntimeError(
                f"{name}: expected score_per_output last dim {self.m}, "
                f"got shape={tuple(score_per_output.shape)}."
            )

        aligned_cols = []
        for i in range(self.m):
            si = _align_pointwise_score_to_X(
                score_per_output[..., i],
                Xt,
                name=f"{name} output {i} score",
                reduce_extra="mean",
            )
            aligned_cols.append(si)
        score_per_output = torch.stack(aligned_cols, dim=-1)

        score = self._aggregate_outputs(score_per_output)
        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} aggregated score",
            reduce_extra="mean",
        )

        score = score - self._pending_penalty_per_point(Xt)

        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name=f"{name} score before objective",
            reduce_extra="sum",
        )
        score = _apply_score_objective(self, score, raw_X, name)

        out = _reduce_q(score, self.reduction)
        if out.shape != original_batch_shape:
            if out.numel() == int(torch.tensor(original_batch_shape).prod().item()):
                out = out.reshape(original_batch_shape)
            else:
                raise RuntimeError(
                    f"{name}: output shape mismatch. "
                    f"Expected {tuple(original_batch_shape)}, got {tuple(out.shape)}."
                )
        return out


# =========================================================
# Public scalar acquisition classes
# =========================================================
class qHeteroMultiOutputOrdinalExpectedUtility(_BaseHeteroMultiOutputOrdinalNormalAcquisition):
    """heteroscedastic multi-output ordinal expected utility acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)
        return self._finalize_pointwise_score(
            summary["robust_mean"],
            raw_X,
            summary=summary,
            name="qHeteroMultiOutputOrdinalExpectedUtility",
        )


class qHeteroMultiOutputOrdinalProbabilityOfImprovement(
    _BaseHeteroMultiOutputOrdinalNormalAcquisition
):
    """heteroscedastic multi-output ordinal probability of improvement acquisition."""

    def __init__(self, model: Model, best_f: float | Sequence[float] | Tensor, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.best_f = best_f

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)

        best_f = self._threshold_vector(
            self.best_f,
            ref=summary["robust_mean"],
            name="best_f",
        )
        z = (summary["robust_mean"] - best_f) / summary["total_std"].clamp_min(self.eps)
        per_obj = _normal_cdf(z)
        return self._finalize_pointwise_score(
            per_obj,
            raw_X,
            summary=summary,
            name="qHeteroMultiOutputOrdinalProbabilityOfImprovement",
        )


class qHeteroMultiOutputOrdinalExpectedImprovement(
    _BaseHeteroMultiOutputOrdinalNormalAcquisition
):
    """heteroscedastic multi-output ordinal expected improvement acquisition."""

    def __init__(self, model: Model, best_f: float | Sequence[float] | Tensor, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.best_f = best_f

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)

        best_f = self._threshold_vector(
            self.best_f,
            ref=summary["robust_mean"],
            name="best_f",
        )
        std = summary["total_std"].clamp_min(self.eps)
        z = (summary["robust_mean"] - best_f) / std
        per_obj = (summary["robust_mean"] - best_f) * _normal_cdf(z) + std * _normal_pdf(z)
        return self._finalize_pointwise_score(
            per_obj,
            raw_X,
            summary=summary,
            name="qHeteroMultiOutputOrdinalExpectedImprovement",
        )


class qHeteroMultiOutputOrdinalUpperConfidenceBound(
    _BaseHeteroMultiOutputOrdinalNormalAcquisition
):
    """heteroscedastic multi-output ordinal upper confidence bound acquisition."""

    def __init__(
        self,
        model: Model,
        beta: float | Sequence[float] | Tensor = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.beta = beta

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)

        beta = self._threshold_vector(
            self.beta,
            ref=summary["robust_mean"],
            name="beta",
        ).clamp_min(0.0)
        per_obj = summary["robust_mean"] + beta.sqrt() * summary["total_std"]
        return self._finalize_pointwise_score(
            per_obj,
            raw_X,
            summary=summary,
            name="qHeteroMultiOutputOrdinalUpperConfidenceBound",
        )


# =========================================================
# Multi-objective EHVI / NEHVI
# =========================================================
class qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement(
    qExpectedHypervolumeImprovement
):
    """heteroscedastic multi-output ordinal qEHVI acquisition."""

    def __init__(
        self,
        model: Model,
        ref_point: Union[Tensor, list[float]],
        partitioning: FastNondominatedPartitioning,
        *,
        utility_values_list=None,
        ordinal_likelihoods=None,
        latent_to_probs_list: Optional[Sequence[Optional[Callable[[Tensor], Tensor]]]] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        beta: float = 1.0,
        noise_penalty: float | Sequence[float] | Tensor = 0.3,
        variance_scale: float | Sequence[float] | Tensor = 1.0,
        tau: float | Sequence[float] | Tensor = 1e-6,
        default_sigma: float | Sequence[float] | Tensor = 0.0,
        eps: float = 1e-12,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
    ) -> None:
        base_objective = objective or IdentityMCMultiOutputObjective()
        hetero_objective = qHeteroMultiOutputOrdinalUtilityObjective(
            base_objective=base_objective,
            model=model,
            utility_values_list=utility_values_list,
            ordinal_likelihoods=ordinal_likelihoods,
            latent_to_probs_list=latent_to_probs_list,
            objective_signs=objective_signs,
            beta=beta,
            noise_penalty=noise_penalty,
            variance_scale=variance_scale,
            tau=tau,
            default_sigma=default_sigma,
            eps=eps,
        )
        constraints_arg = None if constraints is None or len(constraints) == 0 else constraints

        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=hetero_objective,
            constraints=constraints_arg,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )
        self.base_objective = base_objective
        self.hetero_objective = hetero_objective
        self.constraints = constraints_arg
        self.eta = eta
        self.fat = fat

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xq = _ensure_q_batch(X)
        X_eval = _maybe_concat_pending_points(self, Xq)
        posterior = self.model.posterior(X_eval)
        samples = self.get_posterior_samples(posterior)

        # Call the undecorated core computation directly.  This avoids the
        # parent forward decorator assertion before we can remove MC sample dims.
        out = self._compute_qehvi(samples=samples, X=X_eval)
        return _finalize_acq_output_to_batch(
            out,
            Xq,
            name="qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement",
        )


class qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
    qExpectedHypervolumeImprovement
):
    """heteroscedastic multi-output ordinal noisy-EHVI style acquisition.

    Notes:
        BoTorch の qNoisyExpectedHypervolumeImprovement は baseline 側の
        box decomposition を内部キャッシュする。このとき hetero ordinal の
        sample-dependent objective をそのまま使うと、MC sample dim が
        partitioning の batch dim として残り、candidate 側の sample dim と
        二重に解釈されることがある。

        この実装では、X_baseline から deterministic な hetero-adjusted utility
        baseline を作り、FastNondominatedPartitioning を明示的に構成してから
        qExpectedHypervolumeImprovement として評価する。厳密な qNEHVI ではないが、
        実用上は「noisy baseline を utility 空間で固定した安定版」として扱える。
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        X_baseline: Tensor,
        *,
        utility_values_list=None,
        ordinal_likelihoods=None,
        latent_to_probs_list: Optional[Sequence[Optional[Callable[[Tensor], Tensor]]]] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        beta: float = 1.0,
        noise_penalty: float | Sequence[float] | Tensor = 0.3,
        variance_scale: float | Sequence[float] | Tensor = 1.0,
        tau: float | Sequence[float] | Tensor = 1e-6,
        default_sigma: float | Sequence[float] | Tensor = 0.0,
        eps: float = 1e-12,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
        partitioning: Optional[FastNondominatedPartitioning] = None,
        Y_baseline: Optional[Tensor] = None,
        # qNoisyExpectedHypervolumeImprovement 互換の未使用引数。
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = True,
        marginalize_dim: Optional[int] = None,
    ) -> None:
        base_objective = objective or IdentityMCMultiOutputObjective()
        hetero_objective = qHeteroMultiOutputOrdinalUtilityObjective(
            base_objective=base_objective,
            model=model,
            utility_values_list=utility_values_list,
            ordinal_likelihoods=ordinal_likelihoods,
            latent_to_probs_list=latent_to_probs_list,
            objective_signs=objective_signs,
            beta=beta,
            noise_penalty=noise_penalty,
            variance_scale=variance_scale,
            tau=tau,
            default_sigma=default_sigma,
            eps=eps,
        )
        constraints_arg = None if constraints is None or len(constraints) == 0 else constraints

        if Y_baseline is None:
            Y_baseline = compute_hetero_multi_output_ordinal_train_y(
                model,
                X_baseline,
                utility_values_list=utility_values_list,
                objective_signs=objective_signs,
                noise_penalty=noise_penalty,
                variance_scale=variance_scale,
                tau=tau,
                default_sigma=default_sigma,
                eps=eps,
            )

        ref_point_t = torch.as_tensor(
            ref_point,
            device=Y_baseline.device,
            dtype=Y_baseline.dtype,
        ).reshape(-1)

        if Y_baseline.ndim != 2:
            Y_baseline = Y_baseline.reshape(-1, Y_baseline.shape[-1])
        if Y_baseline.shape[-1] != ref_point_t.numel():
            raise RuntimeError(
                "Y_baseline and ref_point dimension mismatch. "
                f"Y_baseline.shape={tuple(Y_baseline.shape)}, "
                f"ref_point.shape={tuple(ref_point_t.shape)}."
            )

        if partitioning is None:
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_point_t,
                Y=Y_baseline,
            )

        super().__init__(
            model=model,
            ref_point=ref_point_t,
            partitioning=partitioning,
            sampler=sampler,
            objective=hetero_objective,
            constraints=constraints_arg,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )

        self.base_objective = base_objective
        self.hetero_objective = hetero_objective
        self.constraints = constraints_arg
        self.eta = eta
        self.fat = fat
        self.X_baseline = X_baseline
        self.Y_baseline = Y_baseline
        self.is_deterministic_baseline_nehvi = True

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xq = _ensure_q_batch(X)
        X_eval = _maybe_concat_pending_points(self, Xq)
        posterior = self.model.posterior(X_eval)
        samples = self.get_posterior_samples(posterior)

        # Stable deterministic-baseline NEHVI uses qEHVI's box decomposition.
        # Do not call super().forward because its decorator asserts shape before
        # our custom objective output can be reduced to t-batch shape.
        out = self._compute_qehvi(samples=samples, X=X_eval)
        return _finalize_acq_output_to_batch(
            out,
            Xq,
            name="qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
        )


class qHeteroMultiOutputOrdinalNParEGO(MCAcquisitionFunction):
    """heteroscedastic multi-output ordinal qNParEGO acquisition.

    Notes:
        MCAcquisitionFunction requires an objective for multi-output models.
        Therefore this class passes the hetero ordinal utility objective to
        ``super().__init__`` and reuses ``self.objective`` in ``forward``.
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Optional[Tensor] = None,
        *,
        utility_values_list=None,
        ordinal_likelihoods=None,
        latent_to_probs_list: Optional[Sequence[Optional[Callable[[Tensor], Tensor]]]] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        weights: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        beta: float = 1.0,
        noise_penalty: float | Sequence[float] | Tensor = 0.3,
        variance_scale: float | Sequence[float] | Tensor = 1.0,
        tau: float | Sequence[float] | Tensor = 1e-6,
        default_sigma: float | Sequence[float] | Tensor = 0.0,
        eps: float = 1e-12,
        objective: Optional[MCMultiOutputObjective] = None,
        rho: float = 0.05,
    ) -> None:
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        # Infer output dimension from ref_point, objective_signs, model.models, or baseline utility.
        with torch.no_grad():
            y_baseline = compute_hetero_multi_output_ordinal_train_y(
                model,
                X_baseline,
                utility_values_list=utility_values_list,
                objective_signs=objective_signs,
                noise_penalty=noise_penalty,
                variance_scale=variance_scale,
                tau=tau,
                default_sigma=default_sigma,
                eps=eps,
            )

        if y_baseline.ndim != 2:
            y_baseline = y_baseline.reshape(-1, y_baseline.shape[-1])

        m = int(y_baseline.shape[-1])
        if ref_point is not None:
            m = int(torch.as_tensor(ref_point).numel())

        tkwargs = {"dtype": X_baseline.dtype, "device": X_baseline.device}
        if weights is None:
            w = torch.rand(m, **tkwargs)
            weights = w / w.sum().clamp_min(1e-12)
        else:
            weights = weights.to(**tkwargs).reshape(-1)
            if weights.numel() != m:
                raise ValueError(
                    f"weights must have length {m}, got {weights.numel()}."
                )
            weights = weights / weights.sum().clamp_min(1e-12)

        # This objective maps latent ordinal posterior samples to hetero-adjusted
        # expected utility samples with shape sample_shape x batch_shape x q x m.
        utility_objective = qHeteroMultiOutputOrdinalUtilityObjective(
            base_objective=objective,
            model=model,
            utility_values_list=utility_values_list,
            ordinal_likelihoods=ordinal_likelihoods,
            latent_to_probs_list=latent_to_probs_list,
            objective_signs=objective_signs,
            beta=beta,
            noise_penalty=noise_penalty,
            variance_scale=variance_scale,
            tau=tau,
            default_sigma=default_sigma,
            eps=eps,
        )

        # IMPORTANT:
        # For multi-output models, MCAcquisitionFunction raises UnsupportedError
        # if objective is None. Pass utility_objective here.
        super().__init__(model=model, sampler=sampler, objective=utility_objective)

        self.register_buffer("weights", weights)
        self.rho = float(rho)
        self.utility_objective = utility_objective
        self.base_objective = objective

        scalarized = self._scalarize(y_baseline.to(**tkwargs))
        self.register_buffer("best_value", scalarized.max())

    def _scalarize(self, Y: Tensor) -> Tensor:
        w = self.weights.to(device=Y.device, dtype=Y.dtype)
        # augmented Chebyshev scalarization for maximization.
        weighted = w.view(*((1,) * (Y.ndim - 1)), -1) * Y
        return weighted.min(dim=-1).values + self.rho * weighted.sum(dim=-1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xq = _ensure_q_batch(X)
        posterior = self.model.posterior(Xq)
        samples = self.get_posterior_samples(posterior)

        # self.objective is qHeteroMultiOutputOrdinalUtilityObjective.
        Y = self.objective(samples, X=Xq)
        Y = _align_mo_samples_to_X(
            Y,
            Xq,
            m=int(self.weights.numel()),
            name="qHeteroMultiOutputOrdinalNParEGO.Y",
        )

        scalarized = self._scalarize(Y)
        best_q = scalarized.max(dim=-1).values

        # Average over MC sample dimensions, but preserve t-batch shape.
        value = (best_q - self.best_value.to(best_q)).clamp_min(0.0)
        out = value.mean(dim=0) if value.ndim > 0 else value
        return _finalize_acq_output_to_batch(
            out,
            Xq,
            name="qHeteroMultiOutputOrdinalNParEGO",
        )


__all__ = [
    "qHeteroMultiOutputOrdinalNormalScoreObjective",
    "qHeteroMultiOutputOrdinalUtilityObjective",
    "ordinal_latent_samples_to_expected_utility",
    "compute_hetero_multi_output_ordinal_train_y",
    "qHeteroMultiOutputOrdinalExpectedUtility",
    "qHeteroMultiOutputOrdinalProbabilityOfImprovement",
    "qHeteroMultiOutputOrdinalExpectedImprovement",
    "qHeteroMultiOutputOrdinalUpperConfidenceBound",
    "qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement",
    "qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputOrdinalNParEGO",
]
