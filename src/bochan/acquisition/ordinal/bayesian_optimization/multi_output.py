
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import t_batch_mode_transform


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum"]
MultiOutputMode = Literal[
    "mean",
    "sum",
    "max",
    "min",
    "weighted_mean",
    "all_feasible",
]
OrdinalFeasibilityMode = Literal[
    "class_ge",
    "class_le",
    "class_interval",
    "expected_utility_ge",
]
LinkType = Literal["auto", "probit", "logit"]


# =========================================================
# Basic tensor / shape helpers
# =========================================================
def ensure_q_batch(X: Tensor) -> Tensor:
    """X を (..., q, d) に揃える。"""
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _as_2d_train_y(train_Y: Tensor) -> Tensor:
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2:
        raise ValueError(f"train_Y must be [n] or [n, m], got {tuple(train_Y.shape)}.")
    return train_Y


def _to_utility_list(
    utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[Tensor]:
    if isinstance(utility_values, Tensor):
        uv = utility_values.to(device=device, dtype=dtype)
        if uv.ndim == 1:
            return [uv]
        if uv.ndim == 2:
            return [uv[i] for i in range(uv.shape[0])]
        raise ValueError("utility_values tensor must be [K] or [m, K].")

    if len(utility_values) == 0:
        raise ValueError("utility_values must not be empty.")

    first = utility_values[0]  # type: ignore[index]
    if isinstance(first, (int, float)):
        return [torch.as_tensor(utility_values, device=device, dtype=dtype)]  # type: ignore[arg-type]

    return [torch.as_tensor(v, device=device, dtype=dtype) for v in utility_values]  # type: ignore[arg-type]


def _normalize_objective_signs(
    objective_signs: Optional[Sequence[float] | Tensor],
    *,
    m: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if objective_signs is None:
        return torch.ones(m, device=device, dtype=dtype)

    signs = torch.as_tensor(objective_signs, device=device, dtype=dtype).reshape(-1)
    if signs.shape != torch.Size([m]):
        raise ValueError(f"objective_signs must have shape [{m}], got {tuple(signs.shape)}.")
    return signs


def _to_index_vector(
    value: Optional[int | Sequence[int] | Tensor],
    *,
    m: int,
    name: str,
    default: Optional[int] = None,
) -> Optional[list[int]]:
    if value is None:
        if default is None:
            return None
        return [int(default)] * m

    if isinstance(value, int):
        return [int(value)] * m

    vals = torch.as_tensor(value).reshape(-1).tolist()
    if len(vals) != m:
        raise ValueError(f"{name} must be scalar or length {m}, got {len(vals)}.")
    return [int(v) for v in vals]


def _to_float_vector(
    value: Optional[float | Sequence[float] | Tensor],
    *,
    m: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    default: Optional[float] = None,
) -> Optional[Tensor]:
    if value is None:
        if default is None:
            return None
        return torch.full((m,), float(default), device=device, dtype=dtype)

    if isinstance(value, (float, int)):
        return torch.full((m,), float(value), device=device, dtype=dtype)

    vals = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if vals.numel() != m:
        raise ValueError(f"{name} must be scalar or length {m}, got {vals.numel()}.")
    return vals


# =========================================================
# Ordinal likelihood / utility helpers
# =========================================================
def _extract_ordinal_likelihoods(
    model: Model,
    ordinal_likelihoods: Optional[Sequence[nn.Module] | nn.Module] = None,
) -> list[nn.Module]:
    if ordinal_likelihoods is not None:
        if isinstance(ordinal_likelihoods, nn.Module):
            return [ordinal_likelihoods]
        return list(ordinal_likelihoods)

    if hasattr(model, "ordinal_likelihoods"):
        return list(getattr(model, "ordinal_likelihoods"))

    if hasattr(model, "models"):
        likes = []
        for sm in getattr(model, "models"):
            if hasattr(sm, "ordinal_likelihood"):
                likes.append(getattr(sm, "ordinal_likelihood"))
            elif hasattr(sm, "likelihood"):
                likes.append(getattr(sm, "likelihood"))
        if len(likes) > 0:
            return likes

    if hasattr(model, "ordinal_likelihood"):
        return [getattr(model, "ordinal_likelihood")]

    if hasattr(model, "likelihood"):
        return [getattr(model, "likelihood")]

    raise ValueError("Could not infer ordinal likelihoods.")


def _get_cutpoints(likelihood: nn.Module) -> Tensor:
    for name in ("cutpoints", "thresholds", "ordered_cutpoints", "transformed_cutpoints"):
        if hasattr(likelihood, name):
            value = getattr(likelihood, name)
            return torch.as_tensor(value() if callable(value) else value).reshape(-1)

    for name in ("raw_cutpoints", "raw_thresholds"):
        if hasattr(likelihood, name):
            return torch.sort(torch.as_tensor(getattr(likelihood, name)).reshape(-1)).values

    raise ValueError("Could not extract cutpoints from ordinal likelihood.")


def _cdf(x: Tensor, link: Literal["probit", "logit"]) -> Tensor:
    if link == "logit":
        return torch.sigmoid(x)
    if link == "probit":
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    raise ValueError(f"Unknown link: {link}")


def ordinal_probs_from_latent(
    latent: Tensor,
    likelihood: nn.Module,
    *,
    num_classes: int,
    link: LinkType = "auto",
    eps: float = 1e-12,
) -> Tensor:
    """latent f samples を ordinal class probability に変換する。"""
    if hasattr(likelihood, "class_probs_from_f"):
        probs = likelihood.class_probs_from_f(latent)
        if probs.shape[-1] != num_classes and probs.ndim >= 2:
            probs = likelihood.class_probs_from_f(latent.unsqueeze(-1))
        probs = torch.as_tensor(probs, device=latent.device, dtype=latent.dtype)
        probs = probs.clamp_min(eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    if hasattr(likelihood, "probs_from_latent"):
        probs = likelihood.probs_from_latent(latent)
        if probs.shape[-1] != num_classes and probs.ndim >= 2:
            probs = likelihood.probs_from_latent(latent.unsqueeze(-1))
        probs = torch.as_tensor(probs, device=latent.device, dtype=latent.dtype)
        probs = probs.clamp_min(eps)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    cutpoints = _get_cutpoints(likelihood).to(device=latent.device, dtype=latent.dtype)
    if link == "auto":
        link_name = str(getattr(likelihood, "link", "probit")).lower()
        link = "logit" if ("logit" in link_name or "logistic" in link_name) else "probit"

    z = cutpoints - latent.unsqueeze(-1)
    cdf = _cdf(z, link=link)

    first = cdf[..., :1]
    middle = cdf[..., 1:] - cdf[..., :-1]
    last = 1.0 - cdf[..., -1:]
    probs = torch.cat([first, middle, last], dim=-1)

    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _num_classes_from_likelihood_or_utilities(
    likelihood: nn.Module,
    utility_values: Optional[Tensor] = None,
) -> int:
    if utility_values is not None:
        return int(utility_values.numel())
    return int(_get_cutpoints(likelihood).numel() + 1)


def _reduce_extra_batch_dims(tensor: Tensor, X: Tensor, n_trailing_keep: int) -> Tensor:
    out = tensor
    Xq = ensure_q_batch(X)
    x_batch_shape = tuple(Xq.shape[:-2])
    target_ndim = len(x_batch_shape) + n_trailing_keep

    while out.ndim > target_ndim:
        prefix = tuple(out.shape[:-n_trailing_keep]) if n_trailing_keep > 0 else tuple(out.shape)

        if len(x_batch_shape) == 0:
            reduce_dim = 0
        else:
            match_start = None
            max_start = len(prefix) - len(x_batch_shape)
            for s in range(max_start + 1):
                if tuple(prefix[s : s + len(x_batch_shape)]) == x_batch_shape:
                    match_start = s
                    break

            if match_start is None:
                reduce_dim = max(out.ndim - n_trailing_keep - 1, 0)
            else:
                protected = set(range(match_start, match_start + len(x_batch_shape)))
                extra_dims = [i for i in range(len(prefix)) if i not in protected]
                if not extra_dims:
                    break
                reduce_dim = extra_dims[0]

        out = out.mean(dim=reduce_dim)

    return out


def _class_probs_from_model_or_likelihood(
    model: Model,
    X: Tensor,
    ordinal_likelihoods: list[nn.Module],
    *,
    utility_values_list: Optional[list[Tensor]] = None,
    link: LinkType = "auto",
    eps: float = 1e-12,
) -> Tensor:
    """
    multi-output ordinal class probability を stack して返す。

    Returns:
        Tensor with shape (*batch, q_like, m, K_max).
        出力ごとの class 数が異なる場合は K_max に合わせて 0 padding する。
    """
    Xq = ensure_q_batch(X)

    if hasattr(model, "class_probs"):
        probs = model.class_probs(Xq)
        # expected: (*batch, q, m, K) or list-like is not supported here.
        if torch.is_tensor(probs):
            return probs.clamp_min(eps) / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    posts = []
    if hasattr(model, "models"):
        for sm in getattr(model, "models"):
            posts.append(sm.posterior(Xq))
    else:
        posterior = model.posterior(Xq)
        if len(ordinal_likelihoods) == 1:
            posts.append(posterior)
        else:
            # model.posterior(X) should be multi-output. Use latent samples fallback below.
            posts = [posterior] * len(ordinal_likelihoods)

    per_output_probs = []
    for i, (post, lik) in enumerate(zip(posts, ordinal_likelihoods)):
        utility_values_i = None if utility_values_list is None else utility_values_list[i]
        num_classes = _num_classes_from_likelihood_or_utilities(lik, utility_values_i)

        if hasattr(lik, "marginal_class_probs"):
            probs_i = lik.marginal_class_probs(post.distribution)
        else:
            samples_i = post.rsample(torch.Size([128]))
            if samples_i.ndim >= 1 and samples_i.shape[-1] == 1:
                samples_i = samples_i.squeeze(-1)
            probs_i = ordinal_probs_from_latent(
                samples_i,
                lik,
                num_classes=num_classes,
                link=link,
                eps=eps,
            ).mean(dim=0)

        probs_i = _reduce_extra_batch_dims(probs_i, Xq, n_trailing_keep=2)
        probs_i = probs_i.clamp_min(eps)
        probs_i = probs_i / probs_i.sum(dim=-1, keepdim=True).clamp_min(eps)
        per_output_probs.append(probs_i)

    max_k = max(p.shape[-1] for p in per_output_probs)
    padded = []
    for p in per_output_probs:
        if p.shape[-1] < max_k:
            pad = p.new_zeros(*p.shape[:-1], max_k - p.shape[-1])
            p = torch.cat([p, pad], dim=-1)
        padded.append(p)

    return torch.stack(padded, dim=-2)  # (*batch, q_like, m, K_max)


def compute_observed_ordinal_utility(
    train_Y: Tensor,
    utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor,
    *,
    objective_signs: Optional[Sequence[float] | Tensor] = None,
    class_offset: int = 0,
) -> Tensor:
    """観測済み ordinal labels を multi-objective utility 値に変換する。"""
    train_Y = _as_2d_train_y(train_Y)
    device = train_Y.device
    dtype = torch.double if not train_Y.is_floating_point() else train_Y.dtype

    utilities = _to_utility_list(utility_values, device=device, dtype=dtype)
    m = len(utilities)

    if train_Y.shape[-1] != m:
        raise ValueError(f"train_Y outputs={train_Y.shape[-1]} but utility_values outputs={m}.")

    signs = _normalize_objective_signs(objective_signs, m=m, device=device, dtype=dtype)

    cols = []
    for i, uv in enumerate(utilities):
        idx = train_Y[:, i].long() - int(class_offset)
        if idx.min() < 0 or idx.max() >= uv.numel():
            raise ValueError(f"train_Y[:, {i}] contains class index outside utility range.")
        cols.append(uv[idx] * signs[i])

    return torch.stack(cols, dim=-1)


# =========================================================
# Input transform / pending penalty helpers
# =========================================================
def _coerce_pending_to_tensor(
    X_pending,
    *,
    ref: Optional[Tensor] = None,
) -> Optional[Tensor]:
    if X_pending is None:
        return None

    if torch.is_tensor(X_pending):
        out = X_pending
    elif isinstance(X_pending, (list, tuple)):
        tensors = []
        for item in X_pending:
            if item is None:
                continue
            t = _coerce_pending_to_tensor(item, ref=ref)
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
            f"Got {type(X_pending)}."
        )

    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out


def shape_X_for_model(model: Model, X: Tensor) -> Tensor:
    """candidate / pending を同じ距離計算空間へ写す。"""
    X = ensure_q_batch(X)

    it = getattr(model, "input_transform", None)
    if it is not None:
        Xt = it(X)
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return ensure_q_batch(Xt)

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        it = getattr(models[0], "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return ensure_q_batch(Xt)

    return X


def _transform_pending_like_candidate(
    model: Model,
    X_pending,
    *,
    ref: Tensor,
) -> Optional[Tensor]:
    Xp = _coerce_pending_to_tensor(X_pending, ref=ref)
    if Xp is None or Xp.numel() == 0:
        return None
    Xp_t = shape_X_for_model(model, ensure_q_batch(Xp))
    return Xp_t.to(device=ref.device, dtype=ref.dtype)


def _pending_penalty_per_point(
    model: Model,
    expanded_X: Tensor,
    X_pending: Optional[Tensor],
    *,
    weight: float,
    beta: float,
) -> Tensor:
    expanded_X = ensure_q_batch(expanded_X)

    if weight <= 0.0:
        return torch.zeros(expanded_X.shape[:-1], device=expanded_X.device, dtype=expanded_X.dtype)

    Xp_t = _transform_pending_like_candidate(model, X_pending, ref=expanded_X)
    if Xp_t is None or Xp_t.numel() == 0:
        return torch.zeros(expanded_X.shape[:-1], device=expanded_X.device, dtype=expanded_X.dtype)

    d = expanded_X.shape[-1]
    X2d = expanded_X.reshape(-1, d)
    Xp2d = Xp_t.reshape(-1, Xp_t.shape[-1])

    if Xp2d.shape[-1] != d:
        raise RuntimeError(
            "X_pending feature dimension mismatch after transform: "
            f"expanded_X.shape={tuple(expanded_X.shape)}, "
            f"X_pending_transformed.shape={tuple(Xp_t.shape)}."
        )

    dist = torch.cdist(X2d, Xp2d).min(dim=-1).values.reshape(*expanded_X.shape[:-1])
    return float(weight) * torch.exp(-float(beta) * dist)


def reduce_q(score: Tensor, reduction: ReductionType) -> Tensor:
    if score.shape[-1] == 1:
        return score.squeeze(-1)
    if reduction == "mean":
        return score.mean(dim=-1)
    if reduction == "sum":
        return score.sum(dim=-1)
    raise ValueError(f"Unknown reduction: {reduction}")


def aggregate_outputs(
    score_per_output: Tensor,
    *,
    output_mode: MultiOutputMode,
    output_weights: Optional[Tensor],
    eps: float,
) -> Tensor:
    """
    Args:
        score_per_output: (*batch, q_like, m)

    Returns:
        Tensor: (*batch, q_like)
    """
    if output_mode == "mean":
        return score_per_output.mean(dim=-1)

    if output_mode == "sum":
        return score_per_output.sum(dim=-1)

    if output_mode == "max":
        return score_per_output.max(dim=-1).values

    if output_mode == "min":
        return score_per_output.min(dim=-1).values

    if output_mode == "weighted_mean":
        if output_weights is None:
            raise ValueError("output_weights must be provided when output_mode='weighted_mean'.")
        w = output_weights.to(device=score_per_output.device, dtype=score_per_output.dtype).reshape(-1)
        if w.numel() != score_per_output.shape[-1]:
            raise ValueError(
                f"output_weights must have length {score_per_output.shape[-1]}, got {w.numel()}."
            )
        w = w / w.sum().clamp_min(eps)
        view_shape = (1,) * (score_per_output.ndim - 1) + (w.numel(),)
        return (score_per_output * w.view(*view_shape)).sum(dim=-1)

    if output_mode == "all_feasible":
        return score_per_output.clamp(eps, 1.0).prod(dim=-1)

    raise ValueError(f"Unknown output_mode: {output_mode}")


def apply_pointwise_score_objective(
    owner,
    score: Tensor,
    *,
    raw_X: Tensor,
    expanded_X: Tensor,
    name: str,
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    if isinstance(objective, MCMultiOutputObjective):
        score_in = score
        if score_in.ndim == expanded_X.ndim - 1:
            score_in = score_in.unsqueeze(-1)
        try:
            out = objective(score_in, X=raw_X)
        except TypeError:
            out = objective(score_in)
    else:
        try:
            out = objective(score, X=raw_X)
        except TypeError:
            out = objective(score)

    if not torch.is_tensor(out):
        raise TypeError(f"{name}: objective must return Tensor. Got {type(out)}.")

    if out.ndim == raw_X.ndim and out.shape[-1] == 1:
        out = out.squeeze(-1)

    return out


# =========================================================
# Multi-output ordinal utility objective for EHVI / NEHVI / NParEGO
# =========================================================
class qMultiOutputOrdinalUtilityObjective(MCMultiOutputObjective):
    """
    multi-output ordinal 用 MCMultiOutputObjective。

    posterior latent samples:
        sample_shape x batch_shape x q_like x m

    returns:
        sample_shape x batch_shape x q x m

    InputPerturbation ありでは input_perturbation_n_w により q_like=q*n_w を q に戻す。
    """

    def __init__(
        self,
        model: Model,
        utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor,
        *,
        ordinal_likelihoods: Optional[Sequence[nn.Module] | nn.Module] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        link: LinkType = "auto",
        input_perturbation_n_w: Optional[int] = None,
        risk_type: RiskType = None,
        risk_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.ordinal_likelihoods = _extract_ordinal_likelihoods(model, ordinal_likelihoods)
        self.link = link

        utility_list = _to_utility_list(
            utility_values,
            device=torch.device("cpu"),
            dtype=torch.double,
        )
        if len(utility_list) == 1 and len(self.ordinal_likelihoods) > 1:
            utility_list = utility_list * len(self.ordinal_likelihoods)
        if len(utility_list) != len(self.ordinal_likelihoods):
            raise ValueError("utility_values and ordinal_likelihoods length mismatch.")

        self.num_outputs = len(utility_list)

        max_k = max(u.numel() for u in utility_list)
        utility_table = torch.full((self.num_outputs, max_k), float("nan"), dtype=torch.double)
        num_classes = torch.empty(self.num_outputs, dtype=torch.long)

        for i, u in enumerate(utility_list):
            utility_table[i, : u.numel()] = u
            num_classes[i] = u.numel()

        self.register_buffer("utility_table", utility_table)
        self.register_buffer("num_classes", num_classes)

        signs = _normalize_objective_signs(
            objective_signs,
            m=self.num_outputs,
            device=torch.device("cpu"),
            dtype=torch.double,
        )
        self.register_buffer("objective_signs", signs)

        self.input_perturbation_n_w = (
            None if input_perturbation_n_w is None else int(input_perturbation_n_w)
        )
        self.risk_type = risk_type
        self.risk_alpha = float(risk_alpha)

        if self.risk_type not in (None, "var", "cvar"):
            raise ValueError(f"Unknown risk_type: {self.risk_type}.")
        if self.risk_type is not None and self.input_perturbation_n_w is None:
            raise ValueError("risk_type is specified but input_perturbation_n_w is None.")
        if self.risk_type is not None and not (0.0 < self.risk_alpha <= 1.0):
            raise ValueError("risk_alpha must be in (0, 1].")

    def _aggregate_input_perturbation(self, Y: Tensor, X: Optional[Tensor]) -> Tensor:
        n_w = self.input_perturbation_n_w
        if n_w is None or n_w <= 1:
            return Y

        q_like = Y.shape[-2]
        if X is not None:
            X_in = ensure_q_batch(X)
            q = X_in.shape[-2]
            if q_like == q:
                return Y
            if q_like != q * n_w:
                raise RuntimeError(f"Unexpected q_like={q_like}; expected q*n_w={q*n_w}.")
        else:
            if q_like % n_w != 0:
                raise RuntimeError(f"q_like={q_like} must be divisible by n_w={n_w}.")
            q = q_like // n_w

        Y_w = Y.reshape(*Y.shape[:-2], q, n_w, Y.shape[-1])

        if self.risk_type is None:
            return Y_w.mean(dim=-2)

        k_tail = max(1, int(math.ceil(n_w * self.risk_alpha)))
        sorted_Y = torch.sort(Y_w, dim=-2, descending=False).values
        tail = sorted_Y[..., :k_tail, :]

        if self.risk_type == "var":
            return tail[..., -1, :]
        if self.risk_type == "cvar":
            return tail.mean(dim=-2)

        raise ValueError(f"Unknown risk_type: {self.risk_type}.")

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if samples.shape[-1] != self.num_outputs:
            if self.num_outputs == 1:
                samples = samples.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Expected samples last dim {self.num_outputs}, got {tuple(samples.shape)}."
                )

        outs = []
        for i in range(self.num_outputs):
            k = int(self.num_classes[i].item())
            uv = self.utility_table[i, :k].to(device=samples.device, dtype=samples.dtype)
            latent_i = samples[..., i]

            probs_i = ordinal_probs_from_latent(
                latent_i,
                self.ordinal_likelihoods[i],
                num_classes=k,
                link=self.link,
            )
            util_i = (probs_i * uv).sum(dim=-1)
            util_i = util_i * self.objective_signs[i].to(
                device=samples.device,
                dtype=samples.dtype,
            )
            outs.append(util_i)

        Y = torch.stack(outs, dim=-1)
        return self._aggregate_input_perturbation(Y, X)


# =========================================================
# Probability of feasibility aligned with classification BO
# =========================================================
class qMultiOutputOrdinalProbabilityOfFeasibility(AcquisitionFunction):
    """
    multi-output ordinal Probability of Feasibility。

    Modes:
        - class_ge: P(y >= min_class)
        - class_le: P(y <= max_class)
        - class_interval: P(min_class <= y <= max_class)
        - expected_utility_ge: sigmoid((E[u(y)] - threshold) / tau)

    output_mode:
        - mean / sum / max / min / weighted_mean
        - all_feasible: product of per-output feasibility probabilities
    """

    def __init__(
        self,
        model: Model,
        ordinal_likelihoods: Optional[Sequence[nn.Module] | nn.Module] = None,
        mode: OrdinalFeasibilityMode = "class_ge",
        min_class: Optional[int | Sequence[int] | Tensor] = None,
        max_class: Optional[int | Sequence[int] | Tensor] = None,
        utility_values: Optional[Sequence[Sequence[float]] | Sequence[float] | Tensor] = None,
        utility_threshold: Optional[float | Sequence[float] | Tensor] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        link: LinkType = "auto",
        tau: float = 1e-3,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "all_feasible",
        output_weights: Optional[Tensor | Sequence[float]] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-8,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if output_mode not in ("mean", "sum", "max", "min", "weighted_mean", "all_feasible"):
            raise ValueError(
                "output_mode must be one of 'mean', 'sum', 'max', 'min', "
                "'weighted_mean', or 'all_feasible'."
            )

        self.ordinal_likelihoods = _extract_ordinal_likelihoods(model, ordinal_likelihoods)
        self.num_outputs = len(self.ordinal_likelihoods)

        device = torch.device("cpu")
        dtype = torch.double

        self.utility_values_list = (
            None if utility_values is None
            else _to_utility_list(utility_values, device=device, dtype=dtype)
        )
        if self.utility_values_list is not None:
            if len(self.utility_values_list) == 1 and self.num_outputs > 1:
                self.utility_values_list = self.utility_values_list * self.num_outputs
            if len(self.utility_values_list) != self.num_outputs:
                raise ValueError("utility_values and ordinal_likelihoods length mismatch.")

        self.objective_signs = _normalize_objective_signs(
            objective_signs,
            m=self.num_outputs,
            device=device,
            dtype=dtype,
        )

        self.mode = mode
        self.min_classes = _to_index_vector(
            min_class,
            m=self.num_outputs,
            name="min_class",
            default=None,
        )
        self.max_classes = _to_index_vector(
            max_class,
            m=self.num_outputs,
            name="max_class",
            default=None,
        )
        self.utility_threshold = utility_threshold

        self.link = link
        self.tau = float(tau)
        self.reduction = reduction
        self.output_mode = output_mode
        self.output_weights = (
            None if output_weights is None
            else torch.as_tensor(output_weights, dtype=torch.double).reshape(-1)
        )
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.eps = float(eps)
        self.objective = objective
        self.set_X_pending(None)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_pending_to_tensor(X_pending)

    def _pointwise_feasibility_per_output(self, X: Tensor) -> Tensor:
        probs = _class_probs_from_model_or_likelihood(
            self.model,
            X,
            self.ordinal_likelihoods,
            utility_values_list=self.utility_values_list,
            link=self.link,
            eps=self.eps,
        )  # (*batch, q_like, m, K_max)

        m = probs.shape[-2]
        if m != self.num_outputs:
            raise RuntimeError(f"Expected {self.num_outputs} outputs, got {m}.")

        feas_cols = []
        for i in range(self.num_outputs):
            probs_i = probs[..., i, :]
            if self.utility_values_list is not None:
                k_i = int(self.utility_values_list[i].numel())
            else:
                k_i = _num_classes_from_likelihood_or_utilities(self.ordinal_likelihoods[i])
            probs_i = probs_i[..., :k_i]
            probs_i = probs_i / probs_i.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            if self.mode == "class_ge":
                if self.min_classes is None:
                    raise ValueError("min_class must be specified for mode='class_ge'.")
                k = int(self.min_classes[i])
                if not (0 <= k < k_i):
                    raise ValueError(f"min_class[{i}] must be in [0, {k_i - 1}].")
                feas_i = probs_i[..., k:].sum(dim=-1)

            elif self.mode == "class_le":
                if self.max_classes is None:
                    raise ValueError("max_class must be specified for mode='class_le'.")
                k = int(self.max_classes[i])
                if not (0 <= k < k_i):
                    raise ValueError(f"max_class[{i}] must be in [0, {k_i - 1}].")
                feas_i = probs_i[..., : k + 1].sum(dim=-1)

            elif self.mode == "class_interval":
                if self.min_classes is None or self.max_classes is None:
                    raise ValueError(
                        "min_class and max_class must be specified for mode='class_interval'."
                    )
                lo = int(self.min_classes[i])
                hi = int(self.max_classes[i])
                if lo > hi:
                    raise ValueError(f"min_class[{i}] must be <= max_class[{i}].")
                if not (0 <= lo < k_i and 0 <= hi < k_i):
                    raise ValueError(f"class bounds for output {i} must be in [0, {k_i - 1}].")
                feas_i = probs_i[..., lo : hi + 1].sum(dim=-1)

            elif self.mode == "expected_utility_ge":
                if self.utility_values_list is None:
                    raise ValueError(
                        "utility_values must be specified for mode='expected_utility_ge'."
                    )
                thresholds = _to_float_vector(
                    self.utility_threshold,
                    m=self.num_outputs,
                    name="utility_threshold",
                    device=probs_i.device,
                    dtype=probs_i.dtype,
                    default=None,
                )
                if thresholds is None:
                    raise ValueError(
                        "utility_threshold must be specified for mode='expected_utility_ge'."
                    )
                utilities_i = self.utility_values_list[i].to(device=probs_i.device, dtype=probs_i.dtype)
                expected_u = (probs_i * utilities_i).sum(dim=-1)
                expected_u = expected_u * self.objective_signs[i].to(
                    device=probs_i.device,
                    dtype=probs_i.dtype,
                )
                tau = torch.as_tensor(self.tau, device=probs_i.device, dtype=probs_i.dtype)
                feas_i = torch.sigmoid((expected_u - thresholds[i]) / tau.clamp_min(1e-9))

            else:
                raise ValueError(f"Unknown ordinal feasibility mode: {self.mode}.")

            feas_cols.append(feas_i.clamp(self.eps, 1.0 - self.eps))

        return torch.stack(feas_cols, dim=-1)  # (*batch, q_like, m)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        expanded_X = shape_X_for_model(self.model, raw_X)

        feas_per_output = self._pointwise_feasibility_per_output(raw_X)
        score = aggregate_outputs(
            feas_per_output,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
            eps=self.eps,
        )
        score = score - _pending_penalty_per_point(
            self.model,
            expanded_X,
            self.X_pending,
            weight=self.pending_penalty_weight,
            beta=self.pending_penalty_beta,
        )

        score = apply_pointwise_score_objective(
            self,
            score,
            raw_X=raw_X,
            expanded_X=expanded_X,
            name="qMultiOutputOrdinalProbabilityOfFeasibility",
        )

        return reduce_q(score, self.reduction)


# =========================================================
# EHVI / NEHVI wrappers
# =========================================================
class qMultiOutputOrdinalExpectedHypervolumeImprovement(qExpectedHypervolumeImprovement):
    """multi-output ordinal qEHVI。

    posterior latent samples を utility objective で目的空間へ変換する。custom objective が渡された場合はそれを優先する。
    `partitioning` が渡された場合は classification 側と同様に外部指定を優先する。constraints は空リストなら None に正規化する。
    未指定の場合は `Y_baseline` または `train_Y` から utility 空間の partitioning を内部生成する。
    """

    def __init__(
        self,
        model: Model,
        ref_point: Sequence[float] | Tensor,
        *,
        partitioning=None,
        utility_values: Optional[Sequence[Sequence[float]] | Sequence[float] | Tensor] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        train_Y: Optional[Tensor] = None,
        Y_baseline: Optional[Tensor] = None,
        ordinal_likelihoods: Optional[Sequence[nn.Module] | nn.Module] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        class_offset: int = 0,
        sampler: Optional[MCSampler] = None,
        constraints: Optional[list[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Tensor | float = 1e-3,
        fat: bool = False,
        link: LinkType = "auto",
        input_perturbation_n_w: Optional[int] = None,
        risk_type: RiskType = None,
        risk_alpha: float = 0.5,
    ) -> None:
        if objective is None:
            if utility_values is None:
                raise ValueError(
                    "utility_values must be provided when objective is None."
                )
            objective = qMultiOutputOrdinalUtilityObjective(
                model=model,
                utility_values=utility_values,
                ordinal_likelihoods=ordinal_likelihoods,
                objective_signs=objective_signs,
                link=link,
                input_perturbation_n_w=input_perturbation_n_w,
                risk_type=risk_type,
                risk_alpha=risk_alpha,
            )

        # classification 側の qEHVI と同じく、partitioning が渡された場合は
        # 外部指定を優先する。partitioning は ordinal label 空間ではなく、
        # utility objective に変換済みの目的値空間で作成されている必要がある。
        if partitioning is None:
            if Y_baseline is None:
                if train_Y is None:
                    raise ValueError(
                        "Either partitioning, Y_baseline, or train_Y must be provided."
                    )
                if utility_values is None:
                    raise ValueError(
                        "utility_values must be provided to build Y_baseline from train_Y. "
                        "If you pass a custom objective, also pass Y_baseline or partitioning."
                    )
                Y_baseline = compute_observed_ordinal_utility(
                    train_Y=train_Y,
                    utility_values=utility_values,
                    objective_signs=objective_signs,
                    class_offset=class_offset,
                )

            ref_point_t = torch.as_tensor(
                ref_point,
                dtype=Y_baseline.dtype,
                device=Y_baseline.device,
            ).reshape(-1)
            partitioning = FastNondominatedPartitioning(ref_point=ref_point_t, Y=Y_baseline)
        else:
            ref_like = getattr(partitioning, "ref_point", None)
            if torch.is_tensor(ref_like):
                ref_point_t = torch.as_tensor(
                    ref_point,
                    dtype=ref_like.dtype,
                    device=ref_like.device,
                ).reshape(-1)
            elif Y_baseline is not None:
                ref_point_t = torch.as_tensor(
                    ref_point,
                    dtype=Y_baseline.dtype,
                    device=Y_baseline.device,
                ).reshape(-1)
            elif train_Y is not None:
                if utility_values is None:
                    raise ValueError(
                        "utility_values must be provided to infer ref_point dtype/device "
                        "from train_Y. If you pass a custom objective with partitioning, "
                        "pass Y_baseline or tensor-valued utility_values."
                    )
                Y_tmp = compute_observed_ordinal_utility(
                    train_Y=train_Y,
                    utility_values=utility_values,
                    objective_signs=objective_signs,
                    class_offset=class_offset,
                )
                ref_point_t = torch.as_tensor(
                    ref_point,
                    dtype=Y_tmp.dtype,
                    device=Y_tmp.device,
                ).reshape(-1)
            elif torch.is_tensor(utility_values):
                ref_point_t = torch.as_tensor(
                    ref_point,
                    dtype=utility_values.dtype,
                    device=utility_values.device,
                ).reshape(-1)
            else:
                ref_point_t = torch.as_tensor(
                    ref_point,
                    dtype=torch.get_default_dtype(),
                ).reshape(-1)

        constraints_arg = None if constraints is None or len(constraints) == 0 else constraints

        super().__init__(
            model=model,
            ref_point=ref_point_t,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints_arg,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )

        # BoTorch のバージョン差異対策:
        # constraints が None でない場合、qEHVI forward 内で self.eta / self.fat を
        # 参照することがあるため、明示的に保持する。
        self.constraints = constraints_arg
        self.eta = eta
        self.fat = fat


class qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
    qNoisyExpectedHypervolumeImprovement
):
    """multi-output ordinal qNEHVI。posterior latent samples を utility objective で目的空間へ変換する。custom objective が渡された場合はそれを優先する。"""

    def __init__(
        self,
        model: Model,
        ref_point: Sequence[float] | Tensor,
        X_baseline: Tensor,
        *,
        utility_values: Optional[Sequence[Sequence[float]] | Sequence[float] | Tensor] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        ordinal_likelihoods: Optional[Sequence[nn.Module] | nn.Module] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        sampler: Optional[MCSampler] = None,
        constraints: Optional[list[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Tensor | float = 1e-3,
        fat: bool = False,
        link: LinkType = "auto",
        input_perturbation_n_w: Optional[int] = None,
        risk_type: RiskType = None,
        risk_alpha: float = 0.5,
        **kwargs,
    ) -> None:
        if objective is None:
            if utility_values is None:
                raise ValueError(
                    "utility_values must be provided when objective is None."
                )
            objective = qMultiOutputOrdinalUtilityObjective(
                model=model,
                utility_values=utility_values,
                ordinal_likelihoods=ordinal_likelihoods,
                objective_signs=objective_signs,
                link=link,
                input_perturbation_n_w=input_perturbation_n_w,
                risk_type=risk_type,
                risk_alpha=risk_alpha,
            )

        ref_point_t = torch.as_tensor(
            ref_point,
            dtype=X_baseline.dtype,
            device=X_baseline.device,
        ).reshape(-1)

        constraints_arg = None if constraints is None or len(constraints) == 0 else constraints

        super().__init__(
            model=model,
            ref_point=ref_point_t,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints_arg,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            **kwargs,
        )

        self.constraints = constraints_arg
        self.eta = eta
        self.fat = fat


# =========================================================
# NParEGO aligned with classification implementation
# =========================================================
class _IdentityMCMultiOutputObjective(MCMultiOutputObjective):
    """samples をそのまま返す identity objective。"""

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return samples


def _squeeze_only_output_singleton(value: Tensor, X: Tensor) -> Tensor:
    q = int(X.shape[-2])
    batch_ndim = len(X.shape[:-2])
    min_ndim_with_q = batch_ndim + 1

    if value.ndim >= min_ndim_with_q + 1 and value.shape[-1] == 1:
        if value.shape[-2] == q:
            return value.squeeze(-1)

    return value


def _reduce_sample_and_q_to_tbatch(value: Tensor, X: Tensor) -> Tensor:
    batch_shape = X.shape[:-2]
    q = int(X.shape[-2])
    batch_prod = _prod(batch_shape)

    value = _squeeze_only_output_singleton(value, X)

    if value.ndim >= 1 and value.shape[-1] == q:
        value = value.max(dim=-1).values
    elif q == 1 and batch_prod == 1 and value.ndim >= 1:
        pass
    else:
        raise RuntimeError(
            "Expected scalarized value to have q dimension as the last dimension. "
            f"value.shape={tuple(value.shape)}, q={q}, batch_shape={tuple(batch_shape)}, "
            f"X.shape={tuple(X.shape)}."
        )

    while value.ndim > len(batch_shape):
        value = value.mean(dim=0)

    if value.shape == batch_shape:
        return value

    if value.numel() == batch_prod:
        return value.reshape(batch_shape)

    if len(batch_shape) == 0 and value.numel() == 1:
        return value.reshape(batch_shape)

    if q == 1 and batch_prod == 1 and value.ndim == 1:
        return value.mean().reshape(batch_shape)

    if batch_prod == 1 and value.numel() == 1:
        return value.reshape(batch_shape)

    raise RuntimeError(
        "qMultiOutputOrdinalNParEGO produced invalid output shape after scalarization. "
        f"value.shape={tuple(value.shape)}, expected batch_shape={tuple(batch_shape)}, "
        f"X.shape={tuple(X.shape)}."
    )


class qMultiOutputOrdinalNParEGO(MCAcquisitionFunction):
    """
    multi-output ordinal qNParEGO。

    classification 側に合わせて augmented Chebyshev scalarization を使う。
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Tensor,
        *,
        utility_values: Optional[Sequence[Sequence[float]] | Sequence[float] | Tensor] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        weights: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
        ordinal_likelihoods: Optional[Sequence[nn.Module] | nn.Module] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        train_Y: Optional[Tensor] = None,
        Y_baseline: Optional[Tensor] = None,
        class_offset: int = 0,
        link: LinkType = "auto",
        input_perturbation_n_w: Optional[int] = None,
        risk_type: RiskType = None,
        risk_alpha: float = 0.5,
        rho: float = 0.05,
    ) -> None:
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        if objective is None:
            if utility_values is None:
                raise ValueError(
                    "utility_values must be provided when objective is None."
                )
            objective = qMultiOutputOrdinalUtilityObjective(
                model=model,
                utility_values=utility_values,
                ordinal_likelihoods=ordinal_likelihoods,
                objective_signs=objective_signs,
                link=link,
                input_perturbation_n_w=input_perturbation_n_w,
                risk_type=risk_type,
                risk_alpha=risk_alpha,
            )

        super().__init__(model=model, sampler=sampler, objective=objective)
        self.base_objective = objective

        tkwargs = {"device": X_baseline.device, "dtype": X_baseline.dtype}
        m = int(ref_point.numel())
        self.num_outputs = m
        self.rho = float(rho)

        if weights is None:
            w = torch.rand(m, **tkwargs)
            weights = w / w.sum().clamp_min(1e-12)
        else:
            weights = weights.to(**tkwargs)
            weights = weights / weights.sum().clamp_min(1e-12)

        self.register_buffer("weights", weights)
        self.register_buffer("ref_point", ref_point.to(**tkwargs).reshape(m))

        if Y_baseline is None:
            if train_Y is None:
                raise ValueError("Either Y_baseline or train_Y must be provided.")
            if utility_values is None:
                raise ValueError(
                    "utility_values must be provided to build Y_baseline from train_Y. "
                    "If you pass a custom objective, pass Y_baseline explicitly."
                )
            Y_baseline = compute_observed_ordinal_utility(
                train_Y=train_Y,
                utility_values=utility_values,
                objective_signs=objective_signs,
                class_offset=class_offset,
            )

        with torch.no_grad():
            values = Y_baseline.to(**tkwargs).unsqueeze(0).unsqueeze(0)
            baseline_score = self._scalarize(values)
            if baseline_score.ndim >= 2 and baseline_score.shape[-1] == 1:
                baseline_score = baseline_score.squeeze(-1)
            self.register_buffer("best_value", baseline_score.max())

    def _scalarize(self, values: Tensor) -> Tensor:
        if values.ndim >= 2 and values.shape[-1] == 1 and self.num_outputs != 1:
            values = values.squeeze(-1)

        if values.ndim >= 1 and values.shape[-1] != self.num_outputs:
            return values

        if values.ndim < 1 or values.shape[-1] != self.num_outputs:
            raise RuntimeError(
                "Cannot scalarize values. Expected last dim to be num_outputs "
                f"{self.num_outputs}, got values.shape={tuple(values.shape)}."
            )

        w = self.weights.to(device=values.device, dtype=values.dtype)
        ref = self.ref_point.to(device=values.device, dtype=values.dtype)

        shifted = values - ref
        weighted = shifted * w
        cheb = weighted.min(dim=-1).values
        aug = self.rho * weighted.sum(dim=-1)

        return cheb + aug

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = ensure_q_batch(X)
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)

        values = self.base_objective(samples, X=X)
        scalarized = self._scalarize(values)
        improvement = (scalarized - self.best_value.to(scalarized)).clamp_min(0.0)

        return _reduce_sample_and_q_to_tbatch(improvement, X)


# =========================================================
# Feasibility-weighted wrapper
# =========================================================
class qMultiOutputOrdinalFeasibilityWeightedAcquisition(AcquisitionFunction):
    """任意の multi-objective acquisition を multi-output ordinal PoF で重み付けする wrapper。"""

    def __init__(
        self,
        objective_acqf: AcquisitionFunction,
        feasibility_acqf: qMultiOutputOrdinalProbabilityOfFeasibility,
        combine_mode: Literal["product", "log_product", "penalty"] = "product",
        feasibility_power: float = 1.0,
        penalty_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(model=getattr(objective_acqf, "model", feasibility_acqf.model))
        self.objective_acqf = objective_acqf
        self.feasibility_acqf = feasibility_acqf
        self.combine_mode = combine_mode
        self.feasibility_power = float(feasibility_power)
        self.penalty_weight = float(penalty_weight)
        self.eps = float(eps)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        if hasattr(self.objective_acqf, "set_X_pending"):
            self.objective_acqf.set_X_pending(X_pending)
        if hasattr(self.feasibility_acqf, "set_X_pending"):
            self.feasibility_acqf.set_X_pending(X_pending)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        base = self.objective_acqf(X)
        feas = self.feasibility_acqf(X).clamp(self.eps, 1.0 - self.eps)

        if self.combine_mode == "product":
            return base.clamp_min(0.0) * feas.pow(self.feasibility_power)
        if self.combine_mode == "log_product":
            return base + self.feasibility_power * feas.log()
        if self.combine_mode == "penalty":
            return base - self.penalty_weight * (1.0 - feas)

        raise ValueError(f"Unknown combine_mode: {self.combine_mode}")


__all__ = [
    "compute_observed_ordinal_utility",
    "ordinal_probs_from_latent",
    "qMultiOutputOrdinalUtilityObjective",
    "qMultiOutputOrdinalProbabilityOfFeasibility",
    "qMultiOutputOrdinalExpectedHypervolumeImprovement",
    "qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
    "qMultiOutputOrdinalNParEGO",
    "qMultiOutputOrdinalFeasibilityWeightedAcquisition",
]
