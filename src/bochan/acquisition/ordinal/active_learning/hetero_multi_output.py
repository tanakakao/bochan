from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform

from ..hetero_utils import (
    aggregate_objectives,
    get_noise_sigma,
    make_weight_tensor,
    stack_multi_summaries,
)


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum", "max"]
OutputMode = Literal["mean", "sum", "max", "min", "weighted_mean"]
NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "inverse_exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract"]
UncertaintyScoreType = Literal["entropy", "variance", "least_confidence"]


# =========================================================
# Probability helpers
# =========================================================
def _entropy_from_probs(probs: Tensor, eps: float = 1e-12) -> Tensor:
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _default_ordinal_likelihood(model: Model):
    lk = getattr(model, "ordinal_likelihood", None)
    if lk is not None:
        return lk
    lk = getattr(model, "likelihood", None)
    if lk is not None:
        return lk
    return None


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
    latent_samples: Tensor,
    ordinal_likelihood,
    eps: float = 1e-12,
) -> Tensor:
    f = latent_samples
    if f.ndim >= 1 and f.shape[-1] == 1:
        f = f.squeeze(-1)

    if ordinal_likelihood is None:
        raise ValueError("ordinal_likelihood is required to convert latent samples to probabilities.")

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


def _latent_samples_to_probs(
    latent_samples: Tensor,
    *,
    ordinal_likelihood=None,
    latent_to_probs: Optional[Callable[[Tensor], Tensor]] = None,
    eps: float = 1e-12,
) -> Tensor:
    if latent_samples.ndim >= 1 and latent_samples.shape[-1] != 1:
        latent_in = latent_samples.unsqueeze(-1)
    else:
        latent_in = latent_samples

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
                "Could not convert latent samples to class probabilities. "
                "Pass latent_to_probs explicitly or add class-probability conversion to the likelihood."
            )

    if probs.ndim >= 2 and probs.shape[-2] == 1:
        probs = probs.squeeze(-2)

    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _expand_list(value, n: int, name: str):
    if value is None:
        return [None] * n
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} must have length {n}, got {len(value)}.")
        return list(value)
    return [value] * n


# =========================================================
# Score objective
# =========================================================
class HeteroMultiOutputOrdinalScoreObjective(torch.nn.Module):
    """
    multi-output hetero ordinal active learning の pointwise score に作用する objective。

    posterior samples ではなく、entropy / utility variance / margin uncertainty / BALD
    などから計算済みの score に作用する。InputPerturbation の q * n_w を q に戻す用途を想定する。
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
                    "HeteroMultiOutputOrdinalScoreObjective received an aggregated score. "
                    "InputPerturbation aggregation requires pointwise score."
                )
            return score

        q_expanded = int(score.shape[-1])
        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
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


# Backward-compatible internal name; not an alias for a public acquisition.
_MultiObjectiveHeteroOrdinalScoreObjective = HeteroMultiOutputOrdinalScoreObjective


# =========================================================
# Generic tensor helpers
# =========================================================
def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
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


def _objective_call(objective, score: Tensor, X: Optional[Tensor]):
    try:
        return objective(score, X=X)
    except TypeError:
        return objective(score)


def _apply_multioutput_hetero_ordinal_objective_to_score(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    name: str = "HeteroMultiOutputOrdinalActiveLearning",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    out = _objective_call(objective, score, X)
    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

    return out


# =========================================================
# Classification-aligned base
# =========================================================
class _BaseHeteroMultiOutputOrdinalActiveLearningAcquisition(AcquisitionFunction):
    """
    heteroscedastic multi-output ordinal active learning base.

    Standard order:
        per-output pointwise score
        -> noise weighting per-output
        -> output aggregation
        -> pending penalty
        -> objective
        -> q reduction

    以前の ordinal 実装の `@concatenate_pending_points` は使わず、
    classification multi-output hetero 側と同じ penalty 型に寄せる。
    """

    def __init__(
        self,
        model: Model,
        *,
        utility_values_list: Optional[Sequence[Optional[Sequence[float] | Tensor]]] = None,
        objective_weights: Optional[Sequence[float] | Tensor] = None,
        reduction: ReductionType = "mean",
        # backward-compatible aliases
        reduce: Optional[str] = None,
        output_mode: OutputMode = "mean",
        aggregate: Optional[str] = None,
        eps: float = 1e-12,
        # pending penalty
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        # noise
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_event_aggregate: OutputMode = "mean",
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # old compatibility
        noise_penalty: float | Sequence[float] | Tensor = 0.0,
        default_sigma: float | Sequence[float] | Tensor = 0.0,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        if not hasattr(model, "models"):
            raise ValueError("Hetero multi-output ordinal acquisition expects model.models.")

        self.submodels = list(model.models)
        self.m = len(self.submodels)
        if self.m == 0:
            raise ValueError("model.models is empty.")

        if reduce is not None:
            reduction = str(reduce)
        if aggregate is not None:
            output_mode = str(aggregate)

        if reduction not in ("mean", "sum", "max"):
            raise ValueError("reduction must be 'mean', 'sum', or 'max'.")
        if output_mode not in ("mean", "sum", "max", "min", "weighted_mean"):
            raise ValueError("output_mode must be one of mean/sum/max/min/weighted_mean.")
        if noise_event_aggregate not in ("mean", "sum", "max", "min", "weighted_mean"):
            raise ValueError("noise_event_aggregate must be one of mean/sum/max/min/weighted_mean.")
        if noise_mode not in ("none", "inverse_linear", "inverse_sqrt", "inverse_exp", "custom"):
            raise ValueError(f"Unknown noise_mode: {noise_mode!r}.")
        if noise_combine not in ("multiply", "subtract"):
            raise ValueError("noise_combine must be 'multiply' or 'subtract'.")

        self.utility_values_list = utility_values_list
        self.objective_weights = objective_weights
        self.reduction = reduction
        self.output_mode = output_mode
        self.eps = float(eps)
        self.objective = objective

        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)

        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_event_aggregate = noise_event_aggregate
        self.noise_weight_fn = noise_weight_fn

        # Old API: noise_penalty can be scalar / sequence. If nonzero, treat as subtract mode.
        self.noise_penalties = _expand_list(noise_penalty, self.m, "noise_penalty")
        self.default_sigmas = _expand_list(default_sigma, self.m, "default_sigma")
        if any(float(v or 0.0) != 0.0 for v in self.noise_penalties):
            self.noise_combine = "subtract"

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
            noise_penalties=0.0,
            default_sigmas=self.default_sigmas,
            eps=self.eps,
        )

    def _weights(self, ref: Tensor) -> Optional[Tensor]:
        return make_weight_tensor(self.objective_weights, ref=ref, m=self.m)

    def _aggregate_outputs(self, values: Tensor) -> Tensor:
        """
        Args:
            values: (*batch, q_like, m)

        Returns:
            Tensor: (*batch, q_like)
        """
        weights = self._weights(values)

        # Use existing helper for consistency with previous ordinal implementation.
        return aggregate_objectives(values, method=self.output_mode, weights=weights)

    def _noise_to_weight(self, sigma_or_var: Tensor) -> Tensor:
        v = sigma_or_var.clamp_min(0.0)

        if self.noise_mode == "none":
            w = torch.ones_like(v)
        elif self.noise_mode == "inverse_linear":
            w = 1.0 / (1.0 + self.noise_penalty_lambda * v)
        elif self.noise_mode == "inverse_sqrt":
            w = 1.0 / torch.sqrt(1.0 + self.noise_penalty_lambda * v)
        elif self.noise_mode == "inverse_exp":
            w = torch.exp(-self.noise_penalty_lambda * v)
        elif self.noise_mode == "custom":
            if self.noise_weight_fn is None:
                raise ValueError("noise_weight_fn must be provided when noise_mode='custom'.")
            try:
                w = self.noise_weight_fn(v, None)
            except TypeError:
                w = self.noise_weight_fn(v)
            if not torch.is_tensor(w):
                raise TypeError(f"noise_weight_fn must return Tensor. Got {type(w)}.")
            return w.to(device=v.device, dtype=v.dtype)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}.")

        if self.noise_min_weight > 0.0:
            w = self.noise_min_weight + (1.0 - self.noise_min_weight) * w
        if self.noise_weight_scale != 1.0:
            w = self.noise_weight_scale * w
        return w

    def _aggregate_noise_weight(self, weight_per_output: Tensor) -> Tensor:
        weights = self._weights(weight_per_output)
        return aggregate_objectives(
            weight_per_output,
            method=self.noise_event_aggregate,
            weights=weights,
        )

    def _apply_noise_to_per_output_score(
        self,
        score_per_output: Tensor,
        summary: dict[str, Tensor],
        Xt: Tensor,
    ) -> Tensor:
        if "sigma" in summary:
            noise = summary["sigma"]
        else:
            # fallback: compute each submodel noise sigma.
            sigmas = []
            for i, submodel in enumerate(self.submodels):
                sigma_i = get_noise_sigma(
                    submodel,
                    Xt,
                    mean_like=score_per_output[..., i:i+1],
                    default_sigma=float(self.default_sigmas[i] or 0.0),
                ).squeeze(-1)
                sigmas.append(sigma_i)
            noise = torch.stack(sigmas, dim=-1)

        noise = _align_pointwise_score_to_X(
            noise,
            Xt.unsqueeze(-2).expand(*Xt.shape[:-1], self.m, Xt.shape[-1]).reshape(*Xt.shape[:-1], self.m, Xt.shape[-1])
            if noise.ndim == Xt.ndim else Xt,
            name="multi-output noise",
            reduce_extra="mean",
        ) if False else noise

        if self.noise_combine == "subtract":
            penalties = torch.as_tensor(
                [float(v or 0.0) for v in self.noise_penalties],
                device=score_per_output.device,
                dtype=score_per_output.dtype,
            )
            if penalties.abs().sum() == 0:
                penalties = torch.full_like(penalties, self.noise_penalty_lambda)
            view_shape = (1,) * (score_per_output.ndim - 1) + (self.m,)
            return score_per_output - noise * penalties.view(*view_shape)

        if self.noise_combine == "multiply":
            if self.noise_mode == "none":
                return score_per_output
            weight = self._noise_to_weight(noise)
            return score_per_output * weight

        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        return _apply_multioutput_hetero_ordinal_objective_to_score(self, score, X=X, name=name)

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

        # score_per_output: (*batch, q_like, m)
        expected_last = self.m
        if score_per_output.shape[-1] != expected_last:
            raise RuntimeError(
                f"{name}: expected score_per_output last dim {expected_last}, "
                f"got shape={tuple(score_per_output.shape)}."
            )

        # Align each output score to Xt.shape[:-1].
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

        score_per_output = self._apply_noise_to_per_output_score(
            score_per_output,
            summary,
            Xt,
        )
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
        score = self._apply_objective_to_score(score, X=raw_X, name=name)

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
# Public acquisition classes: direct implementation, no aliases
# =========================================================
class qHeteroMultiOutputOrdinalPredictiveEntropy(
    _BaseHeteroMultiOutputOrdinalActiveLearningAcquisition
):
    """heteroscedastic multi-output ordinal predictive entropy acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)
        values = summary["entropy"]
        return self._finalize_pointwise_score(
            values,
            raw_X,
            summary=summary,
            name="qHeteroMultiOutputOrdinalPredictiveEntropy",
        )


class qHeteroMultiOutputOrdinalUtilityVariance(
    _BaseHeteroMultiOutputOrdinalActiveLearningAcquisition
):
    """heteroscedastic multi-output ordinal utility variance acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)
        values = summary["var_u"]
        return self._finalize_pointwise_score(
            values,
            raw_X,
            summary=summary,
            name="qHeteroMultiOutputOrdinalUtilityVariance",
        )


class qHeteroMultiOutputOrdinalMarginUncertainty(
    _BaseHeteroMultiOutputOrdinalActiveLearningAcquisition
):
    """heteroscedastic multi-output ordinal margin uncertainty acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)
        summary = self._summary(raw_X)
        values = summary["margin_uncertainty"]
        return self._finalize_pointwise_score(
            values,
            raw_X,
            summary=summary,
            name="qHeteroMultiOutputOrdinalMarginUncertainty",
        )


class qHeteroMultiOutputOrdinalBALD(
    _BaseHeteroMultiOutputOrdinalActiveLearningAcquisition
):
    """
    practical multi-output ordinal BALD。

    各出力の BALD を個別に計算してから output aggregation する。
    joint multi-output BALD ではない。
    """

    def __init__(
        self,
        model: Model,
        *,
        ordinal_likelihoods: Optional[Sequence[object]] = None,
        latent_to_probs_list: Optional[Sequence[Optional[Callable[[Tensor], Tensor]]]] = None,
        num_samples: int = 128,
        sampler: Optional[SobolQMCNormalSampler] = None,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(num_samples)]))
        super().__init__(model=model, **kwargs)
        self.sampler = sampler
        self.num_samples = int(num_samples)

        self.ordinal_likelihoods = (
            list(ordinal_likelihoods)
            if ordinal_likelihoods is not None
            else [_default_ordinal_likelihood(m) for m in self.submodels]
        )
        if len(self.ordinal_likelihoods) != self.m:
            raise ValueError("ordinal_likelihoods length mismatch.")

        self.latent_to_probs_list = _expand_list(
            latent_to_probs_list,
            self.m,
            "latent_to_probs_list",
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = _ensure_q_batch(X)

        per_obj_scores = []
        for i, submodel in enumerate(self.submodels):
            posterior = submodel.posterior(raw_X, observation_noise=False)
            latent_samples = self.sampler(posterior)

            probs_mc = _latent_samples_to_probs(
                latent_samples,
                ordinal_likelihood=self.ordinal_likelihoods[i],
                latent_to_probs=self.latent_to_probs_list[i],
                eps=self.eps,
            )

            pred_probs = probs_mc.mean(dim=0)
            pred_ent = _entropy_from_probs(pred_probs, eps=self.eps)
            cond_ent = _entropy_from_probs(probs_mc, eps=self.eps).mean(dim=0)
            score_i = pred_ent - cond_ent
            per_obj_scores.append(score_i)

        values = torch.stack(per_obj_scores, dim=-1)

        # For BALD, summary is used mainly for sigma/noise weighting. If summary fails,
        # fall back to zero-noise behavior.
        try:
            summary = self._summary(raw_X)
        except Exception:
            summary = {}

        return self._finalize_pointwise_score(
            values,
            raw_X,
            summary=summary,
            name="qHeteroMultiOutputOrdinalBALD",
        )


class qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy(
    qHeteroMultiOutputOrdinalUtilityVariance
):
    """
    Lightweight IPV-style proxy based on hetero noise-aware ordinal utility variance.

    This is a proxy, not true/fantasy integrated posterior variance.
    """
    pass


__all__ = [
    "HeteroMultiOutputOrdinalScoreObjective",
    "_MultiObjectiveHeteroOrdinalScoreObjective",
    "qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy",
    "qHeteroMultiOutputOrdinalPredictiveEntropy",
    "qHeteroMultiOutputOrdinalUtilityVariance",
    "qHeteroMultiOutputOrdinalMarginUncertainty",
    "qHeteroMultiOutputOrdinalBALD",
]
