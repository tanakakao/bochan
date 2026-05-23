from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor
from torch.distributions import Categorical

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum"]
MultiOutputMode = Literal["mean", "sum", "max", "min", "weighted_mean"]
OrdinalScoreShapeMode = Literal[
    "auto",
    "pointwise",
    "multioutput_qm",
    "multioutput_mq",
    "aggregated",
]


def _validate_n_w_risk(*, n_w: Optional[int], risk_type: RiskType, alpha: float) -> None:
    if n_w is not None and int(n_w) <= 0:
        raise ValueError("n_w must be a positive integer or None.")
    if risk_type not in (None, "var", "cvar"):
        raise ValueError(f"Unknown risk_type: {risk_type!r}.")
    if risk_type is not None and n_w is None:
        raise ValueError("risk_type is specified, but n_w is None.")
    if risk_type is not None and not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0, 1].")


def _aggregate_scalar_axis(
    values_w: Tensor,
    *,
    n_w: int,
    risk_type: RiskType,
    alpha: float,
    risk_dim: int,
    maximize: bool = True,
) -> Tensor:
    if risk_type is None:
        return values_w.mean(dim=risk_dim)

    descending = not maximize
    sorted_values = torch.sort(values_w, dim=risk_dim, descending=descending).values
    k = max(1, int(math.ceil(int(n_w) * float(alpha))))
    tail = sorted_values.narrow(dim=risk_dim, start=0, length=k)

    if risk_type == "var":
        return tail.select(dim=risk_dim, index=k - 1)
    if risk_type == "cvar":
        return tail.mean(dim=risk_dim)
    raise ValueError(f"Unknown risk_type: {risk_type!r}.")


def _aggregate_multioutput_axis(
    values_w: Tensor,
    *,
    n_w: int,
    risk_type: RiskType,
    alpha: float,
    risk_dim: int = -2,
) -> Tensor:
    if risk_type is None:
        return values_w.mean(dim=risk_dim)

    sorted_values = torch.sort(values_w, dim=risk_dim, descending=False).values
    k = max(1, int(math.ceil(int(n_w) * float(alpha))))
    tail = sorted_values.narrow(dim=risk_dim, start=0, length=k)

    if risk_type == "var":
        return tail.select(dim=risk_dim, index=k - 1)
    if risk_type == "cvar":
        return tail.mean(dim=risk_dim)
    raise ValueError(f"Unknown risk_type: {risk_type!r}.")


class MultiOutputOrdinalScoreObjective(torch.nn.Module):
    """
    multi-output ordinal active learning / level-set の score に作用する objective。

    acquisition 内で計算済みの score に作用する。
    InputPerturbation では q * n_w を q に戻す。

    Supported score shapes:
        - (*batch, q_like)
        - (*batch, q_like, m)
        - (*batch, m, q_like)
        - (*batch,) aggregated score
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        aggregated_risk_mode: Literal["ignore", "error"] = "ignore",
        score_shape_mode: OrdinalScoreShapeMode = "auto",
    ) -> None:
        super().__init__()
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.aggregated_risk_mode = aggregated_risk_mode
        self.score_shape_mode = score_shape_mode

        _validate_n_w_risk(n_w=self.n_w, risk_type=self.risk_type, alpha=self.alpha)

        if self.aggregated_risk_mode not in ("ignore", "error"):
            raise ValueError("aggregated_risk_mode must be 'ignore' or 'error'.")
        if self.score_shape_mode not in (
            "auto",
            "pointwise",
            "multioutput_qm",
            "multioutput_mq",
            "aggregated",
        ):
            raise ValueError(
                "score_shape_mode must be one of 'auto', 'pointwise', "
                "'multioutput_qm', 'multioutput_mq', or 'aggregated'."
            )

    @staticmethod
    def _ensure_q_batch(X: Tensor) -> Tensor:
        return X if X.dim() > 2 else X.unsqueeze(0)

    def _batch_shape_from_X(self, X: Optional[Tensor]) -> Optional[torch.Size]:
        if X is None:
            return None
        Xq = self._ensure_q_batch(X)
        return Xq.shape[:-2]

    def _q_from_X(self, X: Optional[Tensor]) -> Optional[int]:
        if X is None:
            return None
        Xq = self._ensure_q_batch(X)
        return int(Xq.shape[-2])

    def _infer_score_shape_mode(self, score: Tensor, X: Optional[Tensor]) -> OrdinalScoreShapeMode:
        if self.score_shape_mode != "auto":
            return self.score_shape_mode

        if score.ndim == 0:
            return "aggregated"

        batch_shape = self._batch_shape_from_X(X)
        q = self._q_from_X(X)

        if batch_shape is not None:
            if tuple(score.shape) == tuple(batch_shape):
                return "aggregated"
            if score.ndim >= 1 and tuple(score.shape[:-1]) == tuple(batch_shape):
                return "pointwise"
            if score.ndim >= 2 and tuple(score.shape[:-2]) == tuple(batch_shape):
                if q is not None and self.n_w is not None:
                    q_expanded = q * int(self.n_w)
                    if score.shape[-2] in (q, q_expanded):
                        return "multioutput_qm"
                    if score.shape[-1] in (q, q_expanded):
                        return "multioutput_mq"
                return "multioutput_qm"

        if score.ndim == 1:
            return "pointwise"
        if score.ndim >= 2:
            return "multioutput_qm"
        return "aggregated"

    def _handle_aggregated_score(self, score: Tensor) -> Tensor:
        if self.n_w is not None and self.n_w > 1 and self.aggregated_risk_mode == "error":
            raise RuntimeError(
                "MultiOutputOrdinalScoreObjective received an aggregated score. "
                "InputPerturbation aggregation is only valid for pointwise score."
            )
        return score

    def _aggregate_pointwise_score(self, score: Tensor) -> Tensor:
        if self.n_w is None or self.n_w <= 1:
            return score
        q_expanded = score.shape[-1]
        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )
        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-1], q, int(self.n_w))
        return _aggregate_scalar_axis(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=True,
        )

    def _aggregate_multioutput_qm_score(self, score: Tensor) -> Tensor:
        if self.n_w is None or self.n_w <= 1:
            return score
        q_expanded = score.shape[-2]
        m = score.shape[-1]
        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-2] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )
        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-2], q, int(self.n_w), m)
        return _aggregate_multioutput_axis(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-2,
        )

    def _aggregate_multioutput_mq_score(self, score: Tensor) -> Tensor:
        if score.ndim < 2:
            raise RuntimeError(
                "multioutput_mq score must have shape (*batch, m, q_like). "
                f"Got shape={tuple(score.shape)}."
            )
        if self.n_w is None or self.n_w <= 1:
            return score.transpose(-1, -2)

        m = score.shape[-2]
        q_expanded = score.shape[-1]
        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-2], m, q, int(self.n_w))
        score_mq = _aggregate_scalar_axis(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=True,
        )
        return score_mq.transpose(-1, -2)

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(score):
            raise TypeError(f"score must be a Tensor. Got {type(score)}.")

        mode = self._infer_score_shape_mode(score, X)
        if mode == "aggregated":
            return self._handle_aggregated_score(score)
        if mode == "pointwise":
            return self._aggregate_pointwise_score(score)
        if mode == "multioutput_qm":
            return self._aggregate_multioutput_qm_score(score)
        if mode == "multioutput_mq":
            return self._aggregate_multioutput_mq_score(score)
        raise RuntimeError(f"Unsupported inferred score shape mode: {mode}")


_MultiOutputOrdinalScoreObjective = MultiOutputOrdinalScoreObjective


def _apply_multi_output_ordinal_objective_to_score(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    name: str = "MultiOutputOrdinalAcquisition",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    try:
        out = objective(score, X=X)
    except TypeError:
        out = objective(score)

    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
    return out


def ordinal_entropy_from_probs(probs: Tensor, eps: float = 1e-12) -> Tensor:
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _resolve_submodels(model: Model) -> list[Model]:
    submodels = getattr(model, "models", None)
    if submodels is None:
        raise ValueError(
            "This acquisition expects a model with a `.models` attribute "
            "(e.g. MultiOutputOrdinalModel or ModelList-style model)."
        )
    submodels = list(submodels)
    if len(submodels) == 0:
        raise ValueError("model.models is empty.")
    return submodels


def _is_ordinal_likelihood(obj) -> bool:
    return obj is not None and hasattr(obj, "marginal_class_probs") and hasattr(obj, "class_probs_from_f")


def _resolve_ordinal_likelihoods(
    model: Model,
    ordinal_likelihoods: Optional[Sequence[OrdinalLogitLikelihood]],
) -> list[OrdinalLogitLikelihood]:
    submodels = _resolve_submodels(model)

    if ordinal_likelihoods is not None:
        likelihoods = list(ordinal_likelihoods)
        if len(likelihoods) != len(submodels):
            raise ValueError(
                "ordinal_likelihoods length mismatch: "
                f"expected {len(submodels)}, got {len(likelihoods)}"
            )
        return likelihoods

    resolved: list[OrdinalLogitLikelihood] = []
    for i, sm in enumerate(submodels):
        for cand in (getattr(sm, "ordinal_likelihood", None), getattr(sm, "likelihood", None)):
            if _is_ordinal_likelihood(cand):
                resolved.append(cand)
                break
        else:
            raise ValueError(f"Submodel at index {i} does not expose ordinal_likelihood or likelihood.")
    return resolved


def _resolve_ref_device_dtype(model: Model) -> tuple[torch.device, torch.dtype]:
    for attr in ("train_X", "train_inputs_raw"):
        x = getattr(model, attr, None)
        if torch.is_tensor(x):
            return x.device, x.dtype

    train_inputs = getattr(model, "train_inputs", None)
    if isinstance(train_inputs, tuple) and len(train_inputs) > 0 and torch.is_tensor(train_inputs[0]):
        return train_inputs[0].device, train_inputs[0].dtype

    submodels = _resolve_submodels(model)
    for sm in submodels:
        for attr in ("train_X", "train_inputs_raw"):
            x = getattr(sm, attr, None)
            if torch.is_tensor(x):
                return x.device, x.dtype
        train_inputs = getattr(sm, "train_inputs", None)
        if isinstance(train_inputs, tuple) and len(train_inputs) > 0 and torch.is_tensor(train_inputs[0]):
            return train_inputs[0].device, train_inputs[0].dtype

    return torch.device("cpu"), torch.get_default_dtype()


def _resolve_output_weights(
    model: Model,
    output_weights: Optional[Sequence[float] | Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    submodels = _resolve_submodels(model)
    m = len(submodels)
    if output_weights is None:
        return torch.ones(m, dtype=dtype, device=device)

    weights = torch.as_tensor(output_weights, dtype=dtype, device=device).flatten()
    if weights.numel() != m:
        raise ValueError(f"output_weights length mismatch: expected {m}, got {weights.numel()}")
    return weights


def _resolve_cat_dims(model: Model) -> list[int]:
    cat_dims = getattr(model, "cat_dims", None)
    if cat_dims is not None:
        return [int(i) for i in cat_dims]

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        cat_dims = getattr(models[0], "cat_dims", None)
        if cat_dims is not None:
            return [int(i) for i in cat_dims]
    return []


def _split_cont_cat(X: Tensor, cat_dims: Sequence[int]) -> tuple[Optional[Tensor], Optional[Tensor]]:
    d = X.shape[-1]
    cat_dims = [i for i in cat_dims if 0 <= i < d]
    cont_dims = [i for i in range(d) if i not in cat_dims]
    X_cont = X[..., cont_dims] if len(cont_dims) > 0 else None
    X_cat = X[..., cat_dims] if len(cat_dims) > 0 else None
    return X_cont, X_cat


def _pairwise_distance_proxy(A: Tensor, B: Tensor, cat_dims: Sequence[int]) -> Tensor:
    A_cont, A_cat = _split_cont_cat(A, cat_dims)
    B_cont, B_cat = _split_cont_cat(B, cat_dims)
    dist2: Tensor | float = 0.0
    if A_cont is not None:
        diff = A_cont.unsqueeze(-2) - B_cont.unsqueeze(-3)
        dist2 = dist2 + (diff ** 2).sum(dim=-1)
    if A_cat is not None:
        mismatch = (A_cat.unsqueeze(-2) != B_cat.unsqueeze(-3)).to(A.dtype)
        dist2 = dist2 + mismatch.sum(dim=-1)
    if isinstance(dist2, float):
        raise RuntimeError("No valid dimensions found for distance computation.")
    return dist2


def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _ensure_q_batch_for_pending(X: Tensor) -> Tensor:
    return _ensure_q_batch(X)


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
            "X_pending / X_observed must be None, Tensor, list, or tuple. "
            f"Got {type(X_ref)}."
        )
    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out


_coerce_pending_to_tensor = _coerce_reference_to_tensor


def _apply_input_transform_for_reference(model: Model, X: Tensor) -> Tensor:
    X = _ensure_q_batch(X)
    it = getattr(model, "input_transform", None)
    if it is not None:
        Xt = it(X)
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return _ensure_q_batch(Xt)

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        it = getattr(models[0], "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch(Xt)
    return X


_apply_input_transform_for_pending = _apply_input_transform_for_reference


def _transform_reference_like_candidate(model: Model, X_ref, *, ref: Tensor) -> Optional[Tensor]:
    Xr = _coerce_reference_to_tensor(X_ref, ref=ref)
    if Xr is None or Xr.numel() == 0:
        return None
    Xr_t = _apply_input_transform_for_reference(model, Xr)
    return Xr_t.to(device=ref.device, dtype=ref.dtype)


_transform_pending_like_candidate = _transform_reference_like_candidate


def _resolve_observed_X(model: Model, X_observed: Optional[Tensor] = None) -> Optional[Tensor]:
    if X_observed is not None:
        return X_observed
    for attr in ("train_X_original", "train_X", "train_inputs_raw"):
        x = getattr(model, attr, None)
        if x is not None:
            return x
    train_inputs = getattr(model, "train_inputs", None)
    if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
        return train_inputs[0]
    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        sm = models[0]
        for attr in ("train_X_original", "train_X", "train_inputs_raw"):
            x = getattr(sm, attr, None)
            if x is not None:
                return x
        train_inputs = getattr(sm, "train_inputs", None)
        if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
            return train_inputs[0]
    return None


def _broadcast_reference_to_batch(X_ref: Tensor, batch_shape: torch.Size) -> Tensor:
    X_ref = _ensure_q_batch(X_ref)
    if X_ref.shape[:-2] == batch_shape:
        return X_ref
    try:
        return X_ref.expand(*batch_shape, X_ref.shape[-2], X_ref.shape[-1])
    except RuntimeError:
        X2d = X_ref.reshape(-1, X_ref.shape[-1])
        return X2d.view(*([1] * len(batch_shape)), X2d.shape[-2], X2d.shape[-1]).expand(
            *batch_shape,
            X2d.shape[-2],
            X2d.shape[-1],
        )


def _reference_penalty_per_point(
    X: Tensor,
    X_ref: Optional[Tensor],
    *,
    beta: float,
    weight: float,
    cat_dims: Sequence[int],
) -> Tensor:
    X = _ensure_q_batch(X)
    if weight <= 0.0 or X_ref is None or X_ref.numel() == 0:
        return X.new_zeros(X.shape[:-1])
    X_ref = _broadcast_reference_to_batch(X_ref.to(device=X.device, dtype=X.dtype), X.shape[:-2])
    dist2 = _pairwise_distance_proxy(X, X_ref, cat_dims)
    nearest = dist2.min(dim=-1).values
    return weight * torch.exp(-float(beta) * nearest)


def _reference_penalty_aggregated(
    X: Tensor,
    X_ref: Optional[Tensor],
    *,
    beta: float,
    weight: float,
    cat_dims: Sequence[int],
    reduction: ReductionType = "sum",
) -> Tensor:
    per_point = _reference_penalty_per_point(X=X, X_ref=X_ref, beta=beta, weight=weight, cat_dims=cat_dims)
    if reduction == "mean":
        return per_point.mean(dim=-1)
    if reduction == "sum":
        return per_point.sum(dim=-1)
    raise ValueError(f"Unknown reduction: {reduction}")


def _pending_penalty(X: Tensor, X_pending: Optional[Tensor], beta: float, cat_dims: Sequence[int]) -> Tensor:
    return _reference_penalty_aggregated(
        X=X,
        X_ref=X_pending,
        beta=beta,
        weight=1.0,
        cat_dims=cat_dims,
        reduction="sum",
    )


def _same_batch_penalty_per_point(
    X: Tensor,
    *,
    beta: float,
    weight: float,
    cat_dims: Sequence[int],
) -> Tensor:
    X = _ensure_q_batch(X)
    batch_shape = X.shape[:-2]
    q = X.shape[-2]
    if q <= 1 or weight <= 0.0:
        return X.new_zeros(X.shape[:-1])
    d2 = _pairwise_distance_proxy(X, X, cat_dims)
    eye = torch.eye(q, device=X.device, dtype=torch.bool)
    d2 = d2.masked_fill(eye, float("inf"))
    return weight * torch.exp(-float(beta) * d2).sum(dim=-1).reshape(*batch_shape, q)


def _same_batch_penalty_aggregated(
    X: Tensor,
    *,
    beta: float,
    weight: float,
    cat_dims: Sequence[int],
) -> Tensor:
    per_point = _same_batch_penalty_per_point(X, beta=beta, weight=weight, cat_dims=cat_dims)
    if per_point.shape[-1] <= 1:
        return per_point.new_zeros(per_point.shape[:-1])
    return 0.5 * per_point.sum(dim=-1)


def _ensure_class_probs_shape(probs: Tensor, q: int) -> Tensor:
    if probs.ndim < 2:
        raise ValueError(f"Expected probs.ndim >= 2, got shape={tuple(probs.shape)}")
    if q == 1 and probs.shape[-2] != 1:
        probs = probs.unsqueeze(-2)
    return probs


def _sample_latent(posterior, sampler: MCSampler) -> Tensor:
    samples = sampler(posterior)
    if samples.ndim >= 1 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)
    return samples


class _MultiOutputOrdinalAcquisitionBase(AcquisitionFunction):
    """q=1 用 independent multi-output ordinal acquisition の base。"""

    def __init__(
        self,
        model: Model,
        ordinal_likelihoods: Optional[Sequence[OrdinalLogitLikelihood]] = None,
        output_weights: Optional[Sequence[float] | Tensor] = None,
        eps: float = 1e-12,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)
        self.submodels = _resolve_submodels(model)
        self.ordinal_likelihoods = _resolve_ordinal_likelihoods(model, ordinal_likelihoods)
        device, dtype = _resolve_ref_device_dtype(model)
        self.register_buffer(
            "output_weights",
            _resolve_output_weights(model, output_weights, dtype=dtype, device=device),
        )
        self.eps = float(eps)
        self.objective = objective

    def _weights_like(self, X: Tensor) -> Tensor:
        return self.output_weights.to(device=X.device, dtype=X.dtype)

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        return _apply_multi_output_ordinal_objective_to_score(self, score, X=X, name=name)


class _qMultiOutputOrdinalActiveLearningBase(AcquisitionFunction):
    """
    classification の multi-output active learning base に寄せた ordinal 版。

    Standard order:
        1. submodel ごとの pointwise score を計算
        2. score_per_output: (*batch, q_like, m) に stack
        3. output_mode で出力方向を集約 -> (*batch, q_like)
        4. pending / observed / same-batch penalty を pointwise に差し引く
        5. objective を pointwise score に適用
        6. reduction で q 方向を集約
    """

    def __init__(
        self,
        model: Model,
        ordinal_likelihoods: Optional[Sequence[OrdinalLogitLikelihood]] = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Sequence[float] | Tensor] = None,
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if output_mode not in ("mean", "sum", "max", "min", "weighted_mean"):
            raise ValueError("output_mode must be one of 'mean', 'sum', 'max', 'min', 'weighted_mean'.")

        self.submodels = _resolve_submodels(model)
        self.ordinal_likelihoods = _resolve_ordinal_likelihoods(model, ordinal_likelihoods)
        device, dtype = _resolve_ref_device_dtype(model)
        self.register_buffer(
            "output_weights",
            _resolve_output_weights(model, output_weights, dtype=dtype, device=device),
        )
        self.reduction = reduction
        self.output_mode = output_mode
        self.sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self.eps = float(eps)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.cat_dims = _resolve_cat_dims(model)
        self.objective = objective
        self.X_pending: Optional[Tensor] = None
        self.X_observed: Optional[Tensor] = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    def _weights_like(self, X: Tensor) -> Tensor:
        return self.output_weights.to(device=X.device, dtype=X.dtype)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = _coerce_reference_to_tensor(_resolve_observed_X(self.model, X_observed))

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
        for lik in self.ordinal_likelihoods:
            if hasattr(lik, "eval"):
                lik.eval()

    def _aggregate_outputs(self, score_per_output: Tensor) -> Tensor:
        if self.output_mode == "mean":
            return score_per_output.mean(dim=-1)
        if self.output_mode == "sum":
            return score_per_output.sum(dim=-1)
        if self.output_mode == "max":
            return score_per_output.max(dim=-1).values
        if self.output_mode == "min":
            return score_per_output.min(dim=-1).values
        if self.output_mode == "weighted_mean":
            w = self.output_weights.to(device=score_per_output.device, dtype=score_per_output.dtype)
            if w.ndim != 1 or w.numel() != score_per_output.shape[-1]:
                raise ValueError(
                    f"output_weights must have shape ({score_per_output.shape[-1]},), got {tuple(w.shape)}."
                )
            w = w / w.sum().clamp_min(self.eps)
            view_shape = (1,) * (score_per_output.ndim - 1) + (w.numel(),)
            return (score_per_output * w.view(*view_shape)).sum(dim=-1)
        raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def _reduce_q(self, score: Tensor) -> Tensor:
        if score.ndim == 0:
            return score
        if score.shape[-1] == 1:
            return score.squeeze(-1)
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        raise ValueError(f"Unknown reduction: {self.reduction}")

    def _check_output_shape(self, out: Tensor, expected: torch.Size, name: str) -> None:
        if out.shape != expected:
            raise RuntimeError(f"{name} output shape mismatch: expected {tuple(expected)}, got {tuple(out.shape)}.")

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        return _reference_penalty_per_point(
            Xt,
            Xp_t,
            beta=self.pending_penalty_beta,
            weight=self.pending_penalty_weight,
            cat_dims=self.cat_dims,
        )

    def _observed_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
        return _reference_penalty_per_point(
            Xt,
            Xobs_t,
            beta=self.observed_penalty_beta,
            weight=self.observed_penalty_weight,
            cat_dims=self.cat_dims,
        )

    def _same_batch_penalty_per_point(self, Xt: Tensor) -> Tensor:
        return _same_batch_penalty_per_point(
            Xt,
            beta=self.same_batch_penalty_beta,
            weight=self.same_batch_penalty_weight,
            cat_dims=self.cat_dims,
        )

    def _pointwise_repulsion_penalty(self, Xt: Tensor) -> Tensor:
        return (
            self._pending_penalty_per_point(Xt)
            + self._observed_penalty_per_point(Xt)
            + self._same_batch_penalty_per_point(Xt)
        )

    def _apply_objective_to_pointwise_score(
        self,
        score: Tensor,
        *,
        raw_X: Tensor,
        expanded_X: Tensor,
        name: str,
    ) -> Tensor:
        objective = getattr(self, "objective", None)
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
            raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
        if out.ndim == raw_X.ndim and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def _finalize_pointwise_score(self, score: Tensor, X: Tensor, *, name: str) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = raw_X.shape[:-2]
        Xt = _apply_input_transform_for_reference(self.model, raw_X)
        if score.shape != Xt.shape[:-1]:
            if score.numel() == math.prod(Xt.shape[:-1]):
                score = score.reshape(*Xt.shape[:-1])
            else:
                raise RuntimeError(
                    f"{name}: score shape mismatch before penalty. "
                    f"score.shape={tuple(score.shape)}, Xt.shape={tuple(Xt.shape)}."
                )
        score = score - self._pointwise_repulsion_penalty(Xt)
        score = self._apply_objective_to_pointwise_score(score, raw_X=raw_X, expanded_X=Xt, name=name)
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, name)
        return out


_qMultiOutputOrdinalMCAcquisitionBase = _qMultiOutputOrdinalActiveLearningBase


class _MultiOutputOrdinalPredictiveEntropy(_MultiOutputOrdinalAcquisitionBase):
    """pointwise predictive entropy を出力方向に加算する q=1 acquisition。"""

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        weights = self._weights_like(X)
        total: Optional[Tensor] = None
        for i, (sm, lik) in enumerate(zip(self.submodels, self.ordinal_likelihoods)):
            posterior = sm.posterior(X)
            probs = lik.marginal_class_probs(posterior.distribution)
            probs = _ensure_class_probs_shape(probs, q=1)
            ent = ordinal_entropy_from_probs(probs, eps=self.eps)
            weighted = weights[i] * ent
            total = weighted if total is None else total + weighted
        if total is None:
            raise RuntimeError("No submodels were available.")
        total = self._apply_objective_to_score(total, X=X, name="_MultiOutputOrdinalPredictiveEntropy")
        expected = X.shape[:-2]
        if total.shape == expected:
            return total
        if total.shape[-1] == 1:
            return total.squeeze(-1)
        return total.mean(dim=-1)


class _MultiOutputOrdinalBALD(_MultiOutputOrdinalAcquisitionBase):
    """pointwise BALD を出力方向に加算する q=1 acquisition。"""

    def __init__(
        self,
        model: Model,
        ordinal_likelihoods: Optional[Sequence[OrdinalLogitLikelihood]] = None,
        output_weights: Optional[Sequence[float] | Tensor] = None,
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-12,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            ordinal_likelihoods=ordinal_likelihoods,
            output_weights=output_weights,
            eps=eps,
            objective=objective,
        )
        self.sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([256]))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        weights = self._weights_like(X)
        total: Optional[Tensor] = None
        for i, (sm, lik) in enumerate(zip(self.submodels, self.ordinal_likelihoods)):
            posterior = sm.posterior(X)
            probs = lik.marginal_class_probs(posterior.distribution)
            probs = _ensure_class_probs_shape(probs, q=1)
            predictive_entropy = ordinal_entropy_from_probs(probs, eps=self.eps)
            latent_samples = _sample_latent(posterior, self.sampler)
            class_probs_given_f = lik.class_probs_from_f(latent_samples)
            cond_entropy = ordinal_entropy_from_probs(class_probs_given_f, eps=self.eps).mean(dim=0)
            bald = predictive_entropy - cond_entropy
            weighted = weights[i] * bald
            total = weighted if total is None else total + weighted
        if total is None:
            raise RuntimeError("No submodels were available.")
        total = self._apply_objective_to_score(total, X=X, name="_MultiOutputOrdinalBALD")
        expected = X.shape[:-2]
        if total.shape == expected:
            return total
        if total.shape[-1] == 1:
            return total.squeeze(-1)
        return total.mean(dim=-1)


class qMultiOutputOrdinalPredictiveEntropy(_qMultiOutputOrdinalActiveLearningBase):
    """multi-output ordinal predictive entropy acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        scores: list[Tensor] = []
        for sm, lik in zip(self.submodels, self.ordinal_likelihoods):
            posterior = sm.posterior(X)
            probs = lik.marginal_class_probs(posterior.distribution)
            probs = _ensure_class_probs_shape(probs, q=X.shape[-2])
            ent = ordinal_entropy_from_probs(probs, eps=self.eps)
            scores.append(ent)
        if len(scores) == 0:
            raise RuntimeError("No submodels were available.")
        score_per_output = torch.stack(scores, dim=-1)
        score = self._aggregate_outputs(score_per_output)
        return self._finalize_pointwise_score(score, X, name="qMultiOutputOrdinalPredictiveEntropy")


class qMultiOutputOrdinalBALD(_qMultiOutputOrdinalActiveLearningBase):
    """multi-output ordinal BALD / mutual-information acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        scores: list[Tensor] = []
        for sm, lik in zip(self.submodels, self.ordinal_likelihoods):
            posterior = sm.posterior(X)
            probs = lik.marginal_class_probs(posterior.distribution)
            probs = _ensure_class_probs_shape(probs, q=X.shape[-2])
            predictive_entropy = ordinal_entropy_from_probs(probs, eps=self.eps)
            latent_samples = _sample_latent(posterior, self.sampler)
            class_probs_given_f = lik.class_probs_from_f(latent_samples)
            cond_entropy = ordinal_entropy_from_probs(class_probs_given_f, eps=self.eps).mean(dim=0)
            scores.append(predictive_entropy - cond_entropy)
        if len(scores) == 0:
            raise RuntimeError("No submodels were available.")
        score_per_output = torch.stack(scores, dim=-1)
        score = self._aggregate_outputs(score_per_output)
        return self._finalize_pointwise_score(score, X, name="qMultiOutputOrdinalBALD")


class qMultiOutputOrdinalUtilityVariance(_qMultiOutputOrdinalActiveLearningBase):
    """multi-output ordinal utility-variance acquisition."""

    def __init__(
        self,
        model: Model,
        ordinal_likelihoods: Optional[Sequence[OrdinalLogitLikelihood]] = None,
        utility_values_list: Optional[Sequence[Optional[Sequence[float] | Tensor]]] = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        output_weights: Optional[Sequence[float] | Tensor] = None,
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            ordinal_likelihoods=ordinal_likelihoods,
            reduction=reduction,
            output_mode=output_mode,
            output_weights=output_weights,
            sampler=sampler,
            eps=eps,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_pending=X_pending,
            X_observed=X_observed,
            objective=objective,
        )
        self.utility_values_list = utility_values_list

    def _utility_values(self, i: int, probs: Tensor) -> Tensor:
        if self.utility_values_list is None or self.utility_values_list[i] is None:
            return torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        u = torch.as_tensor(self.utility_values_list[i], device=probs.device, dtype=probs.dtype).reshape(-1)
        if u.numel() != probs.shape[-1]:
            raise ValueError(f"utility_values for output {i} must have length {probs.shape[-1]}.")
        return u

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        scores: list[Tensor] = []
        for i, (sm, lik) in enumerate(zip(self.submodels, self.ordinal_likelihoods)):
            posterior = sm.posterior(X)
            probs = lik.marginal_class_probs(posterior.distribution)
            probs = _ensure_class_probs_shape(probs, q=X.shape[-2])
            probs = probs.clamp_min(self.eps)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            u = self._utility_values(i, probs)
            mean_u = (probs * u).sum(dim=-1)
            second_u = (probs * u.pow(2)).sum(dim=-1)
            var_u = (second_u - mean_u.pow(2)).clamp_min(0.0)
            scores.append(var_u)
        if len(scores) == 0:
            raise RuntimeError("No submodels were available.")
        score_per_output = torch.stack(scores, dim=-1)
        score = self._aggregate_outputs(score_per_output)
        return self._finalize_pointwise_score(score, X, name="qMultiOutputOrdinalUtilityVariance")


class qMultiOutputOrdinalMarginUncertainty(_qMultiOutputOrdinalActiveLearningBase):
    """multi-output ordinal margin uncertainty acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        scores: list[Tensor] = []
        for sm, lik in zip(self.submodels, self.ordinal_likelihoods):
            posterior = sm.posterior(X)
            probs = lik.marginal_class_probs(posterior.distribution)
            probs = _ensure_class_probs_shape(probs, q=X.shape[-2])
            probs = probs.clamp_min(self.eps)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
            if top2.shape[-1] == 1:
                margin_unc = torch.zeros_like(top2[..., 0])
            else:
                margin_unc = 1.0 - (top2[..., 0] - top2[..., 1])
            scores.append(margin_unc)
        if len(scores) == 0:
            raise RuntimeError("No submodels were available.")
        score_per_output = torch.stack(scores, dim=-1)
        score = self._aggregate_outputs(score_per_output)
        return self._finalize_pointwise_score(score, X, name="qMultiOutputOrdinalMarginUncertainty")


class qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance(AcquisitionFunction):
    """
    multi-output ordinal 用 fantasy negative integrated posterior variance。

    candidate X -> fantasy labels -> condition_on_observations
    -> integrated latent posterior variance over mc_points.

    勾配ベース最適化には基本的に不向きです。
    """

    supports_gradient_optimization = False

    def __init__(
        self,
        model: Model,
        mc_points: Tensor,
        ordinal_likelihoods: Optional[Sequence[OrdinalLogitLikelihood]] = None,
        output_weights: Optional[Sequence[float] | Tensor] = None,
        num_fantasies: int = 8,
        conditioning_steps: int = 10,
        conditioning_lr: float | None = None,
        conditioning_batch_size: int | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)
        if mc_points.ndim != 2:
            raise ValueError(f"mc_points must be [N_mc, d], got shape={tuple(mc_points.shape)}")
        self.submodels = _resolve_submodels(model)
        self.ordinal_likelihoods = _resolve_ordinal_likelihoods(model, ordinal_likelihoods)
        device, dtype = _resolve_ref_device_dtype(model)
        self.register_buffer(
            "output_weights",
            _resolve_output_weights(model, output_weights, dtype=dtype, device=device),
        )
        self.register_buffer("mc_points", mc_points.to(device=device, dtype=dtype))
        self.num_fantasies = int(num_fantasies)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.cat_dims = _resolve_cat_dims(model)
        self.objective = objective
        self.X_pending: Optional[Tensor] = None
        self.X_observed: Optional[Tensor] = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    def _weights_like(self, X: Tensor) -> Tensor:
        return self.output_weights.to(device=X.device, dtype=X.dtype)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = _coerce_reference_to_tensor(_resolve_observed_X(self.model, X_observed))

    def _resolve_conditioning_lr_for_submodel(self, submodel: Model) -> float:
        if self.conditioning_lr is not None:
            return float(self.conditioning_lr)
        for attr in ("conditioning_lr", "lr"):
            val = getattr(submodel, attr, None)
            if val is not None:
                return float(val)
        return 0.03

    def _resolve_conditioning_batch_size_for_submodel(self, submodel: Model) -> Optional[int]:
        if self.conditioning_batch_size is not None:
            return int(self.conditioning_batch_size)
        for attr in ("conditioning_batch_size", "batch_size"):
            val = getattr(submodel, attr, None)
            if val is not None:
                return int(val)
        return None

    @torch.no_grad()
    def _sample_fantasy_labels_per_output(self, X: Tensor) -> list[Tensor]:
        fantasies: list[Tensor] = []
        for submodel, lik in zip(self.submodels, self.ordinal_likelihoods):
            posterior = submodel.posterior(X)
            latent_samples = posterior.rsample(torch.Size([self.num_fantasies]))
            if latent_samples.ndim >= 1 and latent_samples.shape[-1] == 1:
                latent_samples = latent_samples.squeeze(-1)
            probs = lik.class_probs_from_f(latent_samples)
            probs = probs.clamp_min(1e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            fantasy_Y = []
            for i in range(self.num_fantasies):
                yi = Categorical(probs=probs[i]).sample()
                fantasy_Y.append(yi.unsqueeze(-1))
            fantasies.append(torch.stack(fantasy_Y, dim=0))
        return fantasies

    @torch.no_grad()
    def _integrated_latent_variance_from_submodels(self, fantasy_submodels: Sequence[Model], ref_X: Tensor) -> Tensor:
        weights = self._weights_like(ref_X)
        total = ref_X.new_zeros(())
        for i, sm in enumerate(fantasy_submodels):
            posterior = sm.posterior(self.mc_points)
            total = total + weights[i] * posterior.variance.mean()
        return total

    def _aggregated_repulsion_penalty(self, X: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_reference(self.model, X)
        penalty = _same_batch_penalty_aggregated(
            Xt,
            beta=self.same_batch_penalty_beta,
            weight=self.same_batch_penalty_weight,
            cat_dims=self.cat_dims,
        )
        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        penalty = penalty + _reference_penalty_aggregated(
            Xt,
            Xp_t,
            beta=self.pending_penalty_beta,
            weight=self.pending_penalty_weight,
            cat_dims=self.cat_dims,
            reduction="sum",
        )
        Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
        penalty = penalty + _reference_penalty_aggregated(
            Xt,
            Xobs_t,
            beta=self.observed_penalty_beta,
            weight=self.observed_penalty_weight,
            cat_dims=self.cat_dims,
            reduction="sum",
        )
        return penalty

    def _apply_objective_to_scalar_score(self, score: Tensor, X: Tensor) -> Tensor:
        objective = getattr(self, "objective", None)
        if objective is None:
            return score
        try:
            out = objective(score, X=X)
        except TypeError:
            out = objective(score)
        if not torch.is_tensor(out):
            raise TypeError(
                "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance: "
                f"objective must return Tensor. Got {type(out)}."
            )
        return out

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = _ensure_q_batch(X)
        batch_shape = X.shape[:-2]
        X_flat = X.reshape(-1, X.shape[-2], X.shape[-1])
        out = []
        for Xb in X_flat:
            Xb = Xb.detach()
            fantasy_labels_per_output = self._sample_fantasy_labels_per_output(Xb)
            vals = []
            for f in range(self.num_fantasies):
                fantasy_submodels = []
                for o, submodel in enumerate(self.submodels):
                    resolved_lr = self._resolve_conditioning_lr_for_submodel(submodel)
                    resolved_bs = self._resolve_conditioning_batch_size_for_submodel(submodel)
                    with torch.enable_grad():
                        fantasy_model = submodel.condition_on_observations(
                            X=Xb,
                            Y=fantasy_labels_per_output[o][f].detach(),
                            refit=True,
                            num_steps=self.conditioning_steps,
                            lr=resolved_lr,
                            batch_size=resolved_bs,
                            verbose=False,
                        )
                    fantasy_submodels.append(fantasy_model)
                with torch.no_grad():
                    ipv = self._integrated_latent_variance_from_submodels(fantasy_submodels, ref_X=Xb)
                vals.append(-ipv)
            out.append(torch.stack(vals).mean())
        score = torch.stack(out).reshape(*batch_shape)
        score = self._apply_objective_to_scalar_score(score, X)
        penalty = self._aggregated_repulsion_penalty(X).to(device=score.device, dtype=score.dtype)
        if penalty.shape != score.shape:
            if penalty.numel() == score.numel():
                penalty = penalty.reshape_as(score)
            else:
                raise RuntimeError(
                    "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance penalty shape mismatch. "
                    f"score.shape={tuple(score.shape)}, penalty.shape={tuple(penalty.shape)}."
                )
        return score - penalty


class qMultiOutputOrdinalUtilityVarianceProxy(qMultiOutputOrdinalUtilityVariance):
    """Lightweight IPV-style proxy based on ordinal utility variance."""


__all__ = [
    "MultiOutputOrdinalScoreObjective",
    "_MultiOutputOrdinalScoreObjective",
    "qMultiOutputOrdinalPredictiveEntropy",
    "qMultiOutputOrdinalBALD",
    "qMultiOutputOrdinalUtilityVariance",
    "qMultiOutputOrdinalMarginUncertainty",
    "qMultiOutputOrdinalUtilityVarianceProxy",
    "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance",
]
