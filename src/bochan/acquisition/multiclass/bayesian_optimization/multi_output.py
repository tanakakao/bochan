from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
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
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.active_learning.multi_output import (
    OutputModeType,
    OutputReductionType,
    ReductionType,
    _DirectMultiOutputMulticlassAcqBase,
)
from bochan.acquisition.multiclass.base import ClassReductionType


class MulticlassTargetProbabilityObjective(MCMultiOutputObjective):
    """Convert multiclass probability samples to target-class objective values.

    The objective accepts either of the following shapes:

    - ``sample_shape x batch_shape x q x m``: already target probability / objective values.
    - ``sample_shape x batch_shape x q x m x C``: multiclass probabilities.

    In the latter case, it selects ``target_class`` for all outputs or
    ``output_target_classes[i]`` for output ``i``.
    """

    def __init__(
        self,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        num_outputs: int | None = None,
        class_reduction: ClassReductionType = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if target_class is None and output_target_classes is None:
            # Identity behavior for HybridMultiOutputModel or user-provided objective-scale models.
            self.target_class = None
            self.output_target_classes = None
        else:
            self.target_class = target_class
            self.output_target_classes = None if output_target_classes is None else [int(i) for i in output_target_classes]
        self.num_outputs = None if num_outputs is None else int(num_outputs)
        self.class_reduction = class_reduction
        self.eps = float(eps)

    def _reduce_classes(self, selected: Tensor) -> Tensor:
        if self.class_reduction == "mean":
            return selected.mean(dim=-1)
        if self.class_reduction == "sum":
            return selected.sum(dim=-1)
        if self.class_reduction == "max":
            return selected.max(dim=-1).values
        if self.class_reduction == "min":
            return selected.min(dim=-1).values
        if self.class_reduction == "prod":
            return selected.prod(dim=-1)
        raise ValueError(f"Unknown class_reduction: {self.class_reduction!r}.")

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        # Already scalar / objective multi-output samples: sample_shape x batch_shape x q x m.
        if self.num_outputs is not None and samples.shape[-1] == self.num_outputs:
            if samples.ndim < 5 or samples.shape[-2] != self.num_outputs:
                return samples

        # If no target is specified, keep identity behavior.
        if self.target_class is None and self.output_target_classes is None:
            return samples

        if samples.ndim < 2:
            raise RuntimeError(f"Multiclass samples must include class dimension. Got {tuple(samples.shape)}.")

        # Expected multiclass shape: ... x m x C.
        if self.output_target_classes is not None:
            n_outputs = int(samples.shape[-2])
            if len(self.output_target_classes) != n_outputs:
                raise ValueError(
                    "output_target_classes length must match number of outputs. "
                    f"Got {len(self.output_target_classes)} and {n_outputs}."
                )
            idx = torch.as_tensor(self.output_target_classes, device=samples.device, dtype=torch.long)
            gather_idx = idx.view(*([1] * (samples.ndim - 2)), n_outputs, 1).expand(*samples.shape[:-1], 1)
            return torch.gather(samples, dim=-1, index=gather_idx).squeeze(-1)

        if isinstance(self.target_class, int):
            return samples[..., int(self.target_class)]

        indices = [int(i) for i in self.target_class]
        selected = samples[..., indices]
        return self._reduce_classes(selected)


class _IdentityMCMultiOutputObjective(MCMultiOutputObjective):
    """Identity objective for qNParEGO when no explicit objective is supplied."""

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return samples


def ensure_q_batch(X: Tensor) -> Tensor:
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
        raise ValueError("utility_values tensor must be [C] or [m, C].")

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


def _reduce_observed_target_indicators(
    indicators: Tensor,
    *,
    class_reduction: ClassReductionType,
) -> Tensor:
    if class_reduction == "mean":
        return indicators.mean(dim=-1)
    if class_reduction == "sum":
        return indicators.sum(dim=-1)
    if class_reduction == "max":
        return indicators.max(dim=-1).values
    if class_reduction == "min":
        return indicators.min(dim=-1).values
    if class_reduction == "prod":
        return indicators.prod(dim=-1)
    raise ValueError(f"Unknown class_reduction: {class_reduction!r}.")


def compute_observed_multiclass_utility(
    train_Y: Tensor,
    *,
    target_class: int | Sequence[int] | None = None,
    output_target_classes: Sequence[int] | Tensor | None = None,
    class_reduction: ClassReductionType = "mean",
    utility_values: Optional[Sequence[Sequence[float]] | Sequence[float] | Tensor] = None,
    objective_signs: Optional[Sequence[float] | Tensor] = None,
    class_offset: int = 0,
) -> Tensor:
    """観測 multiclass label を multi-output BO の目的値空間へ変換する。

    Args:
        train_Y: 観測ラベル。shape は ``[n]`` または ``[n, m]``。
        target_class: 全出力共通で最大化したい class。複数 class を指定した場合は
            ``class_reduction`` で indicator を集約する。
        output_target_classes: 出力ごとに最大化したい class。shape は ``[m]``。
        class_reduction: ``target_class`` が複数 class のときの集約方法。
        utility_values: 指定した場合は label を utility table で直接変換する。
            ``target_class`` / ``output_target_classes`` より優先する。
        objective_signs: 出力ごとの符号。最小化したい目的は ``-1`` を指定する。
        class_offset: ラベルが 1 始まりなどの場合に差し引く offset。

    Returns:
        Tensor: shape ``[n, m]`` の観測目的値。

    Notes:
        - ``utility_values`` がある場合: ordinal の ``compute_observed_ordinal_utility``
          と同様に、観測 class label を utility table に写像する。
        - ``utility_values`` がない場合: 観測 class が target class なら 1、そうでなければ
          0 の target-class probability objective として扱う。
    """
    Y_raw = _as_2d_train_y(torch.as_tensor(train_Y))
    device = Y_raw.device
    dtype = torch.get_default_dtype() if not torch.is_floating_point(Y_raw) else Y_raw.dtype
    Y_idx = Y_raw.long() - int(class_offset)
    n, m = int(Y_idx.shape[0]), int(Y_idx.shape[1])

    signs = _normalize_objective_signs(
        objective_signs,
        m=m,
        device=device,
        dtype=dtype,
    )

    if utility_values is not None:
        utility_list = _to_utility_list(utility_values, device=device, dtype=dtype)
        if len(utility_list) == 1 and m > 1:
            utility_list = utility_list * m
        if len(utility_list) != m:
            raise ValueError(f"utility_values must have length 1 or m={m}, got {len(utility_list)}.")

        values = torch.empty(n, m, device=device, dtype=dtype)
        for j, utility_j in enumerate(utility_list):
            y_j = Y_idx[:, j]
            if y_j.numel() > 0:
                y_min = int(y_j.min().item())
                y_max = int(y_j.max().item())
                if y_min < 0 or y_max >= utility_j.numel():
                    raise ValueError(
                        f"train_Y[:, {j}] contains class outside utility_values[{j}] range. "
                        f"Got min={y_min}, max={y_max}, num_classes={utility_j.numel()}, "
                        f"class_offset={class_offset}."
                    )
            values[:, j] = utility_j[y_j]
        return values * signs.view(1, m)

    if output_target_classes is None and target_class is None:
        raise ValueError(
            "target_class or output_target_classes must be provided when utility_values is None. "
            "If train_Y is already objective-scale, pass it as Y_baseline directly."
        )

    if output_target_classes is not None:
        target_per_output = torch.as_tensor(output_target_classes, device=device, dtype=torch.long).reshape(-1)
        if target_per_output.numel() != m:
            raise ValueError(f"output_target_classes must have shape [{m}], got {tuple(target_per_output.shape)}.")
        values = (Y_idx == target_per_output.view(1, m)).to(dtype=dtype)
        return values * signs.view(1, m)

    if isinstance(target_class, int):
        values = (Y_idx == int(target_class)).to(dtype=dtype)
        return values * signs.view(1, m)

    target_classes = torch.as_tensor([int(i) for i in target_class], device=device, dtype=torch.long).reshape(-1)
    if target_classes.numel() == 0:
        raise ValueError("target_class sequence must not be empty.")
    indicators = (Y_idx.unsqueeze(-1) == target_classes.view(1, 1, -1)).to(dtype=dtype)
    values = _reduce_observed_target_indicators(indicators, class_reduction=class_reduction)
    return values * signs.view(1, m)


def compute_observed_multiclass_target_probability_values(
    train_Y: Tensor,
    *,
    target_class: int | Sequence[int] | None = None,
    output_target_classes: Sequence[int] | Tensor | None = None,
    class_reduction: ClassReductionType = "mean",
    objective_signs: Optional[Sequence[float] | Tensor] = None,
    class_offset: int = 0,
) -> Tensor:
    """target-class probability 目的専用の観測値変換 helper。"""
    return compute_observed_multiclass_utility(
        train_Y=train_Y,
        target_class=target_class,
        output_target_classes=output_target_classes,
        class_reduction=class_reduction,
        utility_values=None,
        objective_signs=objective_signs,
        class_offset=class_offset,
    )


def _squeeze_only_output_singleton(value: Tensor, X: Tensor) -> Tensor:
    q = int(X.shape[-2])
    batch_ndim = len(X.shape[:-2])
    min_ndim_with_q = batch_ndim + 1
    if value.ndim >= min_ndim_with_q + 1 and value.shape[-1] == 1 and value.shape[-2] == q:
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
            f"value.shape={tuple(value.shape)}, q={q}, batch_shape={tuple(batch_shape)}, X.shape={tuple(X.shape)}."
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
        "qMultiOutputMulticlassNParEGO produced invalid output shape after scalarization. "
        f"value.shape={tuple(value.shape)}, expected batch_shape={tuple(batch_shape)}, X.shape={tuple(X.shape)}."
    )


def _make_target_objective(
    *,
    ref_point: Tensor | Sequence[float] | None = None,
    target_class: int | Sequence[int] | None = None,
    output_target_classes: Sequence[int] | None = None,
    class_reduction: ClassReductionType = "mean",
    eps: float = 1e-8,
) -> MulticlassTargetProbabilityObjective:
    num_outputs = None if ref_point is None else int(torch.as_tensor(ref_point).numel())
    return MulticlassTargetProbabilityObjective(
        target_class=target_class,
        output_target_classes=output_target_classes,
        num_outputs=num_outputs,
        class_reduction=class_reduction,
        eps=eps,
    )


class _MultiOutputMulticlassTargetClassBOBase(_DirectMultiOutputMulticlassAcqBase):
    """Direct target-class probability BO base for PoF and legacy scalar variants."""

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        reduction: ReductionType = "mean",
        output_mode: OutputModeType = "mean",
        output_reduction: OutputReductionType | None = None,
        output_weights: Tensor | Sequence[float] | None = None,
        normalize_output_weights: bool = True,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-8,
        objective=None,
    ) -> None:
        if target_class is None and output_target_classes is None:
            raise ValueError(
                "target_class or output_target_classes must be specified for "
                "multi-output multiclass target-probability acquisitions."
            )
        super().__init__(
            model=model,
            reduction=reduction,
            output_mode=output_mode,
            output_reduction=output_reduction,
            output_weights=output_weights,
            normalize_output_weights=normalize_output_weights,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            objective=objective,
        )
        self.target_class = target_class
        self.output_target_classes = None if output_target_classes is None else [int(i) for i in output_target_classes]
        self.class_reduction = class_reduction

    def _reduce_classes(self, selected: Tensor) -> Tensor:
        return MulticlassTargetProbabilityObjective(
            target_class=self.target_class,
            output_target_classes=self.output_target_classes,
            class_reduction=self.class_reduction,
            eps=self.eps,
        )._reduce_classes(selected)

    def _target_prob_per_output(self, probs: Tensor) -> Tensor:
        return MulticlassTargetProbabilityObjective(
            target_class=self.target_class,
            output_target_classes=self.output_target_classes,
            num_outputs=probs.shape[-2],
            class_reduction=self.class_reduction,
            eps=self.eps,
        )(probs)

    def _target_prob_samples_per_output(self, X: Tensor, *, num_samples: int) -> Tensor:
        samples = self._sample_probs(X, num_samples=num_samples)
        return self._target_prob_per_output(samples)

    def _target_prob_mean_per_output(self, X: Tensor) -> Tensor:
        probs = self._mean_probs(X)
        return self._target_prob_per_output(probs)

    def _align_output_param(self, value: float | Tensor, *, ref: Tensor, name: str) -> Tensor:
        value_t = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
        if value_t.ndim == 0:
            return value_t
        n_outputs = int(ref.shape[-1])
        if value_t.numel() == n_outputs:
            return value_t.reshape(*([1] * (ref.ndim - 1)), n_outputs)
        if value_t.numel() == ref.numel():
            return value_t.reshape_as(ref)
        raise ValueError(
            f"{name} must be scalar, length n_outputs, or broadcastable to output score. "
            f"Got {tuple(value_t.shape)}, expected n_outputs={n_outputs}."
        )

    def _pending_q_penalty(self, Xt: Tensor) -> Tensor:
        if self.pending_penalty_weight <= 0:
            return Xt.new_zeros(Xt.shape[:-2])
        return self._pending_penalty_per_point(Xt).sum(dim=-1)


class qMultiOutputMulticlassProbabilityOfFeasibility(_MultiOutputMulticlassTargetClassBOBase):
    """Multi-output probability of target-class feasibility."""

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        threshold: float | None = None,
        tau: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            target_class=target_class,
            output_target_classes=output_target_classes,
            **kwargs,
        )
        self.threshold = None if threshold is None else float(threshold)
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        score_per_output = p if self.threshold is None else torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassExpectedHypervolumeImprovement(qExpectedHypervolumeImprovement):
    """BoTorch qEHVI for multiclass target-class probability objectives."""

    def __init__(
        self,
        model: Model,
        ref_point: Tensor | Sequence[float],
        partitioning: FastNondominatedPartitioning,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        eps: float = 1e-8,
    ) -> None:
        objective = objective or _make_target_objective(
            ref_point=ref_point,
            target_class=target_class,
            output_target_classes=output_target_classes,
            class_reduction=class_reduction,
            eps=eps,
        )
        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints or [],
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )


class qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    """BoTorch qNEHVI for multiclass target-class probability objectives."""

    def __init__(
        self,
        model: Model,
        ref_point: Tensor | Sequence[float],
        X_baseline: Tensor,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = True,
        marginalize_dim: Optional[int] = None,
        eps: float = 1e-8,
    ) -> None:
        objective = objective or _make_target_objective(
            ref_point=ref_point,
            target_class=target_class,
            output_target_classes=output_target_classes,
            class_reduction=class_reduction,
            eps=eps,
        )
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
            marginalize_dim=marginalize_dim,
        )


class qMultiOutputMulticlassNParEGO(MCAcquisitionFunction):
    """Multi-output multiclass qNParEGO-style acquisition.

    This mirrors the binary implementation: posterior samples are transformed by
    an MCMultiOutputObjective, scalarized with augmented Chebyshev, and evaluated
    as qEI on the scalarized objective.
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Tensor | Sequence[float],
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: ClassReductionType = "mean",
        utility_values: Optional[Sequence[Sequence[float]] | Sequence[float] | Tensor] = None,
        objective_signs: Optional[Sequence[float] | Tensor] = None,
        train_Y: Optional[Tensor] = None,
        Y_baseline: Optional[Tensor] = None,
        class_offset: int = 0,
        weights: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        rho: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        ref_tensor = torch.as_tensor(ref_point, device=X_baseline.device, dtype=X_baseline.dtype).reshape(-1)
        base_objective = objective or _make_target_objective(
            ref_point=ref_tensor,
            target_class=target_class,
            output_target_classes=output_target_classes,
            class_reduction=class_reduction,
            eps=eps,
        )
        if objective is None and target_class is None and output_target_classes is None:
            base_objective = _IdentityMCMultiOutputObjective()
        super().__init__(model=model, sampler=sampler, objective=base_objective)
        self.base_objective = base_objective
        self.eps = float(eps)
        self.num_outputs = int(ref_tensor.numel())
        self.rho = float(rho)

        if weights is None:
            w = torch.rand(self.num_outputs, device=X_baseline.device, dtype=X_baseline.dtype)
            weights = w / w.sum().clamp_min(self.eps)
        else:
            weights = weights.to(device=X_baseline.device, dtype=X_baseline.dtype)
            weights = weights / weights.sum().clamp_min(self.eps)
        self.register_buffer("weights", weights)
        self.register_buffer("ref_point", ref_tensor)

        if Y_baseline is None and train_Y is not None:
            if utility_values is None and target_class is None and output_target_classes is None:
                raise ValueError(
                    "target_class, output_target_classes, or utility_values must be provided "
                    "to build Y_baseline from train_Y. If train_Y is already objective-scale, "
                    "pass it as Y_baseline."
                )
            Y_baseline = compute_observed_multiclass_utility(
                train_Y=train_Y,
                target_class=target_class,
                output_target_classes=output_target_classes,
                class_reduction=class_reduction,
                utility_values=utility_values,
                objective_signs=objective_signs,
                class_offset=class_offset,
            )

        with torch.no_grad():
            if Y_baseline is not None:
                values = Y_baseline.to(device=X_baseline.device, dtype=X_baseline.dtype).unsqueeze(0).unsqueeze(0)
            else:
                Xb = ensure_q_batch(X_baseline)
                post = model.posterior(Xb)
                values = self.base_objective(post.mean.unsqueeze(0), X=Xb)
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
        return weighted.min(dim=-1).values + self.rho * weighted.sum(dim=-1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        post = self.model.posterior(Xq)
        samples = self.get_posterior_samples(post)
        values = self.base_objective(samples, X=Xq)
        scalarized = self._scalarize(values)
        improvement = (scalarized - self.best_value.to(scalarized)).clamp_min(0.0)
        return _reduce_sample_and_q_to_tbatch(improvement, Xq)


class qMultiOutputMulticlassExpectedImprovement(_MultiOutputMulticlassTargetClassBOBase):
    """Legacy scalar EI for target-class probability."""

    def __init__(self, model, *, target_class: int | Sequence[int] | None = None, output_target_classes: Sequence[int] | None = None, best_f: float | Tensor, num_samples: int = 128, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, output_target_classes=output_target_classes, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        best_q_per_output = samples.max(dim=-2).values
        best_f = self._align_output_param(self.best_f, ref=best_q_per_output, name="best_f")
        value_per_output = (best_q_per_output - best_f).clamp_min(0.0).mean(dim=0)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassProbabilityOfImprovement(_MultiOutputMulticlassTargetClassBOBase):
    """Legacy scalar PI for target-class probability."""

    def __init__(self, model, *, target_class: int | Sequence[int] | None = None, output_target_classes: Sequence[int] | None = None, best_f: float | Tensor, num_samples: int = 128, tau: float = 1e-3, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, output_target_classes=output_target_classes, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        best_q_per_output = samples.max(dim=-2).values
        best_f = self._align_output_param(self.best_f, ref=best_q_per_output, name="best_f")
        tau = self.tau.to(best_q_per_output).clamp_min(self.eps)
        value_per_output = torch.sigmoid((best_q_per_output - best_f) / tau).mean(dim=0)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qMultiOutputMulticlassUpperConfidenceBound(_MultiOutputMulticlassTargetClassBOBase):
    """Legacy scalar UCB for target-class probability."""

    def __init__(self, model, *, target_class: int | Sequence[int] | None = None, output_target_classes: Sequence[int] | None = None, beta: float | Tensor = 2.0, num_samples: int = 128, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, output_target_classes=output_target_classes, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=False).clamp_min(self.eps)
        beta = self._align_output_param(self.beta, ref=mean, name="beta")
        score_per_output = mean + beta.sqrt() * std
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


__all__ = [
    "MulticlassTargetProbabilityObjective",
    "compute_observed_multiclass_utility",
    "compute_observed_multiclass_target_probability_values",
    "OutputReductionType",
    "OutputModeType",
    "qMultiOutputMulticlassProbabilityOfFeasibility",
    "qMultiOutputMulticlassExpectedHypervolumeImprovement",
    "qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    "qMultiOutputMulticlassNParEGO",
    "qMultiOutputMulticlassExpectedImprovement",
    "qMultiOutputMulticlassProbabilityOfImprovement",
    "qMultiOutputMulticlassUpperConfidenceBound",
]
