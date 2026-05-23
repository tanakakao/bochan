
from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import Tensor

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform

from ._utils import (
    ensure_q_batch,
    get_model_posterior,
    normalize_mean_shape,
    reshape_samples,
    shape_X_for_model,
    to_probability,
)


def _get_noise_posterior(model: Model, X: Tensor):
    X = ensure_q_batch(X)
    for name in ("posterior_noise", "noise_posterior"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn(X)

    if hasattr(model, "models"):
        from ._utils import _StackedPosterior
        return _StackedPosterior([_get_noise_posterior(sm, X) for sm in model.models])

    noise_model = getattr(model, "noise_model", None)
    if noise_model is None:
        inner = getattr(model, "model", None)
        noise_model = getattr(inner, "noise_model", None) if inner is not None else None

    if noise_model is None:
        raise AttributeError("Noise posterior was not found.")
    return noise_model.posterior(X)


def _get_noise_std(
    model: Model,
    X: Tensor,
    *,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-6,
    shape_X: Optional[Tensor] = None,
) -> Tensor:
    X = ensure_q_batch(X)
    shape_X = shape_X_for_model(model, X) if shape_X is None else shape_X

    try:
        noise_post = _get_noise_posterior(model, X)
        noise_mean = normalize_mean_shape(noise_post.mean, shape_X)
        if noise_is_log_var:
            noise_var = torch.exp(noise_mean.clamp(min=math.log(eps), max=30.0))
        else:
            noise_var = noise_mean.clamp_min(eps)
        return noise_var.sqrt().clamp_min(eps)
    except Exception:
        post = get_model_posterior(model, X, samples_are_probs=True)
        mean = normalize_mean_shape(post.mean, shape_X)
        return torch.full_like(mean, float(default_sigma))


def hetero_adjust_classification_samples(
    model: Model,
    X: Tensor,
    samples: Tensor,
    *,
    beta: float = 0.0,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    samples_are_probs: bool = True,
    apply_sigmoid_if_needed: bool = True,
    eps: float = 1e-6,
    posterior=None,
) -> Tensor:
    X = ensure_q_batch(X)
    shape_X = shape_X_for_model(model, X)

    if posterior is None:
        posterior = get_model_posterior(model, X, samples_are_probs=samples_are_probs)

    mean_prob = normalize_mean_shape(posterior.mean, shape_X)
    mean_prob = to_probability(
        mean_prob,
        apply_sigmoid_if_needed=apply_sigmoid_if_needed,
        eps=eps,
        name="posterior.mean",
    )

    samples = reshape_samples(samples, shape_X)
    samples = to_probability(
        samples,
        apply_sigmoid_if_needed=(not samples_are_probs) or apply_sigmoid_if_needed,
        eps=eps,
        name="posterior samples",
    )

    sigma = _get_noise_std(
        model,
        X,
        default_sigma=default_sigma,
        noise_is_log_var=noise_is_log_var,
        eps=eps,
        shape_X=shape_X,
    )

    adjusted = mean_prob.unsqueeze(0) + float(beta) * (samples - mean_prob.unsqueeze(0))
    adjusted = adjusted - float(noise_penalty) * sigma.unsqueeze(0)
    return adjusted.clamp(eps, 1.0 - eps)


def compute_hetero_multi_output_classification_train_y(
    model: Model,
    train_X: Tensor,
    *,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    apply_sigmoid_if_needed: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    with torch.no_grad():
        X = ensure_q_batch(train_X)
        shape_X = shape_X_for_model(model, X)
        post = get_model_posterior(model, X, samples_are_probs=True)
        mean = normalize_mean_shape(post.mean, shape_X)
        mean = to_probability(mean, apply_sigmoid_if_needed=apply_sigmoid_if_needed, eps=eps, name="posterior.mean")
        sigma = _get_noise_std(
            model,
            X,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
            eps=eps,
            shape_X=shape_X,
        )
        adjusted = (mean - float(noise_penalty) * sigma).clamp(eps, 1.0 - eps)
        return adjusted.reshape(-1, adjusted.shape[-1])


class _HeteroClassificationMCMultiOutputObjective(MCMultiOutputObjective):
    """BoTorch qNEHVI/qNParEGO 内で使う hetero-adjusted classification objective。"""

    def __init__(
        self,
        *,
        base_objective: Optional[MCMultiOutputObjective],
        model: Model,
        beta: float = 1.0,
        noise_penalty: float = 0.3,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.base_objective = base_objective
        self.model = model
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if X is None:
            raise ValueError("X must be provided for _HeteroClassificationMCMultiOutputObjective.")
        adjusted = hetero_adjust_classification_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
            samples_are_probs=self.samples_are_probs,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
        )
        if self.base_objective is None:
            return adjusted
        return self.base_objective(adjusted, X=X)


class _qHeteroMultiOutputBinaryDecoupledExpectedHypervolumeImprovement(MCAcquisitionFunction):
    """Decoupled 近似の hetero classification qEHVI。"""

    def __init__(
        self,
        model: Model,
        partitioning: FastNondominatedPartitioning,
        *,
        beta: float = 1.0,
        noise_penalty: float = 0.3,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        **kwargs,
    ) -> None:
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        objective = objective or IdentityMCMultiOutputObjective()
        super().__init__(model=model, sampler=sampler, objective=objective, **kwargs)
        self.partitioning = partitioning
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = get_model_posterior(self.model, X, samples_are_probs=self.samples_are_probs)
        samples = post.rsample(self.sampler.sample_shape)
        hetero = hetero_adjust_classification_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
            samples_are_probs=self.samples_are_probs,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            posterior=post,
        )
        obj = self.objective(hetero, X=X)
        m = obj.shape[-1]
        cell_lower = self.partitioning.hypercell_bounds[0].view(1, 1, 1, -1, m)
        cell_upper = self.partitioning.hypercell_bounds[1].view(1, 1, 1, -1, m)
        pts = obj.unsqueeze(-2)
        overlap = (torch.min(pts, cell_upper) - cell_lower).clamp_min(0.0)
        hvi = overlap.prod(dim=-1).sum(dim=-1).sum(dim=-1)
        return hvi.mean(dim=0)


class qHeteroMultiOutputBinaryExpectedHypervolumeImprovement(qExpectedHypervolumeImprovement):
    """heteroscedastic multi-output classification 用 qEHVI acquisition。Pareto hypervolume の期待改善量を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        ref_point: 多目的 hypervolume 計算で使う参照点。すべての目的が最大化方向に揃っている必要があります。
        partitioning: EHVI で使う non-dominated partitioning。通常は `FastNondominatedPartitioning` を渡します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        noise_penalty: heteroscedastic noise を避けるための penalty 係数。大きいほど noise の大きい点を避けます。
        default_sigma: noise posterior を取得できない場合に使う fallback の noise 標準偏差。
        noise_is_log_var: noise model の出力を log variance として扱うかどうか。
        samples_are_probs: posterior samples が probability 空間の値かどうか。False の場合は sigmoid 変換を検討します。
        apply_sigmoid_if_needed: posterior mean / samples が [0, 1] にない場合に sigmoid 変換するかどうか。
        eps: 数値安定化用の微小値。
        sampler: posterior samples を生成する BoTorch sampler。省略時は SobolQMCNormalSampler を使います。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
        constraints: BoTorch の constraint callable のリスト。
        X_pending: 評価中で、まだ結果が返っていない候補点。重複候補の抑制に使います。
        eta: constraint smoothing に使う温度パラメータ。
        fat: BoTorch の smoothed feasibility で fat-tailed approximation を使うかどうか。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        全目的は最大化方向に揃えてください。classification / ordinal では probability または utility objective を通して目的空間に変換します。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(
        self,
        model: Model,
        ref_point: Union[Tensor, list[float]],
        partitioning: FastNondominatedPartitioning,
        *,
        beta: float = 1.0,
        noise_penalty: float = 0.3,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective or IdentityMCMultiOutputObjective(),
            constraints=constraints or [],
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)
        self.eta = eta
        self.fat = fat

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = get_model_posterior(self.model, X, samples_are_probs=self.samples_are_probs)
        samples = post.rsample(self.sampler.sample_shape)
        hetero = hetero_adjust_classification_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
            samples_are_probs=self.samples_are_probs,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            posterior=post,
        )
        return self._compute_qehvi(samples=hetero, X=X)


class qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    """heteroscedastic multi-output classification 用 qEHVI acquisition。Pareto hypervolume の期待改善量を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        ref_point: 多目的 hypervolume 計算で使う参照点。すべての目的が最大化方向に揃っている必要があります。
        X_baseline: noisy acquisition や NParEGO の baseline 点。通常は既存観測点を渡します。
        sampler: posterior samples を生成する BoTorch sampler。省略時は SobolQMCNormalSampler を使います。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
        constraints: BoTorch の constraint callable のリスト。
        X_pending: 評価中で、まだ結果が返っていない候補点。重複候補の抑制に使います。
        eta: constraint smoothing に使う温度パラメータ。
        fat: BoTorch の smoothed feasibility で fat-tailed approximation を使うかどうか。
        prune_baseline: qNEHVI で baseline を prune するかどうか。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        cache_pending: qNEHVI で pending cache を使うかどうか。
        max_iep: qNEHVI の inclusion-exclusion pruning に関する上限。
        incremental_nehvi: incremental NEHVI を使うかどうか。
        cache_root: root decomposition を cache するかどうか。
        marginalize_dim: fully Bayesian model などで marginalize する batch 次元。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        noise_penalty: heteroscedastic noise を避けるための penalty 係数。大きいほど noise の大きい点を避けます。
        default_sigma: noise posterior を取得できない場合に使う fallback の noise 標準偏差。
        noise_is_log_var: noise model の出力を log variance として扱うかどうか。
        samples_are_probs: posterior samples が probability 空間の値かどうか。False の場合は sigmoid 変換を検討します。
        apply_sigmoid_if_needed: posterior mean / samples が [0, 1] にない場合に sigmoid 変換するかどうか。
        eps: 数値安定化用の微小値。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        全目的は最大化方向に揃えてください。classification / ordinal では probability または utility objective を通して目的空間に変換します。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        X_baseline: Tensor,
        *,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = True,
        marginalize_dim: Optional[int] = None,
        beta: float = 1.0,
        noise_penalty: float = 0.3,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
    ) -> None:
        base_objective = objective or IdentityMCMultiOutputObjective()
        hetero_objective = _HeteroClassificationMCMultiOutputObjective(
            base_objective=base_objective,
            model=model,
            beta=beta,
            noise_penalty=noise_penalty,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
            samples_are_probs=samples_are_probs,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
        )
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=hetero_objective,
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
        self.base_objective = base_objective
        self.hetero_objective = hetero_objective


class qHeteroMultiOutputBinaryNParEGO(MCAcquisitionFunction):
    """heteroscedastic multi-output classification 用 qNParEGO acquisition。多目的を scalarization して qEI 的に評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        X_baseline: noisy acquisition や NParEGO の baseline 点。通常は既存観測点を渡します。
        ref_point: 多目的 hypervolume 計算で使う参照点。すべての目的が最大化方向に揃っている必要があります。
        weights: NParEGO や weighted objective で使う scalarization weight。
        sampler: posterior samples を生成する BoTorch sampler。省略時は SobolQMCNormalSampler を使います。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        noise_penalty: heteroscedastic noise を避けるための penalty 係数。大きいほど noise の大きい点を避けます。
        default_sigma: noise posterior を取得できない場合に使う fallback の noise 標準偏差。
        noise_is_log_var: noise model の出力を log variance として扱うかどうか。
        samples_are_probs: posterior samples が probability 空間の値かどうか。False の場合は sigmoid 変換を検討します。
        apply_sigmoid_if_needed: posterior mean / samples が [0, 1] にない場合に sigmoid 変換するかどうか。
        eps: 数値安定化用の微小値。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        目的数が多い場合や qEHVI / qNEHVI が重い場合の実用的な代替です。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Tensor,
        *,
        weights: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        beta: float = 1.0,
        noise_penalty: float = 0.3,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
    ) -> None:
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        tkwargs = {"dtype": X_baseline.dtype, "device": X_baseline.device}
        m = int(ref_point.numel())
        if weights is None:
            w = torch.rand(m, **tkwargs)
            weights = w / w.sum()
        else:
            weights = weights.to(**tkwargs)
            weights = weights / weights.sum().clamp_min(1e-12)
        base_objective = WeightedMCMultiOutputObjective(weights=weights)
        super().__init__(model=model, sampler=sampler, objective=base_objective)
        self.base_objective = base_objective
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)

        with torch.no_grad():
            y = compute_hetero_multi_output_classification_train_y(
                model,
                X_baseline,
                noise_penalty=noise_penalty,
                default_sigma=default_sigma,
                noise_is_log_var=noise_is_log_var,
                apply_sigmoid_if_needed=apply_sigmoid_if_needed,
                eps=eps,
            )
            obj_train = self.base_objective(y.unsqueeze(0).unsqueeze(0), X=X_baseline.unsqueeze(0)).squeeze()
            self.register_buffer("best_value", obj_train.max())

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = get_model_posterior(self.model, X, samples_are_probs=self.samples_are_probs)
        samples = post.rsample(self.sampler.sample_shape)
        hetero = hetero_adjust_classification_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
            samples_are_probs=self.samples_are_probs,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            posterior=post,
        )
        scalarized = self.base_objective(hetero, X=X)
        best_q = scalarized.max(dim=-1).values
        return (best_q - self.best_value.to(best_q)).clamp_min(0.0).mean(dim=0)

__all__ = [
    "qHeteroMultiOutputBinaryExpectedHypervolumeImprovement",
    "qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputBinaryNParEGO",
]
