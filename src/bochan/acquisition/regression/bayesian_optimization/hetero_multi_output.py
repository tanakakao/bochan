
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
)
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform


def _ensure_q_batch(X: Tensor) -> Tensor:
    """Normalize inputs while preserving BoTorch baseline semantics.

    For MC multi-objective objectives, a 2D tensor ``n x d`` uses ``n`` as the
    q-like dimension. This matters for qNEHVI baseline evaluation where
    ``X_baseline`` is passed to ``objective(samples, X=X_baseline)`` and BoTorch
    verifies that the objective output q dimension equals ``X_baseline.shape[-2]``.

    Therefore, do not convert ``n x d`` to ``n x 1 x d`` here. Only canonicalize
    a single raw point ``d`` to ``1 x d``.
    """
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.unsqueeze(0)
    return X


def _get_noise_posterior(model: Model, X: Tensor):
    for name in ("posterior_noise", "noise_posterior"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn(X)

    noise_model = getattr(model, "noise_model", None)
    if noise_model is None:
        inner = getattr(model, "model", None)
        noise_model = getattr(inner, "noise_model", None) if inner is not None else None

    if noise_model is None:
        raise AttributeError(
            "Noise posterior was not found. Expected model.posterior_noise, "
            "model.noise_posterior, model.noise_model, or model.model.noise_model."
        )
    return noise_model.posterior(X)


def _normalize_output_shape(Y: Tensor, X: Tensor, name: str) -> Tensor:
    X = _ensure_q_batch(X)
    expected_prefix = X.shape[:-1]

    if Y.ndim >= 1 and Y.shape[-1] == 1 and len(expected_prefix) == Y.ndim - 1:
        # keep as output dim if it is actually m=1
        pass

    if Y.shape == expected_prefix:
        return Y.unsqueeze(-1)

    if Y.ndim == X.ndim and Y.shape[:-1] == expected_prefix:
        return Y

    n = math.prod(expected_prefix)
    if Y.numel() % n == 0:
        m = Y.numel() // n
        return Y.reshape(*expected_prefix, m)

    raise RuntimeError(
        f"{name}: cannot normalize shape. X={tuple(X.shape)}, Y={tuple(Y.shape)}"
    )


def get_hetero_noise_std(
    model: Model,
    X: Tensor,
    *,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    X = _ensure_q_batch(X)
    try:
        noise_post = _get_noise_posterior(model, X)
        noise_mean = _normalize_output_shape(noise_post.mean, X, "noise mean")
        if noise_is_log_var:
            noise_var = torch.exp(noise_mean.clamp(min=math.log(eps), max=30.0))
        else:
            noise_var = noise_mean.clamp_min(eps)
        return noise_var.sqrt().clamp_min(eps)
    except Exception:
        post = model.posterior(X)
        mean = _normalize_output_shape(post.mean, X, "posterior mean")
        return torch.full_like(mean, float(default_sigma))


def hetero_adjust_regression_samples(
    model: Model,
    X: Tensor,
    samples: Tensor,
    *,
    beta: float = 0.0,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-8,
    posterior=None,
) -> Tensor:
    """Regression samples を heteroscedastic noise-aware に補正する。

    adjusted = mean + beta * (sample - mean) - noise_penalty * sigma_noise
    """
    X = _ensure_q_batch(X)
    if posterior is None:
        posterior = model.posterior(X)

    mean = _normalize_output_shape(posterior.mean, X, "posterior mean")
    samples = _normalize_samples(samples, X, mean.shape[-1])
    sigma = get_hetero_noise_std(
        model,
        X,
        default_sigma=default_sigma,
        noise_is_log_var=noise_is_log_var,
        eps=eps,
    )

    if sigma.shape != mean.shape and sigma.numel() == mean.numel():
        sigma = sigma.reshape_as(mean)

    return mean.unsqueeze(0) + float(beta) * (samples - mean.unsqueeze(0)) - float(noise_penalty) * sigma.unsqueeze(0)


def _normalize_samples(samples: Tensor, X: Tensor, m: int) -> Tensor:
    X = _ensure_q_batch(X)
    expected_prefix = X.shape[:-1]
    s = samples
    sample_shape = s.shape[: max(1, s.ndim - len(expected_prefix) - 1)]
    sample_numel = math.prod(sample_shape) if len(sample_shape) > 0 else 1
    n = math.prod(expected_prefix)

    if s.ndim >= 1 and s.shape[-1] == m and s.shape[-len(expected_prefix)-1:-1] == expected_prefix:
        return s

    if s.numel() % (sample_numel * n * m) == 0:
        return s.reshape(*sample_shape, *expected_prefix, m)

    # fallback: infer sample shape from first dim
    S = s.shape[0]
    return s.reshape(S, *expected_prefix, m)


def _chebyshev_scalarize_objectives(
    Y: Tensor,
    *,
    weights: Tensor,
    ref_point: Tensor,
    rho: float = 0.05,
) -> Tensor:
    """Augmented Chebyshev scalarization for maximization objectives.

    Args:
        Y: Objective values with shape ``... x m``.
        weights: Non-negative scalarization weights with shape ``m``.
        ref_point: Reference point with shape ``m``. Objectives are assumed to
            be maximized and improvement is measured as ``Y - ref_point``.
        rho: Small augmentation coefficient. Larger values encourage balanced
            improvement while still keeping the Chebyshev term dominant.

    Returns:
        Scalarized values with shape ``...``.
    """
    if Y.shape[-1] != weights.numel():
        raise RuntimeError(
            "Y output dimension and weights length must match. "
            f"Got Y.shape={tuple(Y.shape)}, weights.shape={tuple(weights.shape)}."
        )
    if ref_point.numel() != weights.numel():
        raise RuntimeError(
            "ref_point length and weights length must match. "
            f"Got ref_point.shape={tuple(ref_point.shape)}, weights.shape={tuple(weights.shape)}."
        )

    weights = weights.to(device=Y.device, dtype=Y.dtype).clamp_min(1e-12)
    weights = weights / weights.sum().clamp_min(1e-12)
    ref_point = ref_point.to(device=Y.device, dtype=Y.dtype)

    view_shape = *([1] * (Y.ndim - 1)), -1
    improvement = Y - ref_point.view(view_shape)
    weighted = weights.view(view_shape) * improvement
    return weighted.min(dim=-1).values + float(rho) * weighted.sum(dim=-1)


def compute_hetero_multi_output_regression_train_y(
    model: Model,
    train_X: Tensor,
    *,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """学習点上の hetero-adjusted mean を返す。shape = (n, m)."""
    with torch.no_grad():
        X = _ensure_q_batch(train_X)
        post = model.posterior(X)
        mean = _normalize_output_shape(post.mean, X, "train posterior mean")
        sigma = get_hetero_noise_std(
            model,
            X,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
            eps=eps,
        )
        y = mean - float(noise_penalty) * sigma
        return y.reshape(-1, y.shape[-1])


class qHeteroMultiOutputRegressionDecoupledExpectedHypervolumeImprovement(MCAcquisitionFunction):
    """heteroscedastic multi-output regression 用 qEHVI acquisition。Pareto hypervolume の期待改善量を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        partitioning: EHVI で使う non-dominated partitioning。通常は `FastNondominatedPartitioning` を渡します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        noise_penalty: heteroscedastic noise を避けるための penalty 係数。大きいほど noise の大きい点を避けます。
        default_sigma: noise posterior を取得できない場合に使う fallback の noise 標準偏差。
        noise_is_log_var: noise model の出力を log variance として扱うかどうか。
        sampler: posterior samples を生成する BoTorch sampler。省略時は SobolQMCNormalSampler を使います。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
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
        partitioning: FastNondominatedPartitioning,
        *,
        beta: float = 2.0,
        noise_penalty: float = 2.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
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

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)
        hetero = hetero_adjust_regression_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
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


class qHeteroMultiOutputRegressionExpectedHypervolumeImprovement(qExpectedHypervolumeImprovement):
    """heteroscedastic multi-output regression 用 qEHVI acquisition。Pareto hypervolume の期待改善量を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        ref_point: 多目的 hypervolume 計算で使う参照点。すべての目的が最大化方向に揃っている必要があります。
        partitioning: EHVI で使う non-dominated partitioning。通常は `FastNondominatedPartitioning` を渡します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        noise_penalty: heteroscedastic noise を避けるための penalty 係数。大きいほど noise の大きい点を避けます。
        default_sigma: noise posterior を取得できない場合に使う fallback の noise 標準偏差。
        noise_is_log_var: noise model の出力を log variance として扱うかどうか。
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
        beta: float = 2.0,
        noise_penalty: float = 2.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
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
            objective=objective,
            constraints=constraints or [],
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )
        self.beta_ht = float(beta)
        self.noise_penalty_ht = float(noise_penalty)
        self.default_sigma_ht = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.eta = eta

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)
        hetero = hetero_adjust_regression_samples(
            self.model,
            X,
            samples,
            beta=self.beta_ht,
            noise_penalty=self.noise_penalty_ht,
            default_sigma=self.default_sigma_ht,
            noise_is_log_var=self.noise_is_log_var,
            posterior=post,
        )
        return self._compute_qehvi(samples=hetero, X=X)


class _HeteroRegressionMCMultiOutputObjective(MCMultiOutputObjective):
    """BoTorch qNEHVI 内で使う regression hetero-adjusted objective。"""

    def __init__(
        self,
        *,
        base_objective: Optional[MCMultiOutputObjective],
        model: Model,
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
    ) -> None:
        super().__init__()
        self.base_objective = base_objective
        self.model = model
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if X is None:
            raise ValueError("X must be provided for _HeteroRegressionMCMultiOutputObjective.")
        adjusted = hetero_adjust_regression_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
        )
        if self.base_objective is None:
            return adjusted
        return self.base_objective(adjusted, X=X)


class qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    """heteroscedastic multi-output regression 用 qEHVI acquisition。Pareto hypervolume の期待改善量を評価します。
    
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
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
    ) -> None:
        base_objective = objective or IdentityMCMultiOutputObjective()
        hetero_objective = _HeteroRegressionMCMultiOutputObjective(
            base_objective=base_objective,
            model=model,
            beta=beta,
            noise_penalty=noise_penalty,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
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


class qHeteroMultiOutputRegressionNParEGO(MCAcquisitionFunction):
    """heteroscedastic multi-output regression 用 qNParEGO acquisition。多目的を scalarization して qEI 的に評価します。
    
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
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        rho: float = 0.05,
    ) -> None:
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        X_baseline = X_baseline.detach()
        tkwargs = {"dtype": X_baseline.dtype, "device": X_baseline.device}
        ref_point = ref_point.to(**tkwargs)
        m = int(ref_point.numel())

        if weights is None:
            w = torch.rand(m, **tkwargs)
            weights = w / w.sum().clamp_min(1e-12)
        else:
            weights = weights.to(**tkwargs)
            weights = weights / weights.sum().clamp_min(1e-12)

        super().__init__(
            model=model,
            sampler=sampler,
            objective=IdentityMCMultiOutputObjective(),
        )
        self.X_baseline = X_baseline
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.rho = float(rho)
        self.register_buffer("weights", weights)
        self.register_buffer("ref_point", ref_point)

        with torch.no_grad():
            y = compute_hetero_multi_output_regression_train_y(
                model,
                X_baseline,
                noise_penalty=noise_penalty,
                default_sigma=default_sigma,
                noise_is_log_var=noise_is_log_var,
            )
            obj_train = _chebyshev_scalarize_objectives(
                y,
                weights=self.weights,
                ref_point=self.ref_point,
                rho=self.rho,
            )
            self.register_buffer("best_value", obj_train.max())

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)
        hetero = hetero_adjust_regression_samples(
            self.model,
            X,
            samples,
            beta=self.beta,
            noise_penalty=self.noise_penalty,
            default_sigma=self.default_sigma,
            noise_is_log_var=self.noise_is_log_var,
            posterior=post,
        )
        scalarized = _chebyshev_scalarize_objectives(
            hetero,
            weights=self.weights,
            ref_point=self.ref_point,
            rho=self.rho,
        )
        best_q = scalarized.max(dim=-1).values
        return (best_q - self.best_value.to(best_q)).clamp_min(0.0).mean(dim=0)

__all__ = [
    "qHeteroMultiOutputRegressionDecoupledExpectedHypervolumeImprovement",
    "qHeteroMultiOutputRegressionExpectedHypervolumeImprovement",
    "qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputRegressionNParEGO",
]
