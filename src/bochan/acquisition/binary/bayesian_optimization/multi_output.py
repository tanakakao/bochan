
from __future__ import annotations

from typing import Callable, Literal, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Normal

from botorch.acquisition import AcquisitionFunction
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
    MultiOutputMode,
    PoFMode,
    ReductionType,
    aggregate_outputs,
    apply_pointwise_score_objective,
    ensure_q_batch,
    get_model_posterior,
    normalize_mean_shape,
    reduce_q,
    reshape_samples,
    shape_X_for_model,
    to_probability,
)



def _get_binary_mc_posterior_for_probability_samples(
    model,
    X: Tensor,
    *,
    samples_are_probs: bool,
    prefer_latent: bool = True,
):
    """classification BO 用の posterior 取得 helper。

    BinaryClassificationGPModel 系では、
        posterior(X)        -> probability posterior
        latent_posterior(X) -> latent f posterior
    という設計が多い。

    MC acquisition で sigmoid 変換を使う場合、posterior(X).rsample() に
    sigmoid をかけるより、latent_posterior(X).rsample() に sigmoid をかける方が
    意味的に自然。そのため samples_are_probs=False では latent_posterior を優先する。
    """
    if (not samples_are_probs) and prefer_latent:
        latent_fn = getattr(model, "latent_posterior", None)
        if callable(latent_fn):
            return latent_fn(X)

    return get_model_posterior(model, X, samples_are_probs=samples_are_probs)


def _detach_optional_tensor(X):
    if X is None:
        return None
    if torch.is_tensor(X):
        return X.detach()
    return X


class qMultiOutputBinaryProbabilityOfFeasibility(AcquisitionFunction):
    """multi-output binary classification 用 probability of feasibility acquisition。実現可能確率を最大化します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        num_samples: classification probability や BALD などを MC 近似する sample 数。
        threshold: binary classification や level-set で使う境界値。
        mode: 獲得関数の計算モード。例: `mc_sigmoid` または `latent_cdf`。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        output_mode: classification multi-output の出力方向集約方法。`mean`、`weighted_mean`、`all_positive` など。
        output_weights: multi-output score を集約するときの出力方向の重み。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        pending_penalty_beta: X_pending penalty の距離減衰率。
        samples_are_probs: posterior samples が probability 空間の値かどうか。False の場合は sigmoid 変換を検討します。
        mean_is_latent: この acquisition / objective の動作を制御するパラメータ。
        apply_sigmoid_if_needed: posterior mean / samples が [0, 1] にない場合に sigmoid 変換するかどうか。
        eps: 数値安定化用の微小値。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model,
        num_samples: int = 32,
        threshold: float = 0.0,
        mode: PoFMode = "mc_sigmoid",
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "all_positive",
        output_weights: Optional[Tensor] = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        samples_are_probs: bool = False,
        mean_is_latent: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)
        self.num_samples = int(num_samples)
        self.threshold = float(threshold)
        self.mode = mode
        self.reduction = reduction
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.samples_are_probs = bool(samples_are_probs)
        self.mean_is_latent = bool(mean_is_latent)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)
        self.objective = objective
        self.set_X_pending(None)

    def _coerce_pending_to_tensor(
        self,
        X_pending,
        *,
        ref: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """X_pending を Tensor または None に正規化する。"""
        if X_pending is None:
            return None
        if torch.is_tensor(X_pending):
            out = X_pending
        elif isinstance(X_pending, (list, tuple)):
            tensors = []
            for item in X_pending:
                if item is None:
                    continue
                t = self._coerce_pending_to_tensor(item, ref=ref)
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

        # X_pending は acquisition optimization 中の定数として扱う。
        return out.detach()

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        """pending points を raw input space の値として保持する。"""
        self.X_pending = self._coerce_pending_to_tensor(X_pending)

    def _transform_pending_like_candidate(
        self,
        X_pending,
        *,
        ref: Tensor,
    ) -> Optional[Tensor]:
        """X_pending を expanded_X と同じ距離計算空間へ写す。"""
        Xp = self._coerce_pending_to_tensor(X_pending, ref=ref)
        if Xp is None or Xp.numel() == 0:
            return None
        Xp_t = shape_X_for_model(self.model, ensure_q_batch(Xp))
        return Xp_t.to(device=ref.device, dtype=ref.dtype)

    def _pending_penalty_per_point(self, expanded_X: Tensor) -> Tensor:
        """
        pending points に近い候補点へ pointwise penalty を与える。

        Args:
            expanded_X:
                候補点。すでに `shape_X_for_model(model, raw_X)` を通した
                距離計算用 Tensor。shape は `(*batch, q_like, d)`。

        Returns:
            Tensor:
                pending penalty。shape は `(*batch, q_like)`。
        """
        expanded_X = ensure_q_batch(expanded_X)

        if self.pending_penalty_weight <= 0.0:
            return torch.zeros(expanded_X.shape[:-1], device=expanded_X.device, dtype=expanded_X.dtype)

        Xp_t = self._transform_pending_like_candidate(
            getattr(self, "X_pending", None),
            ref=expanded_X,
        )
        if Xp_t is None or Xp_t.numel() == 0:
            return torch.zeros(expanded_X.shape[:-1], device=expanded_X.device, dtype=expanded_X.dtype)

        d = expanded_X.shape[-1]
        X2d = expanded_X.reshape(-1, d)
        Xp2d = Xp_t.reshape(-1, Xp_t.shape[-1])

        if Xp2d.shape[-1] != d:
            raise RuntimeError(
                "X_pending feature dimension mismatch in pending penalty after transform: "
                f"expanded_X.shape={tuple(expanded_X.shape)}, X_pending_transformed.shape={tuple(Xp_t.shape)}."
            )

        dist = torch.cdist(X2d, Xp2d).min(dim=-1).values.reshape(*expanded_X.shape[:-1])
        return self.pending_penalty_weight * torch.exp(-self.pending_penalty_beta * dist)

    def _pointwise_pof(self, raw_X: Tensor, expanded_X: Tensor) -> Tensor:
        if self.mode == "mc_sigmoid":
            post = _get_binary_mc_posterior_for_probability_samples(
                self.model,
                raw_X,
                samples_are_probs=self.samples_are_probs,
                prefer_latent=not self.samples_are_probs,
            )
            samples = post.rsample(torch.Size([self.num_samples]))
            samples = reshape_samples(samples, expanded_X, torch.Size([self.num_samples]))
            probs = to_probability(
                samples,
                apply_sigmoid_if_needed=(not self.samples_are_probs) or self.apply_sigmoid_if_needed,
                eps=self.eps,
                name="posterior.rsample()",
            )
            p = probs.mean(dim=0)
        elif self.mode == "latent_cdf":
            post = _get_binary_mc_posterior_for_probability_samples(
                self.model,
                raw_X,
                samples_are_probs=False,
                prefer_latent=True,
            )
            mu = normalize_mean_shape(post.mean, expanded_X)
            var = normalize_mean_shape(post.variance, expanded_X).clamp_min(self.eps)
            z = (mu - self.threshold) / var.sqrt()
            p = Normal(torch.zeros_like(z), torch.ones_like(z)).cdf(z).clamp(self.eps, 1.0 - self.eps)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        score = aggregate_outputs(
            p,
            output_mode=self.output_mode,
            output_weights=self.output_weights,
            probs_for_all_positive=p,
            eps=self.eps,
        )
        return score

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        expanded_X = shape_X_for_model(self.model, raw_X)
        score = self._pointwise_pof(raw_X, expanded_X)
        score = score - self._pending_penalty_per_point(expanded_X)
        score = apply_pointwise_score_objective(
            self,
            score,
            raw_X=raw_X,
            expanded_X=expanded_X,
            name="qMultiOutputBinaryProbabilityOfFeasibility",
        )
        return reduce_q(score, self.reduction)


class qMultiOutputBinaryExpectedHypervolumeImprovement(qExpectedHypervolumeImprovement):
    """multi-output classification 用 qEHVI acquisition。Pareto hypervolume の期待改善量を評価します。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        全目的は最大化方向に揃えてください。classification / ordinal では probability または utility objective を通して目的空間に変換します。
    """
    pass


class qMultiOutputBinaryNoisyExpectedHypervolumeImprovement(qNoisyExpectedHypervolumeImprovement):
    """multi-output classification 用 qEHVI acquisition。Pareto hypervolume の期待改善量を評価します。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        全目的は最大化方向に揃えてください。classification / ordinal では probability または utility objective を通して目的空間に変換します。
    """
    pass

def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _squeeze_only_output_singleton(value: Tensor, X: Tensor) -> Tensor:
    """
    objective が返す余分な output singleton のみを落とす。

    重要:
        q=1 のとき、[..., q] の q 次元も singleton になる。
        そのため「末尾が 1 なら全部 squeeze」は不可。

    落としてよい例:
        sample x batch x q x 1 -> sample x batch x q

    落としてはいけない例:
        sample x batch x q     -> sample x batch x q
        sample x batch x 1     -> sample x batch x 1
    """
    q = int(X.shape[-2])
    batch_ndim = len(X.shape[:-2])
    min_ndim_with_q = batch_ndim + 1

    # value が [..., q, 1] のときだけ最後の 1 を output dim とみなして落とす。
    if value.ndim >= min_ndim_with_q + 1 and value.shape[-1] == 1:
        if value.shape[-2] == q:
            return value.squeeze(-1)

    return value


def _reduce_sample_and_q_to_tbatch(value: Tensor, X: Tensor) -> Tensor:
    """
    scalarized value を BoTorch acquisition の期待 output shape に整える。

    Expected:
        value = sample_shape x batch_shape x q
        X     = batch_shape x q x d

    Return:
        batch_shape

    q=1 の場合でも q 次元を誤って squeeze しない。
    もし既に [sample_shape] まで潰れてしまった場合は、
    batch_shape と q がともに singleton のときだけ復元的に扱う。
    """
    batch_shape = X.shape[:-2]
    q = int(X.shape[-2])
    batch_prod = _prod(batch_shape)

    value = _squeeze_only_output_singleton(value, X)

    # Normal case: last dim is q.
    if value.ndim >= 1 and value.shape[-1] == q:
        value = value.max(dim=-1).values

    # Recovery case:
    # q=1, batch_prod=1 で value=[sample] のように q/batch が既に落ちている場合。
    elif q == 1 and batch_prod == 1 and value.ndim >= 1:
        # q reduction は実質不要。後段で sample dims を平均する。
        pass

    else:
        raise RuntimeError(
            "Expected scalarized value to have q dimension as the last dimension. "
            f"value.shape={tuple(value.shape)}, q={q}, batch_shape={tuple(batch_shape)}, "
            f"X.shape={tuple(X.shape)}."
        )

    # MC sample dimension reduction.
    # After q reduction, expected shape is sample_shape x batch_shape.
    while value.ndim > len(batch_shape):
        value = value.mean(dim=0)

    if value.shape == batch_shape:
        return value

    if value.numel() == batch_prod:
        return value.reshape(batch_shape)

    if len(batch_shape) == 0 and value.numel() == 1:
        return value.reshape(batch_shape)

    # Additional recovery:
    # q=1 かつ batch_shape=(1,) で value=[sample_shape] だけが残っている場合。
    # 例:
    #   X.shape     = [1, 1, d]
    #   value.shape = [64]
    # これは MC sample 次元のみが残っているので、平均して [1] に戻す。
    if q == 1 and batch_prod == 1 and value.ndim == 1:
        return value.mean().reshape(batch_shape)

    # batch_shape=(1,) なのに value が scalar になっている場合。
    if batch_prod == 1 and value.numel() == 1:
        return value.reshape(batch_shape)

    raise RuntimeError(
        "qMultiOutputBinaryNParEGO produced invalid output shape after scalarization. "
        f"value.shape={tuple(value.shape)}, expected batch_shape={tuple(batch_shape)}, "
        f"X.shape={tuple(X.shape)}."
    )


class _IdentityMCMultiOutputObjective(MCMultiOutputObjective):
    """
    multi-output samples をそのまま返す identity objective。

    BoTorch の MCAcquisitionFunction は multi-output model に対して
    objective=None を許さない。そのため、qNParEGO 内部で独自 scalarization
    する場合でも、super().__init__ には MCMultiOutputObjective を渡す必要がある。

    この objective は samples を変更せず返し、m 次元の scalarization は
    qMultiOutputBinaryNParEGO._scalarize(...) で行う。
    """

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return samples

class qMultiOutputBinaryNParEGO(MCAcquisitionFunction):
    """multi-output classification 用 qNParEGO acquisition。多目的を scalarization して qEI 的に評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        X_baseline: noisy acquisition や NParEGO の baseline 点。通常は既存観測点を渡します。
        ref_point: 多目的 hypervolume 計算で使う参照点。すべての目的が最大化方向に揃っている必要があります。
        weights: NParEGO や weighted objective で使う scalarization weight。
        sampler: posterior samples を生成する BoTorch sampler。省略時は SobolQMCNormalSampler を使います。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        目的数が多い場合や qEHVI / qNEHVI が重い場合の実用的な代替です。
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Tensor,
        *,
        weights: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        rho: float = 0.05,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
    ) -> None:
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        # BoTorch MCAcquisitionFunction は multi-output model に対して
        # objective=None を許さない。
        # qNParEGO では objective 適用後に _scalarize(...) で m 次元を潰すため、
        # objective 未指定時は identity objective を渡す。
        base_objective = objective if objective is not None else _IdentityMCMultiOutputObjective()

        super().__init__(model=model, sampler=sampler, objective=base_objective)
        self.base_objective = base_objective
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)

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

        # ref_point は scalarization の安定化・下限補正に使う。
        self.register_buffer("ref_point", ref_point.to(**tkwargs).reshape(m))

        with torch.no_grad():
            Xb = ensure_q_batch(X_baseline)
            # baseline は deterministic な mean objective から作る。
            # probability_posterior があれば優先し、posterior.mean が latent の場合は
            # apply_sigmoid_if_needed に従って probability に変換する。
            prob_fn = getattr(model, "probability_posterior", None)
            post = prob_fn(X_baseline) if callable(prob_fn) else model.posterior(X_baseline)
            y = normalize_mean_shape(post.mean, Xb)
            y = to_probability(
                y,
                apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
                eps=self.eps,
                name="NParEGO baseline posterior.mean",
            ).reshape(-1, m)

            if X_baseline.ndim == 2:
                X_obj = X_baseline.unsqueeze(0)  # [1, N, d]
            else:
                X_obj = X_baseline

            values = y.unsqueeze(0).unsqueeze(0)  # [1, 1, N, m]
            values = self.base_objective(values, X=X_obj)

            baseline_score = self._scalarize(values)
            # baseline_score may be [..., N] or [..., N, 1]. Remove output singleton only.
            if baseline_score.ndim >= 2 and baseline_score.shape[-1] == 1:
                baseline_score = baseline_score.squeeze(-1)
            self.register_buffer("best_value", baseline_score.max())

    def _scalarize(self, values: Tensor) -> Tensor:
        """
        values を ParEGO scalar に変換する。

        Expected:
            values.shape = sample_shape x batch_shape x q x m

        Return:
            sample_shape x batch_shape x q

        If values is already scalar:
            values.shape = sample_shape x batch_shape x q
            return as-is.

        Notes:
            Classification probability objective では値は概ね [0, 1]。
            ここでは ref_point との差を取り、weighted augmented Chebyshev を使う。
        """
        # If objective returns [..., q, 1] and num_outputs != 1, the last singleton
        # is not a multi-output dimension. Remove only that output singleton.
        if values.ndim >= 2 and values.shape[-1] == 1 and self.num_outputs != 1:
            values = values.squeeze(-1)

        # Already scalarized: [..., q]
        if values.ndim >= 1 and values.shape[-1] != self.num_outputs:
            return values

        if values.ndim < 1 or values.shape[-1] != self.num_outputs:
            raise RuntimeError(
                "Cannot scalarize values. Expected last dim to be num_outputs "
                f"{self.num_outputs}, got values.shape={tuple(values.shape)}."
            )

        w = self.weights.to(device=values.device, dtype=values.dtype)
        ref = self.ref_point.to(device=values.device, dtype=values.dtype)

        # Improvement-like shifted values. Since objectives are maximized,
        # values below ref_point contribute less.
        shifted = values - ref

        weighted = shifted * w
        cheb = weighted.min(dim=-1).values
        aug = self.rho * weighted.sum(dim=-1)

        return cheb + aug

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)

        post = _get_binary_mc_posterior_for_probability_samples(
            self.model,
            Xq,
            samples_are_probs=self.samples_are_probs,
            prefer_latent=(not self.samples_are_probs) and self.apply_sigmoid_if_needed,
        )
        samples = self.get_posterior_samples(post)

        values = to_probability(
            samples,
            apply_sigmoid_if_needed=(not self.samples_are_probs) or self.apply_sigmoid_if_needed,
            eps=self.eps,
            name="NParEGO posterior samples",
        )

        values = self.base_objective(values, X=Xq)

        scalarized = self._scalarize(values)
        improvement = (scalarized - self.best_value.to(scalarized)).clamp_min(0.0)

        return _reduce_sample_and_q_to_tbatch(improvement, Xq)

class _qMultiOutputBinaryFeasibilityWeightedAcquisition(AcquisitionFunction):
    """任意の multi-objective acquisition を multi-output PoF で重み付けする wrapper。"""

    def __init__(
        self,
        objective_acqf: AcquisitionFunction,
        feasibility_acqf: qMultiOutputBinaryProbabilityOfFeasibility,
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
        X_pending = _detach_optional_tensor(X_pending)
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
    "qMultiOutputBinaryProbabilityOfFeasibility",
    "qMultiOutputBinaryExpectedHypervolumeImprovement",
    "qMultiOutputBinaryNoisyExpectedHypervolumeImprovement",
    "qMultiOutputBinaryNParEGO",
]
