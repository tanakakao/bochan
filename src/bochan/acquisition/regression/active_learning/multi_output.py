
from __future__ import annotations

import math
from typing import Any, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform


QReduceType = Literal["mean", "sum", "max", "min"]
OutputAggregation = Literal[
    "mean",
    "sum",
    "max",
    "min",
    "weighted_sum",
    "weighted_mean",
]




def _reduce(t: Tensor, dim: int, mode: str) -> Tensor:
    if mode == "mean":
        return t.mean(dim=dim)
    if mode == "sum":
        return t.sum(dim=dim)
    if mode == "max":
        return t.max(dim=dim).values
    if mode == "min":
        return t.min(dim=dim).values
    raise ValueError(f"Unknown reduction mode: {mode}")


class _MultiOutputRegressionActiveLearningBase(AcquisitionFunction):
    """multi-output regression active-learning acquisition base.

    The family is aligned with classification / ordinal active learning:

    - PredictiveEntropy
    - BALD / mutual-information proxy
    - PosteriorVariance
    - MarginUncertainty
    - IntegratedPosteriorVarianceProxy

    This class intentionally does not depend on a shared cross-task utility module,
    so regression / classification / ordinal acquisitions can be maintained independently.
    """

    def __init__(
        self,
        model,
        *,
        output_aggregation: OutputAggregation = "weighted_mean",
        output_weights: Optional[Tensor] = None,
        normalize_output_weights: bool = True,
        q_reduction: QReduceType = "mean",
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_penalty: float = 0.0,
        hard_duplicate_tol: float = 1e-8,
        eps: float = 1e-12,
        objective: Optional[Any] = None,
        n_w: Optional[int] = None,
    ) -> None:
        super().__init__(model=model)
        self.output_aggregation = output_aggregation
        self.normalize_output_weights = bool(normalize_output_weights)
        if output_weights is not None:
            if output_weights.ndim != 1:
                raise ValueError("output_weights must have shape [m].")
            self.register_buffer("output_weights", output_weights.detach().clone())
        else:
            self.output_weights = None

        self.q_reduction = q_reduction
        self.eps = float(eps)
        self.objective = objective
        if n_w is None and objective is not None:
            n_w = getattr(objective, "n_w", None)
        self.n_w = int(n_w) if n_w is not None else None

        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)
        self.hard_duplicate_tol = float(hard_duplicate_tol)
        self.X_pending: Optional[Tensor] = None
        self.X_observed: Optional[Tensor] = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        """pending points を raw input space の値として保持する。"""
        self.X_pending = self._coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        """observed points を raw input space の値として保持する。"""
        self.X_observed = self._coerce_reference_to_tensor(X_observed)

    def _ensure_q_batch_for_distance(self, X: Tensor) -> Tensor:
        """距離計算用に X を `batch_shape x q x d` へ正規化する。"""
        if X.ndim == 2:
            return X.unsqueeze(-2)
        return X

    def _coerce_reference_to_tensor(
        self,
        ref,
        *,
        like: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """X_pending / X_observed を Tensor または None に正規化する。"""
        if ref is None:
            return None

        if torch.is_tensor(ref):
            out = ref

        elif isinstance(ref, (list, tuple)):
            tensors = []
            for item in ref:
                if item is None:
                    continue
                t = self._coerce_reference_to_tensor(item, like=like)
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
                    out = torch.cat(
                        [t.reshape(-1, t.shape[-1]) for t in tensors],
                        dim=-2,
                    )
        else:
            raise TypeError(
                "Reference points must be None, Tensor, list, or tuple. "
                f"Got {type(ref)}."
            )

        if like is not None:
            out = out.to(device=like.device, dtype=like.dtype)

        return out

    def _apply_input_transform_for_distance(self, X: Tensor) -> Tensor:
        """
        距離計算用に candidate / reference を同じ feature space へ写す。

        `X_pending` / `X_observed` は raw input space で保持する。一方、
        candidate 側の score は model の `input_transform` や submodel の
        `input_transform` により transformed / InputPerturbation 展開後の
        q_like と整合している場合がある。

        そのため、距離ペナルティは raw X ではなく、この関数を通した
        transformed space で計算する。
        """
        X = self._ensure_q_batch_for_distance(X)

        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return self._ensure_q_batch_for_distance(Xt)

        models = getattr(self.model, "models", None)
        if models is not None and len(models) > 0:
            it = getattr(models[0], "input_transform", None)
            if it is not None:
                Xt = it(X)
                if isinstance(Xt, tuple):
                    Xt = Xt[0]
                return self._ensure_q_batch_for_distance(Xt)

        return X

    def _reference_to_distance_space(
        self,
        ref,
        *,
        like: Tensor,
    ) -> Optional[Tensor]:
        """raw-space reference を candidate と同じ距離計算空間へ写す。"""
        ref = self._coerce_reference_to_tensor(ref, like=like)
        if ref is None or ref.numel() == 0:
            return None

        ref_t = self._apply_input_transform_for_distance(ref)
        ref_t = self._ensure_q_batch_for_distance(ref_t)
        return ref_t.to(device=like.device, dtype=like.dtype)

    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        if X.ndim == 2:
            X = X.unsqueeze(-2)
        return X

    def _aggregate_outputs(self, t: Tensor) -> Tensor:
        """Aggregate output dimension m of a tensor shaped [..., q, m]."""
        if t.ndim < 3:
            return t

        weights = self.output_weights
        if weights is not None:
            if t.shape[-1] != weights.shape[0]:
                raise ValueError(
                    f"Mismatch between output dim {t.shape[-1]} and "
                    f"output_weights {weights.shape[0]}."
                )
            w = weights.to(device=t.device, dtype=t.dtype)
            if self.normalize_output_weights:
                w = w / w.sum().clamp_min(self.eps)
        else:
            w = None

        if self.output_aggregation == "weighted_sum":
            if w is None:
                raise ValueError("output_aggregation='weighted_sum' requires output_weights.")
            return (t * w).sum(dim=-1)
        if self.output_aggregation == "weighted_mean":
            if w is None:
                return t.mean(dim=-1)
            return (t * w).sum(dim=-1)
        if self.output_aggregation == "mean":
            return t.mean(dim=-1)
        if self.output_aggregation == "sum":
            return t.sum(dim=-1)
        if self.output_aggregation == "max":
            return t.max(dim=-1).values
        if self.output_aggregation == "min":
            return t.min(dim=-1).values
        raise ValueError(f"Unknown output_aggregation={self.output_aggregation!r}.")

    def _reduce_q(self, t: Tensor) -> Tensor:
        return _reduce(t, dim=-1, mode=self.q_reduction)

    def _apply_risk_objective(self, score_per_point: Tensor, q: int, *, name: str) -> Tensor:
        """Optionally reduce InputPerturbation-expanded q*n_w to q."""
        if self.n_w is None:
            if self.objective is not None:
                try:
                    out = self.objective(score_per_point, X=None)
                except TypeError:
                    out = self.objective(score_per_point)
                if torch.is_tensor(out):
                    return out
                raise TypeError(f"{name}: objective must return a Tensor, got {type(out)}.")
            return score_per_point

        qnw = q * self.n_w
        if score_per_point.shape[-1] != qnw:
            return score_per_point

        if self.objective is None:
            return score_per_point.reshape(*score_per_point.shape[:-1], q, self.n_w).mean(dim=-1)

        # Use score as deterministic pseudo-samples for risk objectives.
        pseudo_samples = score_per_point.unsqueeze(0).unsqueeze(-1)
        out = self.objective(pseudo_samples)
        if out.ndim >= 1 and out.shape[0] == 1:
            out = out.squeeze(0)
        if out.shape[-1] != q:
            raise RuntimeError(
                f"{name}: expected objective output last dim q={q}, got {tuple(out.shape)}."
            )
        return out

    def _aggregate_point_scores(self, score_per_point: Tensor, X: Tensor, *, name: str) -> Tensor:
        q = X.shape[-2]
        score = self._apply_risk_objective(score_per_point, q=q, name=name)
        return self._reduce_q(score)

    def _posterior_mean_var(self, X: Tensor, *, observation_noise: bool = False) -> tuple[Tensor, Tensor]:
        post = self.model.posterior(X, observation_noise=observation_noise)
        mean = post.mean
        var = post.variance.clamp_min(self.eps)
        if mean.ndim == X.ndim and mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        if var.ndim == X.ndim and var.shape[-1] == 1:
            var = var.squeeze(-1)
        mean = self._aggregate_outputs(mean)
        var = self._aggregate_outputs(var)
        return mean, var

    def _same_batch_penalty(self, X: Tensor) -> Tensor:
        if self.same_batch_penalty_weight <= 0 or X.shape[-2] <= 1:
            return X.new_zeros(X.shape[:-2])
        d2 = (X.unsqueeze(-2) - X.unsqueeze(-3)).pow(2).sum(dim=-1)
        q = X.shape[-2]
        eye = torch.eye(q, dtype=torch.bool, device=X.device)
        while eye.ndim < d2.ndim:
            eye = eye.unsqueeze(0)
        valid = ~eye
        soft = torch.exp(-self.same_batch_penalty_beta * d2)
        soft = torch.where(valid, soft, torch.zeros_like(soft)).sum(dim=(-2, -1))
        if self.hard_duplicate_penalty > 0:
            dup = (d2 <= self.hard_duplicate_tol).to(X.dtype)
            dup = torch.where(valid, dup, torch.zeros_like(dup)).sum(dim=(-2, -1))
            hard = self.hard_duplicate_penalty * dup
        else:
            hard = torch.zeros_like(soft)
        return self.same_batch_penalty_weight * soft + hard

    def _ref_penalty(self, X: Tensor, ref: Optional[Tensor], weight: float, beta: float) -> Tensor:
        """
        距離計算空間を candidate / reference で揃えた reference penalty。

        Args:
            X:
                すでに `_apply_input_transform_for_distance(...)` を通した
                candidate。shape は `batch_shape x q_like x d`。
            ref:
                raw input space の reference points。
            weight:
                penalty weight。
            beta:
                exponential penalty の距離減衰係数。
        """
        if weight <= 0:
            return X.new_zeros(X.shape[:-2])

        ref_t = self._reference_to_distance_space(ref, like=X)
        if ref_t is None or ref_t.numel() == 0:
            return X.new_zeros(X.shape[:-2])

        ref2d = ref_t.reshape(-1, ref_t.shape[-1])
        if ref2d.shape[-1] != X.shape[-1]:
            raise RuntimeError(
                "Reference feature dimension mismatch after transform: "
                f"X.shape={tuple(X.shape)}, ref_transformed.shape={tuple(ref_t.shape)}."
            )

        d2 = (
            X.unsqueeze(-2)
            - ref2d.view(*([1] * (X.ndim - 2)), ref2d.shape[0], ref2d.shape[1])
        ).pow(2).sum(dim=-1)
        per_point = torch.exp(-beta * d2).sum(dim=-1)
        return weight * self._reduce_q(per_point)

    def _total_penalty(self, X: Tensor) -> Tensor:
        """
        same-batch / pending / observed penalty を transformed space で計算する。
        """
        Xt = self._apply_input_transform_for_distance(X)
        return (
            self._same_batch_penalty(Xt)
            + self._ref_penalty(Xt, self.X_pending, self.pending_penalty_weight, self.pending_penalty_beta)
            + self._ref_penalty(Xt, self.X_observed, self.observed_penalty_weight, self.observed_penalty_beta)
        )

class qMultiOutputRegressionPredictiveEntropy(_MultiOutputRegressionActiveLearningBase):
    """multi-output regression 用 predictive entropy acquisition。予測分布の曖昧さが大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        予測が曖昧な点を探索したい場合の基本的な active learning acquisition です。
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        _, var = self._posterior_mean_var(X, observation_noise=True)
        entropy = 0.5 * torch.log(2.0 * math.pi * math.e * var.clamp_min(self.eps))
        score = self._aggregate_point_scores(entropy, X, name="qMultiOutputRegressionPredictiveEntropy")
        return score - self._total_penalty(X)


class qMultiOutputRegressionBALDProxy(_MultiOutputRegressionActiveLearningBase):
    """multi-output regression 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        BALD は predictive entropy から条件付き entropy を引いた情報利得として解釈できます。
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        _, latent_var = self._posterior_mean_var(X, observation_noise=False)
        _, total_var = self._posterior_mean_var(X, observation_noise=True)
        noise_var = (total_var - latent_var).clamp_min(self.eps)
        mi = 0.5 * torch.log(total_var.clamp_min(self.eps) / noise_var)
        score = self._aggregate_point_scores(mi, X, name="qMultiOutputRegressionBALDProxy")
        return score - self._total_penalty(X)


class qMultiOutputRegressionPosteriorVariance(_MultiOutputRegressionActiveLearningBase):
    """multi-output regression 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        _, var = self._posterior_mean_var(X, observation_noise=False)
        score = self._aggregate_point_scores(var, X, name="qMultiOutputRegressionPosteriorVariance")
        return score - self._total_penalty(X)


class qMultiOutputRegressionMarginUncertainty(_MultiOutputRegressionActiveLearningBase):
    """multi-output regression 用 margin uncertainty acquisition。決定境界または class 境界に近い点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        target: straddle / margin 系で近づけたい目標値。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model,
        target: float | Tensor = 0.0,
        beta: float = 1.96,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if torch.is_tensor(target):
            self.register_buffer("target", target.detach().clone())
        else:
            self.target = torch.tensor(float(target))
        self.beta = float(beta)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        post = self.model.posterior(X, observation_noise=False)
        mean = post.mean
        std = post.variance.clamp_min(self.eps).sqrt()
        if mean.ndim == X.ndim and mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
            std = std.squeeze(-1)
            score_per_output = self.beta * std - (mean - self.target.to(mean)).abs()
        else:
            target = self.target.to(mean)
            if target.ndim == 0:
                score_per_output = self.beta * std - (mean - target).abs()
            else:
                if target.ndim != 1 or target.shape[0] != mean.shape[-1]:
                    raise ValueError(
                        f"target must be scalar or shape [m], got {tuple(target.shape)}."
                    )
                score_per_output = self.beta * std - (mean - target.view(*([1] * (mean.ndim - 1)), -1)).abs()
        point_score = self._aggregate_outputs(score_per_output)
        score = self._aggregate_point_scores(point_score, X, name="qMultiOutputRegressionMarginUncertainty")
        return score - self._total_penalty(X)


class qMultiOutputRegressionIntegratedPosteriorVarianceProxy(_MultiOutputRegressionActiveLearningBase):
    """multi-output regression 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        X_ref: 参照点集合。IPV / NIPV proxy などで全体の不確実性を評価するために使います。
        kernel_lengthscale: 参照点被覆 proxy で使う RBF kernel の lengthscale。
        normalize_weights: 参照点重みや出力重みを正規化するかどうか。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model,
        X_ref: Tensor,
        kernel_lengthscale: float = 0.2,
        normalize_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("X_ref", X_ref)
        self.kernel_lengthscale = float(kernel_lengthscale)
        self.normalize_weights = bool(normalize_weights)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        X_ref = self.X_ref.to(device=X.device, dtype=X.dtype)
        _, ref_var = self._posterior_mean_var(X_ref, observation_noise=False)
        ref_score = ref_var
        if ref_score.ndim >= 2:
            ref_score = self._aggregate_outputs(ref_score)
        if ref_score.ndim > 1:
            ref_score = ref_score.reshape(-1)
        X_ref_flat = X_ref.reshape(-1, X_ref.shape[-1])
        d2 = (X.unsqueeze(-2) - X_ref_flat.view(*([1] * (X.ndim - 2)), X_ref_flat.shape[0], X_ref_flat.shape[1])).pow(2).sum(dim=-1)
        weights = torch.exp(-0.5 * d2 / (self.kernel_lengthscale ** 2 + self.eps))
        if self.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        covered = (weights * ref_score.view(*([1] * (weights.ndim - 1)), -1)).sum(dim=-1)
        score = self._aggregate_point_scores(covered, X, name="qMultiOutputRegressionIntegratedPosteriorVarianceProxy")
        return score - self._total_penalty(X)


class qMultiOutputRegressionNegIntegratedPosteriorVariance(AcquisitionFunction):
    """multi-output regression 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        mc_points: 不確実性を積分するための Monte Carlo / 参照点集合。
        sampler: posterior samples を生成する BoTorch sampler。省略時は SobolQMCNormalSampler を使います。
        X_pending: 評価中で、まだ結果が返っていない候補点。重複候補の抑制に使います。
        output_reduction: multi-output の出力方向の集約方法。
        output_weights: multi-output score を集約するときの出力方向の重み。
        integrate_reduction: この acquisition / objective の動作を制御するパラメータ。
        eps: 数値安定化用の微小値。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model,
        mc_points: Tensor,
        sampler=None,
        X_pending: Tensor | None = None,
        output_reduction: str = "sum",
        output_weights: Tensor | None = None,
        integrate_reduction: str = "mean",
        eps: float = 1e-12,
    ) -> None:
        super().__init__(model=model)
        self.register_buffer("mc_points", mc_points)
        self.sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([1]))
        self.output_reduction = output_reduction
        self.output_weights = output_weights
        self.integrate_reduction = integrate_reduction
        self.eps = float(eps)
        self.set_X_pending(X_pending)

    def _coerce_pending_to_tensor(
        self,
        X_pending,
        *,
        ref: Tensor | None = None,
    ) -> Tensor | None:
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
        return out

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        """pending points を raw input space の値として保持する。"""
        self.X_pending = self._coerce_pending_to_tensor(X_pending)

    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        return X.unsqueeze(-2) if X.ndim == 2 else X

    def _expand_pending(self, X: Tensor) -> Tensor:
        """
        posterior 入力用に raw-space pending points を X に連結する。

        これは距離ペナルティではなく posterior 評価用の concatenation なので、
        `X_pending` は raw input space のまま扱う。
        """
        Xp = self._coerce_pending_to_tensor(getattr(self, "X_pending", None), ref=X)
        if Xp is None or Xp.numel() == 0:
            return X

        batch_shape = X.shape[:-2]
        if Xp.ndim == 2:
            m, d = Xp.shape
            Xp = Xp.view(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        elif Xp.ndim >= 3:
            m, d = Xp.shape[-2], Xp.shape[-1]
            Xp = Xp.reshape(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        else:
            raise ValueError(f"Unexpected X_pending shape: {tuple(Xp.shape)}")
        return torch.cat([Xp, X], dim=-2)

    def _normalize_variance_shape(self, var: Tensor) -> Tensor:
        n_mc = self.mc_points.shape[-2]
        if var.shape[-1] == n_mc:
            return var.unsqueeze(-1)
        if var.ndim >= 2 and var.shape[-2] == n_mc:
            return var
        raise RuntimeError(
            f"Unsupported posterior.variance shape {tuple(var.shape)} for "
            f"mc_points shape {tuple(self.mc_points.shape)}."
        )

    def _reduce_outputs(self, var: Tensor) -> Tensor:
        if self.output_reduction == "sum":
            return var.sum(dim=-1)
        if self.output_reduction == "mean":
            return var.mean(dim=-1)
        if self.output_reduction == "max":
            return var.max(dim=-1).values
        if self.output_reduction == "weighted_sum":
            if self.output_weights is None:
                raise ValueError("output_weights must be provided for weighted_sum.")
            w = self.output_weights.to(device=var.device, dtype=var.dtype)
            return (var * w.view(*([1] * (var.ndim - 1)), -1)).sum(dim=-1)
        raise ValueError(f"Unknown output_reduction: {self.output_reduction}")

    def _integrate_mc_points(self, score_per_mc: Tensor) -> Tensor:
        if self.integrate_reduction == "mean":
            return score_per_mc.mean(dim=-1)
        if self.integrate_reduction == "sum":
            return score_per_mc.sum(dim=-1)
        raise ValueError(f"Unknown integrate_reduction: {self.integrate_reduction}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        Xt = self._expand_pending(X)
        self.model.eval()
        fantasy_model = self.model.fantasize(X=Xt, sampler=self.sampler)
        posterior = fantasy_model.posterior(self.mc_points)
        var = self._normalize_variance_shape(posterior.variance.clamp_min(self.eps))
        integrated_var = self._integrate_mc_points(self._reduce_outputs(var))
        fantasy_ndim = len(self.sampler.sample_shape)
        if fantasy_ndim > 0:
            integrated_var = integrated_var.mean(dim=tuple(range(fantasy_ndim)))
        out = -integrated_var
        expected = X.shape[:-2]
        if out.shape != expected:
            raise RuntimeError(f"Output shape mismatch: expected {tuple(expected)}, got {tuple(out.shape)}")
        return out



# DeepGP-oriented previous names

__all__ = [
    "qMultiOutputRegressionPredictiveEntropy",
    "qMultiOutputRegressionBALDProxy",
    "qMultiOutputRegressionPosteriorVariance",
    "qMultiOutputRegressionMarginUncertainty",
    "qMultiOutputRegressionIntegratedPosteriorVarianceProxy",
    "qMultiOutputRegressionNegIntegratedPosteriorVariance",
]
