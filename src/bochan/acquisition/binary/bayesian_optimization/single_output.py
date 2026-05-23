from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models import ModelListGP
from botorch.models.gpytorch import ModelListGPyTorchModel
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform

from bochan.acquisition.binary.base import ReductionType, _BinaryClassificationAcqBase

from ._utils import (
    apply_score_objective,
    binary_entropy,
    ensure_q_batch,
    normalize_binary_mean_shape,
    reshape_binary_samples,
    to_probability,
)


PoFMode = Literal["mc_sigmoid", "latent_cdf"]
QFeasMode = Literal["prod", "mean", "min", "max"]
CombineMode = Literal["product", "log_product", "penalty"]
BaseTransformMode = Literal["identity", "clamp_nonnegative", "softplus"]



def _finalize_binary_acq_output_to_batch(
    value: Tensor,
    X: Tensor,
    *,
    name: str,
) -> Tensor:
    """binary BO acquisition output を BoTorch の t-batch shape に揃える。

    q=1 / batch=1 で MC 平均後に scalar になると、
    gen_batch_initial_conditions の torch.cat で落ちるため、必ず batch shape を
    保持して返す。
    """
    Xq = ensure_q_batch(X)
    target = tuple(Xq.shape[:-2])
    out = value

    if out.shape == target:
        return out

    if len(target) == 0:
        if out.ndim == 0:
            return out
        return out.mean()

    if out.ndim == 0:
        return out.expand(*target)

    while out.ndim > len(target):
        out = out.mean(dim=0)
        if out.shape == target:
            return out

    if out.shape == target:
        return out

    if out.numel() == int(torch.tensor(target).prod().item()):
        return out.reshape(target)

    if out.ndim == 1 and len(target) == 1:
        if out.shape[0] == target[0]:
            return out
        return out.mean().expand(*target)

    raise RuntimeError(
        f"{name}: could not align acquisition output to t-batch shape. "
        f"value.shape={tuple(value.shape)}, target={target}, X.shape={tuple(Xq.shape)}."
    )


class qBinaryProbabilityOfFeasibility(_BinaryClassificationAcqBase):
    """binary classification 用 probability of feasibility acquisition。実現可能確率を最大化します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        num_samples: classification probability や BALD などを MC 近似する sample 数。
        threshold: binary classification や level-set で使う境界値。
        mode: 獲得関数の計算モード。例: `mc_sigmoid` または `latent_cdf`。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        pending_penalty_beta: X_pending penalty の距離減衰率。
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
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        try:
            super().__init__(
                model=model,
                reduction=reduction,
                pending_penalty_weight=pending_penalty_weight,
                pending_penalty_beta=pending_penalty_beta,
                eps=eps,
                objective=objective,
            )
        except TypeError:
            super().__init__(
                model=model,
                reduction=reduction,
                pending_penalty_weight=pending_penalty_weight,
                pending_penalty_beta=pending_penalty_beta,
                eps=eps,
            )
        self.num_samples = int(num_samples)
        self.threshold = float(threshold)
        self.mode = mode
        self.objective = objective

    def _mc_sigmoid_prob(self, latent_dist, orig: torch.Size) -> Tensor:
        f_samples = latent_dist.rsample(torch.Size([self.num_samples]))
        expected = self.num_samples * math.prod(orig)
        if f_samples.numel() != expected:
            raise RuntimeError(
                f"Unexpected sample shape: got {tuple(f_samples.shape)}, "
                f"numel={f_samples.numel()}, expected={expected}"
            )
        f_samples = f_samples.reshape(self.num_samples, *orig)
        return torch.sigmoid(f_samples).clamp(self.eps, 1.0 - self.eps).mean(dim=0)

    def _latent_cdf_prob(self, latent_dist, orig: torch.Size) -> Tensor:
        mu = self._reshape_pointwise_tensor(latent_dist.mean, orig)
        var = self._reshape_pointwise_tensor(latent_dist.variance, orig).clamp_min(self.eps)
        sigma = var.sqrt()
        z = (mu - self.threshold) / sigma
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return normal.cdf(z).clamp(self.eps, 1.0 - self.eps)

    def _pointwise_pof_from_latent_dist(self, latent_dist, orig: torch.Size) -> Tensor:
        if self.mode == "mc_sigmoid":
            return self._mc_sigmoid_prob(latent_dist, orig)
        if self.mode == "latent_cdf":
            return self._latent_cdf_prob(latent_dist, orig)
        raise ValueError(f"Unknown mode: {self.mode}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.ndim > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        latent_dist, orig, Xt = self._get_latent_dist_and_orig(X)
        score = self._pointwise_pof_from_latent_dist(latent_dist, orig)

        penalty = self._pending_penalty_per_point(Xt)
        if penalty.shape == score.shape:
            score = score - penalty
        elif penalty.numel() == score.numel():
            score = score - penalty.reshape_as(score)
        elif self.pending_penalty_weight > 0:
            raise RuntimeError(
                f"Pending penalty shape mismatch: score={tuple(score.shape)}, penalty={tuple(penalty.shape)}"
            )

        score = apply_score_objective(self, score, X=X, attr_name="objective", name="PoF")
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "PoF")
        return out


class _BinaryProbabilityBOBase(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        *,
        sampler: Optional[SobolQMCNormalSampler] = None,
        apply_sigmoid_if_needed: bool = False,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        # MCAcquisitionFunction は objective=None でも single-output model では
        # default Identity objective を self.objective として持つことがある。
        # そのため「ユーザーが明示的に渡した objective」だけを別管理する。
        super().__init__(model=model, sampler=sampler, objective=None, **kwargs)

        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)
        self.score_objective = objective

    # def _posterior_samples_as_prob(self, X: Tensor) -> Tensor:
    #     X = ensure_q_batch(X)
    #     post = self.model.posterior(X)
    #     samples = self.get_posterior_samples(post)
    #     samples = reshape_binary_samples(samples, X)
    #     return to_probability(
    #         samples,
    #         apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
    #         eps=self.eps,
    #         name="posterior samples",
    #     )
    @staticmethod
    def _squeeze_binary_output_dim_if_present(probs: Tensor, X: Tensor) -> Tensor:
        """
        binary output dim=1 が本当に存在するときだけ squeeze する。

        OK:
            X.shape     = batch_shape x q x d
            probs.shape = sample_shape x batch_shape x q x 1
            -> sample_shape x batch_shape x q

        NG:
            X.shape     = batch_shape x 1 x d
            probs.shape = sample_shape x batch_shape x 1
            -> これは最後の 1 が q=1 なので squeeze しない
        """
        if probs.ndim > X.ndim and probs.shape[-1] == 1:
            return probs.squeeze(-1)
        return probs

    def _posterior_samples_as_prob(self, X: Tensor) -> Tensor:
        """posterior samples を binary probability samples に変換する。

        方針:
            - apply_sigmoid_if_needed=True かつ model.latent_posterior がある場合は、
              latent posterior から sample して sigmoid で probability に変換する。
            - これは BinaryClassificationGPModel の設計
              posterior() = probability posterior
              latent_posterior() = latent f posterior
              に合わせた挙動。
            - model.posterior(X).rsample() は SimpleBernoulliPosterior の実装次第で
              [0, 1] 外の連続値を返すことがあるため、latent -> sigmoid を優先する。
        """
        X = ensure_q_batch(X)

        if self.apply_sigmoid_if_needed and hasattr(self.model, "latent_posterior"):
            post = self.model.latent_posterior(X)
            samples = self.get_posterior_samples(post)
            probs = torch.sigmoid(samples).clamp(self.eps, 1.0 - self.eps)
        else:
            post = self.model.posterior(X)
            samples = self.get_posterior_samples(post)
            probs = to_probability(
                samples,
                apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
                eps=self.eps,
                name="posterior samples",
            )

        # ユーザーが明示的に渡した objective のみ適用する。
        # BoTorch 親クラスが持つ default Identity objective は使わない。
        #
        # objective は [mc, batch, q, 1] / [mc, batch, q] のどちらにも
        # 対応している可能性があるため、unsafe squeeze より前に適用する。
        if getattr(self, "score_objective", None) is not None:
            probs = self.score_objective(probs, X=X)

        # binary output dim=1 だけを落とす。
        # q=1 の q 次元は落とさない。
        probs = self._squeeze_binary_output_dim_if_present(probs, X)

        # 最終的に binary acquisition 用 shape に整える
        probs = reshape_binary_samples(probs, X)

        return probs

    def _posterior_mean_std_prob(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """posterior samples を probability 空間に変換して mean/std を返す。

        qBinaryUpperConfidenceBound 用の helper。
        classification posterior は latent / probability のどちらを返す実装もあるため、
        EI/PI と同じ _posterior_samples_as_prob() を経由して probability 空間に揃える。
        """
        probs = self._posterior_samples_as_prob(X)

        if probs.ndim < 2:
            raise RuntimeError(
                "_posterior_samples_as_prob must return sample_shape x ... x q. "
                f"Got probs.shape={tuple(probs.shape)}."
            )

        mean = probs.mean(dim=0)
        std = probs.std(dim=0, unbiased=False).clamp_min(self.eps)

        return mean, std


class qBinaryExpectedImprovement(_BinaryProbabilityBOBase):
    """classification 用 expected improvement acquisition。現在の best_f からの改善量を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        best_f: 既存観測点または baseline から計算した現在の最良値。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
            latent posterior を持つ分類モデルでは `apply_sigmoid_if_needed=True` を推奨します。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(self, model: Model, best_f: float | Tensor, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        probs = self._posterior_samples_as_prob(X)
        best_q = probs.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        value = (best_q - best_f).clamp_min(0.0).mean(dim=0)
        return _finalize_binary_acq_output_to_batch(
            value,
            X,
            name="qBinaryExpectedImprovement",
        )


class qBinaryProbabilityOfImprovement(_BinaryProbabilityBOBase):
    """classification 用 probability of improvement acquisition。best_f を上回る確率を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        best_f: 既存観測点または baseline から計算した現在の最良値。
        tau: soft PI や境界近傍重み付けに使う温度・幅パラメータ。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(self, model: Model, best_f: float | Tensor, tau: float = 1e-3, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        probs = self._posterior_samples_as_prob(X)
        best_q = probs.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        tau = self.tau.to(best_q).clamp_min(1e-9)
        value = torch.sigmoid((best_q - best_f) / tau).mean(dim=0)
        return _finalize_binary_acq_output_to_batch(
            value,
            X,
            name="qBinaryProbabilityOfImprovement",
        )


class qBinaryUpperConfidenceBound(_BinaryProbabilityBOBase):
    """classification 用 upper confidence bound acquisition。平均と不確実性を組み合わせて探索します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
            `apply_sigmoid_if_needed` が未指定の場合、このクラスでは True を既定値にします。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(self, model: Model, beta: float | Tensor = 2.0, **kwargs) -> None:
        # UCB は probability 空間の mean/std を使う acquisition。
        # BinaryClassificationGPModel の posterior samples は latent のことが多いため、
        # ユーザーが明示しない場合は sigmoid 変換を有効にする。
        # 既に probability posterior を返すモデルでは、明示的に
        # apply_sigmoid_if_needed=False を渡せば従来通りにできる。
        kwargs.setdefault("apply_sigmoid_if_needed", True)
        super().__init__(model=model, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, std = self._posterior_mean_std_prob(X)
        beta = self.beta.to(mean)
        score = mean + beta.sqrt() * std
        value = score.max(dim=-1).values
        return _finalize_binary_acq_output_to_batch(
            value,
            X,
            name="qBinaryUpperConfidenceBound",
        )


class _qBinaryFeasibilityWeightedAcquisition(AcquisitionFunction):
    """
    Feasibility-weighted wrapper for arbitrary objective-side acquisition.

    Examples:
        EI * PoF, NEI * PoF, softplus(UCB) * PoF, logEI + log(PoF)
    """

    def __init__(
        self,
        objective_acqf: AcquisitionFunction,
        feasibility_model,
        num_pof_samples: int = 32,
        threshold: float = 0.0,
        pof_mode: PoFMode = "mc_sigmoid",
        combine_mode: CombineMode = "product",
        q_feas_mode: QFeasMode = "prod",
        feasibility_power: float = 1.0,
        base_transform: BaseTransformMode = "identity",
        penalty_weight: float = 1.0,
        eps: float = 1e-8,
        feasibility_objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        if isinstance(feasibility_model, (ModelListGP, ModelListGPyTorchModel)):
            feasibility_model = feasibility_model.models[0]

        super().__init__(getattr(objective_acqf, "model", feasibility_model))
        self.objective_acqf = objective_acqf
        self.feasibility_model = feasibility_model
        self.num_pof_samples = int(num_pof_samples)
        self.threshold = float(threshold)
        self.pof_mode = pof_mode
        self.combine_mode = combine_mode
        self.q_feas_mode = q_feas_mode
        self.feasibility_power = float(feasibility_power)
        self.base_transform = base_transform
        self.penalty_weight = float(penalty_weight)
        self.eps = float(eps)
        self.feasibility_objective = feasibility_objective
        self.set_X_pending(None)

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = X_pending
        if hasattr(self.objective_acqf, "set_X_pending"):
            self.objective_acqf.set_X_pending(X_pending)

    def _pof_acqf(self) -> qBinaryProbabilityOfFeasibility:
        acqf = qBinaryProbabilityOfFeasibility(
            model=self.feasibility_model,
            num_samples=self.num_pof_samples,
            threshold=self.threshold,
            mode=self.pof_mode,
            reduction="mean",
            eps=self.eps,
            objective=self.feasibility_objective,
        )
        if getattr(self, "X_pending", None) is not None:
            acqf.set_X_pending(self.X_pending)
        return acqf

    def _transform_objective(self, base_val: Tensor) -> Tensor:
        if self.base_transform == "identity":
            return base_val
        if self.base_transform == "clamp_nonnegative":
            return base_val.clamp_min(0.0)
        if self.base_transform == "softplus":
            return F.softplus(base_val)
        raise ValueError(f"Unknown base_transform: {self.base_transform}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        base_val = self.objective_acqf(X)

        pof_acqf = self._pof_acqf()
        latent_dist, orig, _ = pof_acqf._get_latent_dist_and_orig(X)
        pof_point = pof_acqf._pointwise_pof_from_latent_dist(latent_dist, orig)
        pof_point = apply_score_objective(
            self,
            pof_point,
            X=X,
            attr_name="feasibility_objective",
            name="FeasibilityWeightedAcquisitionBinary",
        )
        if self.q_feas_mode == "prod":
            q_pof = pof_point.prod(dim=-1)
        elif self.q_feas_mode == "mean":
            q_pof = pof_point.mean(dim=-1)
        elif self.q_feas_mode == "min":
            q_pof = pof_point.min(dim=-1).values
        elif self.q_feas_mode == "max":
            q_pof = pof_point.max(dim=-1).values
        else:
            raise ValueError(f"Unknown q_feas_mode: {self.q_feas_mode}")

        q_pof = q_pof.clamp(self.eps, 1.0 - self.eps)

        if self.combine_mode == "product":
            value = self._transform_objective(base_val) * q_pof.pow(self.feasibility_power)
            return _finalize_binary_acq_output_to_batch(
                value,
                X,
                name="qBinaryFeasibilityWeightedAcquisition",
            )
        if self.combine_mode == "log_product":
            value = base_val + self.feasibility_power * torch.log(q_pof)
            return _finalize_binary_acq_output_to_batch(
                value,
                X,
                name="qBinaryFeasibilityWeightedAcquisition",
            )
        if self.combine_mode == "penalty":
            value = self._transform_objective(base_val) - self.penalty_weight * (1.0 - q_pof)
            return _finalize_binary_acq_output_to_batch(
                value,
                X,
                name="qBinaryFeasibilityWeightedAcquisition",
            )
        raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

__all__ = [
    "qBinaryProbabilityOfFeasibility",
    "qBinaryExpectedImprovement",
    "qBinaryProbabilityOfImprovement",
    "qBinaryUpperConfidenceBound",
]
