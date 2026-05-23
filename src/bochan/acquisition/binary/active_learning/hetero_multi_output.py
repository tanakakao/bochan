
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor
from botorch.utils.transforms import t_batch_mode_transform

from .multi_output import (
    _MultiOutputBinaryClassificationAcqBase,
    MultiOutputMode,
    ReductionType,
    UncertaintyScoreType,
)

class _MultiOutputHeteroMixin:
    """
    multi-output heteroscedastic active learning 用の共通 mixin。
    """

    def __init__(
        self,
        *,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_event_aggregate: str = "product",
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_model_outputs_log_var = bool(noise_model_outputs_log_var)
        self.noise_event_aggregate = noise_event_aggregate
        self.noise_weight_fn = noise_weight_fn

    # =========================================================
    # noise posterior helpers
    # =========================================================
    def _get_single_model_noise_posterior(self, model, X: Tensor):
        fn = getattr(model, "posterior_noise", None)
        if callable(fn):
            return fn(X)

        fn = getattr(model, "noise_posterior", None)
        if callable(fn):
            return fn(X)

        noise_model = getattr(model, "noise_model", None)
        if noise_model is None:
            inner_model = getattr(model, "model", None)
            if inner_model is not None:
                noise_model = getattr(inner_model, "noise_model", None)

        if noise_model is None:
            raise AttributeError(
                f"Noise posterior was not found for submodel {type(model).__name__}. "
                "Expected one of posterior_noise / noise_posterior / noise_model.posterior."
            )

        return noise_model.posterior(X)

    def _get_noise_values(self, X: Tensor) -> Tensor:
        """
        return:
            noise: (*batch, q_like, m)
        """
        if self.noise_weight_fn is not None:
            v = self.noise_weight_fn(None, X)
            return v.to(device=X.device, dtype=X.dtype)

        if hasattr(self.model, "models"):
            noise_list = []
            for submodel in self.model.models:
                noise_post = self._get_single_model_noise_posterior(submodel, X)
                noise_i = self._normalize_mean_shape(noise_post.mean, X)

                if noise_i.shape[-1] != 1:
                    raise RuntimeError(
                        f"Each submodel must contribute one noise output, got "
                        f"noise mean shape {tuple(noise_i.shape)}"
                    )

                if self.noise_model_outputs_log_var:
                    noise_i = torch.exp(noise_i.clamp(min=math.log(self.eps), max=30.0))
                else:
                    noise_i = noise_i.clamp_min(self.eps)

                noise_list.append(noise_i)

            return torch.cat(noise_list, dim=-1)

        noise_post = self._get_single_model_noise_posterior(self.model, X)
        noise_mean = self._normalize_mean_shape(noise_post.mean, X)

        if self.noise_model_outputs_log_var:
            noise_mean = torch.exp(noise_mean.clamp(min=math.log(self.eps), max=30.0))
        else:
            noise_mean = noise_mean.clamp_min(self.eps)

        return noise_mean

    # =========================================================
    # noise -> weight
    # =========================================================
    def _noise_to_weight(self, noise: Tensor) -> Tensor:
        lam = self.noise_penalty_lambda

        if self.noise_mode == "none":
            w = torch.ones_like(noise)
        elif self.noise_mode == "inverse_linear":
            w = 1.0 / (1.0 + lam * noise)
        elif self.noise_mode == "inverse_sqrt":
            w = 1.0 / torch.sqrt(1.0 + lam * noise)
        elif self.noise_mode == "exp":
            w = torch.exp(-lam * noise)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

        if self.noise_min_weight > 0.0:
            w = self.noise_min_weight + (1.0 - self.noise_min_weight) * w
        if self.noise_weight_scale != 1.0:
            w = self.noise_weight_scale * w

        return w

    def _combine_score_and_weight(self, score: Tensor, weight: Tensor) -> Tensor:
        if self.noise_combine == "multiply":
            return score * weight
        if self.noise_combine == "add":
            return score - (1.0 - weight)
        raise ValueError(f"Unknown noise_combine: {self.noise_combine}")

    def _aggregate_noise_weight(
        self,
        weight_per_output: Tensor,
        *,
        output_mode: str,
        output_weights: Tensor | None = None,
    ) -> Tensor:
        if output_mode == "all_positive":
            mode = self.noise_event_aggregate
        else:
            mode = output_mode

        if mode == "product":
            return weight_per_output.prod(dim=-1)
        if mode == "mean":
            return weight_per_output.mean(dim=-1)
        if mode == "sum":
            return weight_per_output.sum(dim=-1)
        if mode == "max":
            return weight_per_output.max(dim=-1).values
        if mode == "min":
            return weight_per_output.min(dim=-1).values
        if mode == "weighted_mean":
            if output_weights is None:
                raise ValueError("output_weights must be provided when mode='weighted_mean'.")
            w = output_weights.to(device=weight_per_output.device, dtype=weight_per_output.dtype)
            w = w / w.sum().clamp_min(self.eps)
            view_shape = (1,) * (weight_per_output.ndim - 1) + (w.numel(),)
            return (weight_per_output * w.view(*view_shape)).sum(dim=-1)

        raise ValueError(f"Unknown noise aggregation mode: {mode}")

    def _apply_noise_weight_per_output_score(self, score_per_output: Tensor, X: Tensor) -> Tensor:
        noise = self._get_noise_values(X)
        weight = self._noise_to_weight(noise)
        return self._combine_score_and_weight(score_per_output, weight)

    def _apply_noise_weight_event_score(
        self,
        score: Tensor,
        X: Tensor,
        *,
        output_mode: str,
        output_weights: Tensor | None = None,
    ) -> Tensor:
        noise = self._get_noise_values(X)
        weight_per_output = self._noise_to_weight(noise)
        weight = self._aggregate_noise_weight(
            weight_per_output,
            output_mode=output_mode,
            output_weights=output_weights,
        )
        return self._combine_score_and_weight(score, weight)


class _HeteroMultiOutputBinaryBALDBase(
    _MultiOutputHeteroMixin,
    _MultiOutputBinaryClassificationAcqBase,
):
    """
    heteroscedastic multi-output BALD acquisition.

    objective:
        noise / pending 反映後の pointwise score に適用する。
    """

    def __init__(
        self,
        model,
        num_samples: int = 16,
        reduction: str = "mean",
        output_mode: str = "all_positive",
        output_weights: Tensor | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        samples_are_probs: bool = True,
        eps: float = 1e-6,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_event_aggregate: str = "product",
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        _MultiOutputBinaryClassificationAcqBase.__init__(
            self,
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        _MultiOutputHeteroMixin.__init__(
            self,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            noise_event_aggregate=noise_event_aggregate,
            noise_weight_fn=noise_weight_fn,
        )
        self.num_samples = int(num_samples)
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.samples_are_probs = bool(samples_are_probs)
        self._set_multioutput_classification_objective(objective)

    def _event_bald(self, p: Tensor) -> Tensor:
        entropy_conditional = self._binary_entropy(p, self.eps).mean(dim=0)
        mean_prob = p.mean(dim=0)
        mean_entropy = self._binary_entropy(mean_prob, self.eps)
        return mean_entropy - entropy_conditional

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        self._set_eval_mode()

        raw_X = X
        original_batch_shape = raw_X.shape[:-2]

        Xt = self._apply_input_transform(raw_X)
        posterior = self._get_probability_posterior(raw_X)

        samples = posterior.rsample(torch.Size([self.num_samples]))
        probs = self._reshape_samples(samples, Xt, self.num_samples)
        probs = self._to_probability(
            probs,
            apply_sigmoid_if_needed=not self.samples_are_probs,
            name="probability_posterior.rsample()",
        )

        if self.output_mode == "all_positive":
            log_p_all = probs.log().sum(dim=-1)
            p_all = log_p_all.exp().clamp(self.eps, 1.0 - self.eps)
            score = self._event_bald(p_all)
            score = self._apply_noise_weight_event_score(
                score,
                Xt,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
            )
        else:
            score_per_output = self._event_bald(probs)
            score_per_output = self._apply_noise_weight_per_output_score(score_per_output, Xt)
            score = self._aggregate_outputs(
                score_per_output,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
            )

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="qHeteroMultiOutputBinaryBALD",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroMultiOutputBinaryBALD")
        return out


class _HeteroMultiOutputBinaryProbabilityVarianceBase(
    _MultiOutputHeteroMixin,
    _MultiOutputBinaryClassificationAcqBase,
):
    """
    heteroscedastic multi-output probability variance acquisition.

    objective:
        noise / pending 反映後の pointwise score に適用する。
    """

    def __init__(
        self,
        model,
        reduction: str = "mean",
        output_mode: str = "all_positive",
        output_weights: Tensor | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        mean_is_probs: bool = True,
        eps: float = 1e-6,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_event_aggregate: str = "product",
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        _MultiOutputBinaryClassificationAcqBase.__init__(
            self,
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        _MultiOutputHeteroMixin.__init__(
            self,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            noise_event_aggregate=noise_event_aggregate,
            noise_weight_fn=noise_weight_fn,
        )
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.mean_is_probs = bool(mean_is_probs)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        self._set_eval_mode()

        raw_X = X
        original_batch_shape = raw_X.shape[:-2]

        Xt = self._apply_input_transform(raw_X)
        posterior = self._get_probability_posterior(raw_X)

        probs = self._normalize_mean_shape(posterior.mean, Xt)
        probs = self._to_probability(
            probs,
            apply_sigmoid_if_needed=not self.mean_is_probs,
            name="probability_posterior.mean",
        )

        if self.output_mode == "all_positive":
            log_p_all = probs.log().sum(dim=-1)
            p_all = log_p_all.exp().clamp(self.eps, 1.0 - self.eps)
            score = p_all * (1.0 - p_all)
            score = self._apply_noise_weight_event_score(
                score,
                Xt,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
            )
        else:
            score_per_output = probs * (1.0 - probs)
            score_per_output = self._apply_noise_weight_per_output_score(score_per_output, Xt)
            score = self._aggregate_outputs(
                score_per_output,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
            )

        score = score - self._pending_penalty_per_point(Xt)

        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="qHeteroMultiOutputBinaryProbabilityVariance",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroMultiOutputBinaryProbabilityVariance")
        return out



class _HeteroMultiOutputUncertaintySamplingClassifierAcquisition(
    _MultiOutputHeteroMixin,
    _MultiOutputBinaryClassificationAcqBase,
):
    """Heteroscedastic multi-output uncertainty sampling.

    Supports the unified family through ``score_type``:
    ``entropy`` / ``variance`` / ``least_confidence``.
    """

    def __init__(
        self,
        model,
        reduction: str = "mean",
        score_type: str = "variance",
        output_mode: str = "all_positive",
        output_weights: Tensor | None = None,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        mean_is_probs: bool = True,
        eps: float = 1e-6,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_event_aggregate: str = "product",
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        _MultiOutputBinaryClassificationAcqBase.__init__(
            self,
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        _MultiOutputHeteroMixin.__init__(
            self,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            noise_event_aggregate=noise_event_aggregate,
            noise_weight_fn=noise_weight_fn,
        )
        self.score_type = score_type
        self.output_mode = output_mode
        self.output_weights = output_weights
        self.mean_is_probs = bool(mean_is_probs)
        self._set_multioutput_classification_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        self._set_eval_mode()
        raw_X = X
        original_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        posterior = self._get_probability_posterior(raw_X)
        probs = self._normalize_mean_shape(posterior.mean, Xt)
        probs = self._to_probability(
            probs,
            apply_sigmoid_if_needed=not self.mean_is_probs,
            name="probability_posterior.mean",
        )

        if self.output_mode == "all_positive":
            p_all = probs.log().sum(dim=-1).exp().clamp(self.eps, 1.0 - self.eps)
            score = self._uncertainty_score_binary_event(p_all, self.score_type)
            score = self._apply_noise_weight_event_score(
                score,
                Xt,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
            )
        else:
            score_per_output = self._uncertainty_score_binary_event(probs, self.score_type)
            score_per_output = self._apply_noise_weight_per_output_score(score_per_output, Xt)
            score = self._aggregate_outputs(
                score_per_output,
                output_mode=self.output_mode,
                output_weights=self.output_weights,
                probs_for_all_positive=probs,
                score_type_for_all_positive=self.score_type,
            )

        score = score - self._pending_penalty_per_point(Xt)
        score = self._apply_objective_to_pointwise_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name="HeteroMultiOutputUncertaintySampling",
        )
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "HeteroMultiOutputUncertaintySampling")
        return out


class qHeteroMultiOutputBinaryPredictiveEntropy(_HeteroMultiOutputUncertaintySamplingClassifierAcquisition):
    """heteroscedastic multi-output classification 用 predictive entropy acquisition。予測分布の曖昧さが大きい点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        予測が曖昧な点を探索したい場合の基本的な active learning acquisition です。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    def __init__(self, model, *args, **kwargs) -> None:
        kwargs.pop("score_type", None)
        super().__init__(model, *args, score_type="entropy", **kwargs)


class qHeteroMultiOutputBinaryProbabilityVariance(_HeteroMultiOutputUncertaintySamplingClassifierAcquisition):
    """heteroscedastic multi-output classification 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    def __init__(self, model, *args, **kwargs) -> None:
        kwargs.pop("score_type", None)
        super().__init__(model, *args, score_type="variance", **kwargs)


class qHeteroMultiOutputBinaryMarginUncertainty(_HeteroMultiOutputUncertaintySamplingClassifierAcquisition):
    """heteroscedastic multi-output classification 用 margin uncertainty acquisition。決定境界または class 境界に近い点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    def __init__(self, model, *args, **kwargs) -> None:
        kwargs.pop("score_type", None)
        super().__init__(model, *args, score_type="least_confidence", **kwargs)


class qHeteroMultiOutputBinaryBALD(_HeteroMultiOutputBinaryBALDBase):
    """heteroscedastic multi-output classification 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        BALD は predictive entropy から条件付き entropy を引いた情報利得として解釈できます。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    pass


class qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy(qHeteroMultiOutputBinaryProbabilityVariance):
    """heteroscedastic multi-output classification 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    pass

__all__ = [
    "qHeteroMultiOutputBinaryPredictiveEntropy",
    "qHeteroMultiOutputBinaryProbabilityVariance",
    "qHeteroMultiOutputBinaryMarginUncertainty",
    "qHeteroMultiOutputBinaryBALD",
    "qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy",
]
