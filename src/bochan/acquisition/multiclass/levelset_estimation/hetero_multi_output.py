from __future__ import annotations

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.active_learning.hetero_multi_output import _HeteroMultiOutputMulticlassMixin

from .sample_compat import apply_levelset_q_like_compat
from .multi_output import (
    _boundary_weight,
    _class_entropy,
    qMultiOutputMulticlassBoundaryVarianceAcquisition,
    qMultiOutputMulticlassClassEntropyAcquisition,
    qMultiOutputMulticlassICUAcquisition,
    qMultiOutputMulticlassJointLatentStraddleAcquisition,
    qMultiOutputMulticlassLatentStraddleAcquisition,
    qMultiOutputMulticlassLevelSetUncertainty,
    qMultiOutputMulticlassProbabilityOfExceedance,
)

# Direct imports from this module should also get the q_like compatibility patch.
apply_levelset_q_like_compat()


class qHeteroMultiOutputMulticlassLatentStraddleAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassLatentStraddleAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output_raw(raw_X, name=f"{self.__class__.__name__}.target_prob")
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score_per_output = self.beta * uncertainty - (p - self.threshold).abs()
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassJointLatentStraddleAcquisition,
):
    def _joint_uncertainty_per_output(self, samples: Tensor) -> Tensor:
        """Compute joint q uncertainty per output.

        Args:
            samples: ``S x batch_shape x q x m`` target-probability samples.

        Returns:
            Tensor with shape ``batch_shape x m``.
        """
        sample_count = int(samples.shape[0])
        q = int(samples.shape[-2])
        m = int(samples.shape[-1])
        pieces = []
        eye = torch.eye(q, dtype=samples.dtype, device=samples.device)
        for output_index in range(m):
            y = samples[..., output_index]  # S x batch_shape x q
            y = y.movedim(0, -1)  # batch_shape x q x S
            y = y - y.mean(dim=-1, keepdim=True)
            cov = torch.matmul(y, y.transpose(-1, -2)) / max(sample_count - 1, 1)
            cov = 0.5 * (cov + cov.transpose(-1, -2)) + self.jitter * eye

            if self.uncertainty_mode == "logdet1p":
                tau2 = max(self.tau**2, self.eps)
                sign, logabsdet = torch.linalg.slogdet(eye + cov / tau2)
                if not torch.all(sign > 0):
                    raise RuntimeError("Non-positive definite matrix encountered in logdet1p.")
                value = 0.5 * logabsdet
            elif self.uncertainty_mode == "logdet":
                sign, logabsdet = torch.linalg.slogdet(cov)
                if not torch.all(sign > 0):
                    raise RuntimeError("Non-positive definite covariance encountered in logdet.")
                value = 0.5 * logabsdet
            elif self.uncertainty_mode == "sqrt_trace":
                value = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(self.eps).sqrt()
            elif self.uncertainty_mode == "trace":
                value = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
            else:
                raise ValueError(f"Unknown uncertainty_mode: {self.uncertainty_mode!r}.")
            pieces.append(value.unsqueeze(-1))
        return torch.cat(pieces, dim=-1)

    def _boundary_distance_per_output(self, mean: Tensor) -> Tensor:
        """Compute distance from target level per output.

        Args:
            mean: ``batch_shape x q x m`` target-probability mean.

        Returns:
            Tensor with shape ``batch_shape x m``.
        """
        diff = mean - self.threshold
        if self.boundary_mode in {"mean_abs", "l2_mean"}:
            if self.boundary_mode == "mean_abs":
                return diff.abs().mean(dim=-2)
            return diff.pow(2).mean(dim=-2).sqrt()
        if self.boundary_mode == "max_abs":
            return diff.abs().amax(dim=-2)
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")

    def _joint_score_per_output(self, X: Tensor) -> Tensor:
        """Return q-aggregated joint score for each output.

        Shape is ``batch_shape x m``. This method is intentionally hetero-specific:
        noise weighting is applied after q aggregation but before output aggregation.
        """
        Xq = self._ensure_q_batch(X)
        target_samples = self._sample_target_probs(Xq)
        samples = self._flatten_samples(target_samples)
        mean = samples.mean(dim=0)
        uncertainty = self._joint_uncertainty_per_output(samples)
        boundary = self._boundary_distance_per_output(mean)
        return self.beta * uncertainty - boundary

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        batch_shape = raw_X.shape[:-2]
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            Xp = Xp.detach().to(device=raw_X.device, dtype=raw_X.dtype)

        if Xp is None or Xp.numel() == 0 or not self.marginalize_pending:
            score_per_output = self._joint_score_per_output(raw_X)
            score_per_output = self._apply_noise_to_q_aggregated_output_score(score_per_output, self._apply_input_transform(raw_X))
            value = self._aggregate_outputs(score_per_output)
            value = value - self._repulsion_penalty(raw_X)
            value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
            return self._finalize(value, raw_X, name=self.__class__.__name__)

        Xp_batch = self._expand_pending_to_batch(Xp, batch_shape)
        pending_score = self._joint_score_per_output(Xp_batch)
        all_score = self._joint_score_per_output(torch.cat([Xp_batch, raw_X], dim=-2))
        score_per_output = all_score - pending_score
        score_per_output = self._apply_noise_to_q_aggregated_output_score(score_per_output, self._apply_input_transform(raw_X))
        value = self._aggregate_outputs(score_per_output)
        value = value - self._repulsion_penalty(raw_X)
        value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassICUAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassICUAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output_raw(raw_X, name=f"{self.__class__.__name__}.target_prob")
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score_per_output = uncertainty.pow(2) * contour_weight
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassBoundaryVarianceAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output_raw(raw_X, name=f"{self.__class__.__name__}.target_prob")
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score_per_output = uncertainty.pow(2) * _boundary_weight(p, self.threshold, bandwidth=self.bandwidth, eps=self.eps)
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassClassEntropyAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassClassEntropyAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        if self.target_class is None and self.output_target_classes is None:
            score_per_output = _class_entropy(probs, eps=self.eps)
        else:
            selected = self._target_prob_per_output(probs)
            score_per_output = -(selected.clamp_min(self.eps) * selected.clamp_min(self.eps).log())
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassProbabilityOfExceedance(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassProbabilityOfExceedance,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output_raw(raw_X, name=f"{self.__class__.__name__}.target_prob")
        score_per_output = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassLevelSetUncertainty(qHeteroMultiOutputMulticlassICUAcquisition):
    pass


__all__ = [
    "qHeteroMultiOutputMulticlassLatentStraddleAcquisition",
    "qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition",
    "qHeteroMultiOutputMulticlassICUAcquisition",
    "qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition",
    "qHeteroMultiOutputMulticlassClassEntropyAcquisition",
    "qHeteroMultiOutputMulticlassProbabilityOfExceedance",
    "qHeteroMultiOutputMulticlassLevelSetUncertainty",
]
