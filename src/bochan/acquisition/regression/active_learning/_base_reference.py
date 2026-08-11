"""Reference handling and transformed-distance helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ._base_common import ReductionType, _ensure_q_batch, _reduce, _safe_prod


class _RegressionReferenceMixin:
    # ------------------------------------------------------------
    # Reference handling
    # ------------------------------------------------------------
    def _coerce_reference_to_tensor(
        self,
        ref,
        *,
        like: Tensor | None = None,
    ) -> Tensor | None:
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
                    out = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=-2)
        else:
            raise TypeError(
                "Reference points must be None, Tensor, list, or tuple. "
                f"Got {type(ref)}."
            )

        if like is not None:
            out = out.to(device=like.device, dtype=like.dtype)

        # Reference points are constants during acquisition optimization.
        return out.detach()

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = self._coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        self.X_observed = self._coerce_reference_to_tensor(X_observed)

    # ------------------------------------------------------------
    # Transform / shape helpers
    # ------------------------------------------------------------
    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None and hasattr(likelihood, "eval"):
            likelihood.eval()

    @staticmethod
    def _unwrap_transformed_inputs(Xt: Tensor | tuple[Tensor, ...]) -> Tensor:
        if isinstance(Xt, tuple):
            return Xt[0]
        return Xt

    def _apply_input_transform_for_distance(self, X: Tensor) -> Tensor:
        """Apply model input transform for distance / penalty calculations.

        Prefer ``model.transform_inputs`` over directly calling
        ``model.input_transform``.  Wrapper models such as SaasGaussianMixedGPModel,
        PCA, and REMBO have an internal representation that differs from raw
        candidate space.  Calling ``input_transform`` directly on raw ``X`` can
        therefore apply an encoded-space transform to raw-space data.
        """
        X = _ensure_q_batch(X)

        transform_inputs = getattr(self.model, "transform_inputs", None)
        if callable(transform_inputs):
            try:
                Xt = self._unwrap_transformed_inputs(transform_inputs(X))
                return _ensure_q_batch(Xt)
            except Exception:
                # Fall back to raw input_transform below for plain models or
                # wrappers whose transform_inputs intentionally rejects X.
                pass

        models = getattr(self.model, "models", None)
        if models is not None and len(models) > 0:
            transform_inputs = getattr(models[0], "transform_inputs", None)
            if callable(transform_inputs):
                try:
                    Xt = self._unwrap_transformed_inputs(transform_inputs(X))
                    return _ensure_q_batch(Xt)
                except Exception:
                    pass

        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = self._unwrap_transformed_inputs(it(X))
            return _ensure_q_batch(Xt)

        if models is not None and len(models) > 0:
            it = getattr(models[0], "input_transform", None)
            if it is not None:
                Xt = self._unwrap_transformed_inputs(it(X))
                return _ensure_q_batch(Xt)

        return X
    def _reference_to_distance_space(
        self,
        ref,
        *,
        like: Tensor,
    ) -> Tensor | None:
        ref = self._coerce_reference_to_tensor(ref, like=like)
        if ref is None or ref.numel() == 0:
            return None
        ref_t = self._apply_input_transform_for_distance(ref)
        return _ensure_q_batch(ref_t).to(device=like.device, dtype=like.dtype)

    def _align_pointwise_score_to_X(
        self,
        score: Tensor,
        Xt: Tensor,
        *,
        name: str,
        reduce_extra: ReductionType = "mean",
    ) -> Tensor:
        """Align pointwise score to ``Xt.shape[:-1]``.

        This handles extra leading dimensions from fully Bayesian / ensemble
        models by reducing them.
        """
        Xt = _ensure_q_batch(Xt)
        target = torch.Size(Xt.shape[:-1])
        out = score

        if out.shape == target:
            return out

        # Drop singleton output dim only, not q=1.
        if out.ndim >= 1 and out.shape[-1] == 1:
            out_s = out.squeeze(-1)
            if out_s.shape == target:
                return out_s
            out = out_s

        if out.shape == target:
            return out

        # Reduce leading extra dims until ranks match.
        while out.ndim > len(target):
            out = _reduce(out, dim=0, mode=reduce_extra)
            if out.shape == target:
                return out

        if out.shape == target:
            return out

        if out.numel() == _safe_prod(target):
            return out.reshape(target)

        raise RuntimeError(
            f"{name}: score shape mismatch. "
            f"score.shape={tuple(score.shape)}, expected={tuple(target)}, Xt.shape={tuple(Xt.shape)}."
        )

    def _reduce_outputs_if_needed(self, value: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        """Reduce posterior output dimension to a pointwise scalar score."""
        Xt = _ensure_q_batch(Xt)
        target_prefix = torch.Size(Xt.shape[:-1])
        out = value

        if out.shape == target_prefix:
            return out

        # Reduce leading MCMC / model-list batch dims.
        while out.ndim > len(target_prefix) + 1:
            out = out.mean(dim=0)
            if out.shape == target_prefix:
                return out

        # Single-output posterior: (..., q_like, 1)
        if out.ndim == len(target_prefix) + 1 and out.shape[:-1] == target_prefix:
            if out.shape[-1] == 1:
                return out.squeeze(-1)
            return _reduce(out, dim=-1, mode=self.output_reduction)

        if out.ndim == len(target_prefix) and out.shape == target_prefix:
            return out

        if out.numel() % max(_safe_prod(target_prefix), 1) == 0:
            m = out.numel() // max(_safe_prod(target_prefix), 1)
            out = out.reshape(*target_prefix, m)
            if m == 1:
                return out.squeeze(-1)
            return _reduce(out, dim=-1, mode=self.output_reduction)

        raise RuntimeError(
            f"{name}: could not reduce output dimension. "
            f"value.shape={tuple(value.shape)}, Xt.shape={tuple(Xt.shape)}."
        )

