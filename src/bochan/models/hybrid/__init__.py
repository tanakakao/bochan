from __future__ import annotations

from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from torch import Tensor

from .class_probability_shapes import _select_class_probability_output
from .multi_output import HybridMultiOutputModel as _HybridMultiOutputModel
from .posterior import HybridPosterior
from .prediction import predict_class as _predict_class
from .prediction import predict_class_list as _predict_class_list
from .specs import OutputIndex, OutputSpec, PosteriorMode, TaskType
from .task_aware_posterior import TaskAwareHybridPosterior
from .task_aware_sampling import _task_aware_posterior


class HybridMultiOutputModel(_HybridMultiOutputModel):
    """Canonical hybrid multi-output model with task-aware posterior behavior."""

    _task_aware_original_posterior = _HybridMultiOutputModel.posterior

    def _set_transformed_inputs(self) -> None:
        return None

    def eval(self):
        self.training = False
        for model in self.models:
            model.eval()
        return self

    def _ordinal_class_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return _select_class_probability_output(
                    probs,
                    X,
                    output_index=spec.output_index,
                    name=f"{spec.name}.class_probs",
                ).clamp_min(0.0)
        return super()._ordinal_class_probs(spec, X, **kwargs)

    def _multiclass_probs(self, spec: OutputSpec, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return _select_class_probability_output(
                    probs,
                    X,
                    output_index=spec.output_index,
                    name=f"{spec.name}.class_probs",
                ).clamp_min(0.0)

        post = self._call_accessor(
            spec.model,
            ("probability_posterior", "posterior"),
            X,
            **kwargs,
        )
        probs, _ = self._posterior_mean_variance(post, spec.name)
        return _select_class_probability_output(
            probs,
            X,
            output_index=spec.output_index,
            name=f"{spec.name}.posterior.mean",
        ).clamp_min(0.0)

    def posterior(
        self,
        X: Tensor,
        output_indices: Any = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        *,
        output_mode: PosteriorMode = "objective",
        **kwargs: Any,
    ):
        self.eval()
        return _task_aware_posterior(
            self,
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            output_mode=output_mode,
            **kwargs,
        )

    def class_probs_list(
        self,
        X: Tensor,
        output_indices: OutputIndex | list[OutputIndex] | Tensor | None = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Return class probabilities for selected classification outputs."""
        self.eval()
        X = self._unwrap_X(X)
        outputs: list[Tensor] = []
        for index in self._normalize_output_indices(output_indices):
            spec = self.specs[index]
            if spec.task_type == "binary":
                _, _, p1 = self._binary_probability_stats(spec, X, **kwargs)
                outputs.append(torch.stack([1.0 - p1, p1], dim=-1))
            elif spec.task_type == "ordinal":
                outputs.append(self._ordinal_class_probs(spec, X, **kwargs))
            elif spec.task_type == "multiclass":
                outputs.append(self._multiclass_probs(spec, X, **kwargs))
            else:
                raise TypeError(
                    f"Output {spec.name!r} is regression and has no class probabilities."
                )
        return outputs

    def predict_class_list(
        self,
        X: Tensor,
        output_indices: OutputIndex | list[OutputIndex] | Tensor | None = None,
        *,
        binary_threshold: float | Tensor = 0.5,
        **kwargs: Any,
    ) -> list[Tensor]:
        return _predict_class_list(
            self,
            X,
            output_indices=output_indices,
            binary_threshold=binary_threshold,
            **kwargs,
        )

    def predict_class(
        self,
        X: Tensor,
        output_indices: OutputIndex | list[OutputIndex] | Tensor | None = None,
        *,
        binary_threshold: float | Tensor = 0.5,
        **kwargs: Any,
    ) -> Tensor:
        return _predict_class(
            self,
            X,
            output_indices=output_indices,
            binary_threshold=binary_threshold,
            **kwargs,
        )


__all__ = [
    "HybridMultiOutputModel",
    "HybridPosterior",
    "TaskAwareHybridPosterior",
    "OutputIndex",
    "OutputSpec",
    "PosteriorMode",
    "TaskType",
]
