"""Independent model-list composition for non-Gaussian regression."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models.model import Model
from botorch.posteriors.posterior_list import PosteriorList
from torch import Tensor
from torch.nn import ModuleList


class NonGaussianModelList(Model):
    """Treat independent single-output non-Gaussian models as multiple outputs.

    Unlike ``ModelListGP``, this class never unwraps or Gaussianizes a custom
    response posterior. Consequently Gamma/Beta support and count-family rate
    semantics remain owned by each submodel.
    """

    is_non_gaussian_model_list = True

    def __init__(self, *models: Model) -> None:
        """Initialize an independent model list.

        Args:
            *models: One or more single-output BoTorch-compatible models.

        Raises:
            ValueError: If no model is supplied or a model is not single-output.
        """
        super().__init__()
        if not models:
            raise ValueError("At least one non-Gaussian submodel is required.")
        for index, model in enumerate(models):
            if getattr(model, "num_outputs", None) != 1:
                raise ValueError(f"Submodel {index} must be single-output.")
        self.models = ModuleList(models)

    @property
    def num_outputs(self) -> int:
        """Return the number of independent outputs."""
        return len(self.models)

    @property
    def batch_shape(self) -> torch.Size:
        """Return the common submodel batch shape."""
        shapes = [getattr(model, "batch_shape", torch.Size()) for model in self.models]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise NotImplementedError("All submodels must have the same batch_shape.")
        return shapes[0]

    @staticmethod
    def _raw_x(model: Model) -> Tensor:
        """Extract a submodel's raw training input."""
        for name in ("raw_train_X", "train_X_original", "train_inputs_raw", "train_inputs"):
            if hasattr(model, name):
                value = getattr(model, name)
                return value[0] if isinstance(value, tuple) else value
        raise AttributeError(f"{type(model).__name__} does not expose training inputs.")

    @staticmethod
    def _y(model: Model) -> Tensor:
        """Extract a submodel's training target."""
        for name in ("train_Y", "train_targets_raw", "train_targets"):
            if hasattr(model, name):
                value = getattr(model, name)
                return value.unsqueeze(-1) if value.ndim == 1 else value
        raise AttributeError(f"{type(model).__name__} does not expose training targets.")

    @property
    def raw_train_X(self) -> Tensor:
        """Return raw training inputs from the first submodel."""
        return self._raw_x(self.models[0])

    @property
    def train_X(self) -> Tensor:
        """Return the public raw input design."""
        return self.raw_train_X

    @property
    def train_inputs(self) -> tuple[Tensor]:
        """Return BoTorch-style training inputs."""
        return (self.train_X,)

    @property
    def train_Y(self) -> Tensor:
        """Concatenate submodel targets along the output axis."""
        return torch.cat([self._y(model) for model in self.models], dim=-1)

    @property
    def train_targets(self) -> Tensor:
        """Return the wide training targets."""
        return self.train_Y

    def _indices(self, output_indices: Sequence[int] | Tensor | None) -> list[int]:
        """Normalize and validate requested output indices."""
        values = list(range(self.num_outputs)) if output_indices is None else list(output_indices)
        values = [int(value) for value in values]
        if any(value < 0 or value >= self.num_outputs for value in values):
            raise IndexError("output_indices contains an out-of-range output.")
        return values

    def posterior(self, X: Tensor, output_indices: Sequence[int] | Tensor | None = None, **kwargs: Any) -> PosteriorList:
        """Return unmodified response-scale posteriors in a ``PosteriorList``."""
        return PosteriorList(*[self.models[i].posterior(X, **kwargs) for i in self._indices(output_indices)])

    def latent_posterior(self, X: Tensor, output_indices: Sequence[int] | Tensor | None = None, **kwargs: Any) -> PosteriorList:
        """Return independent latent posteriors in a ``PosteriorList``."""
        posts = []
        for i in self._indices(output_indices):
            accessor = getattr(self.models[i], "latent_posterior", None)
            posts.append(accessor(X, **kwargs) if callable(accessor) else self.models[i].posterior(X, **kwargs))
        return PosteriorList(*posts)

    def subset_output(self, idcs: Sequence[int]) -> NonGaussianModelList:
        """Return a model list containing only selected outputs."""
        return type(self)(*[self.models[i] for i in self._indices(idcs)])

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> NonGaussianModelList:
        """Condition each independent submodel on its corresponding target column."""
        if Y.shape[-1] != self.num_outputs:
            raise ValueError("Y's final dimension must equal num_outputs.")
        conditioned = [
            model.condition_on_observations(X, Y[..., i : i + 1], **kwargs)
            for i, model in enumerate(self.models)
        ]
        return type(self)(*conditioned)

    def fantasize(self, X: Tensor, sampler: Any, **kwargs: Any) -> NonGaussianModelList:
        """Delegate fantasy construction only when every submodel supports it."""
        unsupported = [type(model).__name__ for model in self.models if not callable(getattr(model, "fantasize", None))]
        if unsupported:
            raise NotImplementedError(f"fantasize is unsupported by submodels: {unsupported}.")
        return type(self)(*[model.fantasize(X, sampler, **kwargs) for model in self.models])


__all__ = ["NonGaussianModelList"]
