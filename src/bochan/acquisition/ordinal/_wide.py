"""Explicit adapters for wide correlated ordinal acquisition models."""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.model import Model
from torch import Tensor


class WideOrdinalTaskProxy(Model):
    """Expose one task of a correlated wide ordinal model as a scalar model.

    The proxy does not copy model parameters. Posterior evaluation is delegated
    to the parent long-format multitask GP with a fixed task-id column.
    """

    def __init__(self, parent: Model, task_index: int) -> None:
        super().__init__()
        object.__setattr__(self, "_parent", parent)
        self.task_index = int(task_index)

    @property
    def parent(self) -> Model:
        return object.__getattribute__(self, "_parent")

    @property
    def likelihood(self):
        return getattr(self.parent, "likelihood", None)

    @property
    def ordinal_likelihood(self):
        likelihood = getattr(self.parent, "ordinal_likelihood", None)
        return self.likelihood if likelihood is None else likelihood

    @property
    def input_transform(self):
        return getattr(self.parent, "input_transform", None)

    @property
    def num_outputs(self) -> int:
        return 1

    def eval(self):
        self.parent.eval()
        likelihood = self.likelihood
        if likelihood is not None and hasattr(likelihood, "eval"):
            likelihood.eval()
        return self

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ):
        if output_indices not in (None, [0], (0,)):
            raise ValueError("A fixed ordinal task proxy exposes one output only.")

        X = torch.as_tensor(X)
        task = torch.full(
            (*X.shape[:-1], 1),
            float(self.task_index),
            device=X.device,
            dtype=X.dtype,
        )
        X_long = torch.cat([X, task], dim=-1)

        from bochan.models.ordinal.base.multitask import MultiTaskOrdinalGPModel

        return MultiTaskOrdinalGPModel.posterior(
            self.parent,
            X_long,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.parent, name)


class WideOrdinalAcquisitionAdapter(Model):
    """ModelList-like acquisition view over a correlated wide ordinal model."""

    def __init__(self, parent: Model) -> None:
        super().__init__()
        object.__setattr__(self, "_parent", parent)
        num_tasks = int(getattr(parent, "num_tasks"))
        object.__setattr__(
            self,
            "models",
            [WideOrdinalTaskProxy(parent, index) for index in range(num_tasks)],
        )

    @property
    def parent(self) -> Model:
        return object.__getattribute__(self, "_parent")

    @property
    def num_outputs(self) -> int:
        return len(object.__getattribute__(self, "models"))

    def eval(self):
        self.parent.eval()
        return self

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return self.parent.posterior(X, *args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.parent, name)


def adapt_wide_ordinal_model(model: Model) -> Model:
    """Return an explicit fixed-task acquisition adapter when required."""

    if isinstance(model, WideOrdinalAcquisitionAdapter):
        return model
    if getattr(model, "models", None) is not None:
        return model
    if not callable(getattr(model, "_wrap_wide_posterior", None)):
        return model
    try:
        num_tasks = int(getattr(model, "num_tasks"))
    except (AttributeError, TypeError, ValueError):
        return model
    if num_tasks <= 1:
        return model
    return WideOrdinalAcquisitionAdapter(model)


__all__ = [
    "WideOrdinalAcquisitionAdapter",
    "WideOrdinalTaskProxy",
    "adapt_wide_ordinal_model",
]
