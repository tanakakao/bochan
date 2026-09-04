"""Representation-aware explicit-task Gaussian surrogates for materials."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from torch import Tensor, nn

from .explicit_task import (
    MaterialExplicitTaskSpec,
    normalize_material_task_feature,
    validate_explicit_material_task_data,
)
from .surrogate import resolve_material_latent_dim


class MaterialExplicitTaskFeatureTransform(InputTransform):
    """Encode material inputs while preserving an explicit task coordinate.

    Raw inputs follow the Phase 35 long-format contract ``[x, task]`` with the
    task column at ``task_feature``. Only the non-task material columns are sent
    through ``feature_extractor``. The transformed representation is always
    ``[latent(x), task]``, so the downstream :class:`MultiTaskGP` can use
    ``task_feature=-1`` independently of the raw task-column position.
    """

    is_one_to_many = False

    def __init__(
        self,
        feature_extractor: nn.Module,
        *,
        task_feature: int = -1,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(feature_extractor, nn.Module):
            raise TypeError("feature_extractor must be a torch.nn.Module.")
        if isinstance(task_feature, bool) or not isinstance(task_feature, int):
            raise TypeError("task_feature must be an integer column index.")

        self.feature_extractor = feature_extractor
        self.task_feature = task_feature
        self.latent_dim = resolve_material_latent_dim(feature_extractor, latent_dim)
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True

    def transform(self, X: Tensor) -> Tensor:
        """Return ``[latent(material_X), task]`` for arbitrary batch shapes."""

        if not isinstance(X, Tensor):
            raise TypeError("X must be a torch.Tensor instance.")
        if X.ndim < 2:
            raise ValueError("X must have shape [..., q, d].")

        task_feature = normalize_material_task_feature(self.task_feature, X.shape[-1])
        task = X[..., task_feature : task_feature + 1]
        material_X = torch.cat(
            (X[..., :task_feature], X[..., task_feature + 1 :]),
            dim=-1,
        )
        latent = self.feature_extractor(material_X)
        if not isinstance(latent, Tensor):
            raise TypeError("feature_extractor must return a torch.Tensor.")
        if latent.shape[:-1] != material_X.shape[:-1]:
            raise ValueError(
                "feature_extractor must preserve all leading input dimensions."
            )
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                "feature_extractor output width does not match latent_dim: "
                f"{latent.shape[-1]} != {self.latent_dim}."
            )
        return torch.cat((latent, task.to(dtype=latent.dtype)), dim=-1)


def build_material_explicit_task_surrogate(
    *,
    train_X: Tensor,
    train_Y: Tensor,
    feature_extractor: nn.Module,
    task_spec: MaterialExplicitTaskSpec | None = None,
    train_Yvar: Tensor | None = None,
    latent_dim: int | None = None,
    rank: int | None = None,
    likelihood: Any = None,
    outcome_transform: Any = None,
    covar_module: Any = None,
    mean_module: Any = None,
    task_covar_prior: Any = None,
    validate_task_values: bool = True,
) -> MultiTaskGP:
    """Build a material-representation-aware BoTorch ``MultiTaskGP``.

    The public training inputs remain in raw material coordinates plus one task
    column. The installed input transform removes that task column, applies the
    material encoder, and appends the task id to the latent representation.

    ``outcome_transform`` defaults to ``None`` intentionally. BoTorch notes that
    multitask standardization should be stratified by task; a caller can provide
    an explicit transform when that policy is desired.
    """

    if task_spec is None:
        task_spec = MaterialExplicitTaskSpec()
    if not isinstance(task_spec, MaterialExplicitTaskSpec):
        raise TypeError("task_spec must be a MaterialExplicitTaskSpec.")

    normalized_task_feature, observed_tasks = validate_explicit_material_task_data(
        train_X,
        train_Y,
        train_Yvar,
        task_feature=task_spec.task_feature,
        all_tasks=task_spec.all_tasks,
        model_name="material explicit-task MultiTaskGP",
    )
    if isinstance(validate_task_values, bool) is False:
        raise TypeError("validate_task_values must be a bool.")

    feature_transform = MaterialExplicitTaskFeatureTransform(
        feature_extractor,
        task_feature=normalized_task_feature,
        latent_dim=latent_dim,
    )

    all_tasks: Sequence[int] | None = task_spec.all_tasks
    if all_tasks is None:
        all_tasks = observed_tasks
    output_tasks: Sequence[int] | None = task_spec.output_tasks

    kwargs: dict[str, Any] = {
        "train_X": train_X,
        "train_Y": train_Y.unsqueeze(-1) if train_Y.ndim == 1 else train_Y,
        "task_feature": -1,
        "train_Yvar": (
            train_Yvar.unsqueeze(-1)
            if train_Yvar is not None and train_Yvar.ndim == 1
            else train_Yvar
        ),
        "input_transform": feature_transform,
        "outcome_transform": outcome_transform,
        "rank": rank,
        "all_tasks": list(all_tasks),
        "output_tasks": list(output_tasks) if output_tasks is not None else None,
        "validate_task_values": validate_task_values,
    }
    if likelihood is not None:
        kwargs["likelihood"] = likelihood
    if covar_module is not None:
        kwargs["covar_module"] = covar_module
    if mean_module is not None:
        kwargs["mean_module"] = mean_module
    if task_covar_prior is not None:
        kwargs["task_covar_prior"] = task_covar_prior

    return MultiTaskGP(**kwargs)


__all__ = [
    "MaterialExplicitTaskFeatureTransform",
    "build_material_explicit_task_surrogate",
]
