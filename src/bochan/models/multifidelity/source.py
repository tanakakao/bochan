"""Discrete multi-information-source Gaussian process models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from botorch.models.multitask import MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


def _resolve_feature_index(index: int, *, d: int) -> int:
    index = int(index)
    resolved = index if index >= 0 else d + index
    if resolved < 0 or resolved >= d:
        raise ValueError(f"information source feature {index} is out of range for d={d}.")
    return resolved


def _normalize_source_value(value: Any, *, name: str) -> int:
    scalar = float(value)
    if not torch.isfinite(torch.tensor(scalar)):
        raise ValueError(f"{name} must be finite.")
    integer = int(round(scalar))
    if float(integer) != scalar:
        raise ValueError(f"{name} must be an integer task/source id, got {value!r}.")
    return integer


@dataclass(frozen=True)
class InformationSourceSpec:
    """Contract for an unordered discrete information-source feature.

    Sources are represented by integer ids in one input column, e.g.
    ``0=simulation_A, 1=simulation_B, 2=experiment``. They are modeled as ICM
    tasks rather than as an ordered continuous fidelity coordinate.
    """

    source_feature: int = -1
    source_values: Sequence[int] | None = None
    target_source: int | None = None
    source_names: Mapping[int, str] | None = None

    def resolve(self, *, d: int, train_X: Tensor | None = None) -> ResolvedInformationSourceSpec:
        feature = _resolve_feature_index(self.source_feature, d=d)
        if self.source_values is None:
            if train_X is None:
                raise ValueError(
                    "InformationSourceSpec requires source_values when train_X is unavailable."
                )
            values = tuple(
                sorted(
                    {
                        _normalize_source_value(value, name="observed source value")
                        for value in torch.as_tensor(train_X)[..., feature].reshape(-1).tolist()
                    }
                )
            )
        else:
            values = tuple(
                _normalize_source_value(value, name="source_values")
                for value in self.source_values
            )
        if not values:
            raise ValueError("source_values must contain at least one source.")
        if len(set(values)) != len(values):
            raise ValueError("source_values must not contain duplicates.")

        target = (
            values[-1]
            if self.target_source is None
            else _normalize_source_value(self.target_source, name="target_source")
        )
        if target not in values:
            raise ValueError("target_source must be one of source_values.")

        names = None
        if self.source_names is not None:
            names = {
                _normalize_source_value(key, name="source_names key"): str(value)
                for key, value in self.source_names.items()
            }
            unknown = set(names) - set(values)
            if unknown:
                raise ValueError(
                    f"source_names contains ids not present in source_values: {sorted(unknown)}."
                )

        return ResolvedInformationSourceSpec(
            source_feature=feature,
            source_values=values,
            target_source=target,
            source_names=names,
        )


@dataclass(frozen=True)
class ResolvedInformationSourceSpec:
    source_feature: int
    source_values: tuple[int, ...]
    target_source: int
    source_names: Mapping[int, str] | None = None


class GaussianMultiSourceGP(MultiTaskGP):
    """ICM multi-task GP for unordered discrete information sources."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        source_spec: InformationSourceSpec | ResolvedInformationSourceSpec | None = None,
        source_feature: int = -1,
        source_values: Sequence[int] | None = None,
        target_source: int | None = None,
        source_names: Mapping[int, str] | None = None,
        rank: int | None = None,
        outcome_transform: Any | None = None,
        input_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        X = torch.as_tensor(train_X)
        Y = torch.as_tensor(train_Y, dtype=X.dtype, device=X.device)
        if X.ndim != 2:
            raise ValueError("GaussianMultiSourceGP requires train_X with shape n x d.")
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)
        if Y.ndim != 2 or Y.shape[-1] != 1:
            raise ValueError("GaussianMultiSourceGP requires scalar train_Y with shape n or n x 1.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of rows.")
        if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(Y).all()):
            raise ValueError("GaussianMultiSourceGP requires finite train_X and train_Y.")

        spec = source_spec or InformationSourceSpec(
            source_feature=source_feature,
            source_values=source_values,
            target_source=target_source,
            source_names=source_names,
        )
        resolved = (
            spec
            if isinstance(spec, ResolvedInformationSourceSpec)
            else spec.resolve(d=int(X.shape[-1]), train_X=X)
        )
        observed = {
            _normalize_source_value(value, name="observed source value")
            for value in X[:, resolved.source_feature].tolist()
        }
        unknown = observed - set(resolved.source_values)
        if unknown:
            raise ValueError(
                f"train_X contains source ids not declared in source_values: {sorted(unknown)}."
            )

        yvar = None
        if train_Yvar is not None:
            yvar = torch.as_tensor(train_Yvar, dtype=X.dtype, device=X.device)
            if yvar.ndim == 2 and yvar.shape[-1] == 1:
                yvar = yvar.squeeze(-1)
            if yvar.ndim != 1 or yvar.shape[0] != X.shape[0]:
                raise ValueError("train_Yvar must have shape n or n x 1 for GaussianMultiSourceGP.")

        model_kwargs = dict(kwargs)
        if outcome_transform is not None:
            model_kwargs["outcome_transform"] = outcome_transform
        if input_transform is not None:
            model_kwargs["input_transform"] = input_transform
        super().__init__(
            train_X=X,
            train_Y=Y,
            train_Yvar=yvar,
            task_feature=resolved.source_feature,
            all_tasks=list(resolved.source_values),
            output_tasks=[resolved.target_source],
            rank=rank,
            **model_kwargs,
        )

        self.source_spec = resolved
        self.information_source_feature = resolved.source_feature
        self.information_source_values = tuple(resolved.source_values)
        self.target_information_source = resolved.target_source
        self.information_source_names = dict(resolved.source_names or {})

        # Compatibility bridge: existing MFKG / MF-MES helpers operate on a
        # fidelity-feature metadata contract. For an information-source model,
        # these fields identify a discrete task axis, not a continuous fidelity.
        self.fidelity_mode = "information_source"
        self.fidelity_features = (resolved.source_feature,)
        self.target_fidelities = {resolved.source_feature: float(resolved.target_source)}
        self.input_mode = "continuous"
        self.cat_dims: tuple[int, ...] = ()
        self.is_multifidelity_model = True
        self.information_source_model = True

    def fidelity_metadata(self) -> dict[str, Any]:
        return {
            "fidelity_mode": self.fidelity_mode,
            "fidelity_features": self.fidelity_features,
            "target_fidelities": dict(self.target_fidelities),
            "input_mode": self.input_mode,
            "cat_dims": self.cat_dims,
            "information_source_feature": self.information_source_feature,
            "information_source_values": self.information_source_values,
            "target_information_source": self.target_information_source,
            "information_source_names": dict(self.information_source_names),
        }

    def make_mll(self) -> ExactMarginalLogLikelihood:
        return ExactMarginalLogLikelihood(self.likelihood, self)


__all__ = [
    "GaussianMultiSourceGP",
    "InformationSourceSpec",
    "ResolvedInformationSourceSpec",
]
