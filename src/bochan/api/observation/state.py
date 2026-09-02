"""Observation-state handling for physical experiment workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ObservationRowStatus = Literal["success", "failed", "pending"]
FailureQReduction = Literal["prod", "min", "mean"]


def _torch():
    import torch

    return torch


def _as_bool_row_mask(value: Any | None, *, n_rows: int, device: Any, name: str):
    torch = _torch()
    if value is None:
        return torch.zeros(n_rows, dtype=torch.bool, device=device)
    mask = torch.as_tensor(value, dtype=torch.bool, device=device)
    if mask.ndim != 1 or int(mask.shape[0]) != n_rows:
        raise ValueError(f"{name} must have shape [{n_rows}], got {tuple(mask.shape)}.")
    return mask


def _as_observed_mask(value: Any | None, *, Y: Any):
    torch = _torch()
    finite = torch.isfinite(Y)
    if value is None:
        return finite
    mask = torch.as_tensor(value, dtype=torch.bool, device=Y.device)
    if tuple(mask.shape) != tuple(Y.shape):
        raise ValueError(
            "observed_mask must have the same shape as Y. "
            f"Y={tuple(Y.shape)}, observed_mask={tuple(mask.shape)}."
        )
    if bool((mask & ~finite).any()):
        raise ValueError("observed_mask cannot mark NaN / inf target cells as observed.")
    return mask


@dataclass
class ObservationData:
    """Canonical tensor representation of experiment observations."""

    X: Any
    Y: Any
    observed_mask: Any | None = None
    failed_mask: Any | None = None
    pending_mask: Any | None = None
    Yvar: Any | None = None

    def __post_init__(self) -> None:
        torch = _torch()
        X = torch.as_tensor(self.X)
        if X.ndim != 2:
            raise ValueError(f"X must have shape [n, d], got {tuple(X.shape)}.")

        Y = torch.as_tensor(self.Y, device=X.device)
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)
        if Y.ndim != 2:
            raise ValueError(f"Y must have shape [n, m], got {tuple(Y.shape)}.")
        if int(Y.shape[0]) != int(X.shape[0]):
            raise ValueError("X and Y must contain the same number of rows.")
        if not torch.is_floating_point(Y):
            Y = Y.to(
                dtype=X.dtype if torch.is_floating_point(X) else torch.get_default_dtype()
            )
        if bool(torch.isinf(Y).any()):
            raise ValueError("Y may contain NaN for unobserved targets, but not +/-inf.")

        n_rows = int(X.shape[0])
        observed = _as_observed_mask(self.observed_mask, Y=Y)
        failed = _as_bool_row_mask(
            self.failed_mask,
            n_rows=n_rows,
            device=X.device,
            name="failed_mask",
        )
        pending = _as_bool_row_mask(
            self.pending_mask,
            n_rows=n_rows,
            device=X.device,
            name="pending_mask",
        )
        if bool((failed & pending).any()):
            raise ValueError("A row cannot be both failed and pending.")

        unavailable = failed | pending
        if bool(unavailable.any()):
            observed = observed & ~unavailable.unsqueeze(-1)

        canonical_y = torch.full_like(Y, float("nan"))
        canonical_y = torch.where(observed, Y, canonical_y)

        canonical_yvar = None
        if self.Yvar is not None:
            Yvar = torch.as_tensor(self.Yvar, device=X.device)
            if Yvar.ndim == 1:
                Yvar = Yvar.unsqueeze(-1)
            if tuple(Yvar.shape) != tuple(Y.shape):
                raise ValueError(
                    "Yvar must have the same shape as Y. "
                    f"Y={tuple(Y.shape)}, Yvar={tuple(Yvar.shape)}."
                )
            Yvar = Yvar.to(dtype=Y.dtype)
            invalid_observed = observed & (~torch.isfinite(Yvar) | (Yvar <= 0.0))
            if bool(invalid_observed.any()):
                raise ValueError(
                    "Every observed target cell requires a finite, strictly positive Yvar."
                )
            canonical_yvar = torch.full_like(Yvar, float("nan"))
            canonical_yvar = torch.where(observed, Yvar, canonical_yvar)

        self.X = X
        self.Y = canonical_y
        self.Yvar = canonical_yvar
        self.observed_mask = observed
        self.failed_mask = failed
        self.pending_mask = pending

    @classmethod
    def from_status(
        cls,
        X: Any,
        Y: Any,
        *,
        status: Any,
        observed_mask: Any | None = None,
        Yvar: Any | None = None,
    ) -> ObservationData:
        statuses = [str(value).strip().lower() for value in list(status)]
        valid = {"success", "failed", "pending"}
        invalid = sorted(set(statuses) - valid)
        if invalid:
            raise ValueError(
                "status values must be 'success', 'failed', or 'pending'. "
                f"Invalid values: {invalid}."
            )
        return cls(
            X=X,
            Y=Y,
            observed_mask=observed_mask,
            failed_mask=[value == "failed" for value in statuses],
            pending_mask=[value == "pending" for value in statuses],
            Yvar=Yvar,
        )

    @property
    def completed_mask(self):
        return ~self.pending_mask

    @property
    def success_mask(self):
        return self.completed_mask & ~self.failed_mask

    @property
    def objective_row_mask(self):
        return self.success_mask & self.observed_mask.any(dim=-1)

    @property
    def pending_X(self):
        return self.X[self.pending_mask]

    @property
    def completed_X(self):
        return self.X[self.completed_mask]

    def objective_training_data(self) -> tuple[Any, Any]:
        mask = self.objective_row_mask
        if not bool(mask.any()):
            raise ValueError("No successful experiment contains an observed objective value.")
        return self.X[mask], self.Y[mask]

    def objective_training_data_with_variance(self) -> tuple[Any, Any, Any | None]:
        """Return successful objective rows with cell-aligned known variance."""
        mask = self.objective_row_mask
        if not bool(mask.any()):
            raise ValueError("No successful experiment contains an observed objective value.")
        Yvar = None if self.Yvar is None else self.Yvar[mask]
        return self.X[mask], self.Y[mask], Yvar

    def output_training_data(self, output_index: int) -> tuple[Any, Any]:
        index = int(output_index)
        if index < 0 or index >= int(self.Y.shape[-1]):
            raise IndexError(
                f"output_index={index} is outside [0, {int(self.Y.shape[-1]) - 1}]."
            )
        mask = self.success_mask & self.observed_mask[:, index]
        if not bool(mask.any()):
            raise ValueError(f"Output {index} has no successful observed values.")
        return self.X[mask], self.Y[mask, index : index + 1]

    def output_training_data_with_variance(
        self,
        output_index: int,
    ) -> tuple[Any, Any, Any | None]:
        """Return one observed output with its known observation variance."""
        index = int(output_index)
        if index < 0 or index >= int(self.Y.shape[-1]):
            raise IndexError(
                f"output_index={index} is outside [0, {int(self.Y.shape[-1]) - 1}]."
            )
        mask = self.success_mask & self.observed_mask[:, index]
        if not bool(mask.any()):
            raise ValueError(f"Output {index} has no successful observed values.")
        Yvar = None
        if self.Yvar is not None:
            Yvar = self.Yvar[mask, index : index + 1]
        return self.X[mask], self.Y[mask, index : index + 1], Yvar

    def success_training_data(self) -> tuple[Any, Any]:
        torch = _torch()
        completed = self.completed_mask
        if not bool(completed.any()):
            raise ValueError("No completed experiments are available for a success model.")
        X = self.X[completed]
        y = (~self.failed_mask[completed]).to(dtype=self.Y.dtype).unsqueeze(-1)
        if not torch.isfinite(y).all():
            raise RuntimeError("Success labels must be finite.")
        return X, y

    def append(self, other: ObservationData) -> ObservationData:
        torch = _torch()
        if int(self.X.shape[-1]) != int(other.X.shape[-1]):
            raise ValueError("ObservationData feature dimensions must match.")
        if int(self.Y.shape[-1]) != int(other.Y.shape[-1]):
            raise ValueError("ObservationData target dimensions must match.")
        if (self.Yvar is None) != (other.Yvar is None):
            raise ValueError(
                "ObservationData with known Yvar cannot be mixed with observations without Yvar."
            )
        Yvar = None
        if self.Yvar is not None:
            Yvar = torch.cat([self.Yvar, other.Yvar.to(self.Yvar)], dim=0)
        return ObservationData(
            X=torch.cat([self.X, other.X.to(self.X)], dim=0),
            Y=torch.cat([self.Y, other.Y.to(self.Y)], dim=0),
            Yvar=Yvar,
            observed_mask=torch.cat(
                [self.observed_mask, other.observed_mask.to(self.observed_mask.device)],
                dim=0,
            ),
            failed_mask=torch.cat(
                [self.failed_mask, other.failed_mask.to(self.failed_mask.device)],
                dim=0,
            ),
            pending_mask=torch.cat(
                [self.pending_mask, other.pending_mask.to(self.pending_mask.device)],
                dim=0,
            ),
        )

    def resolve_pending(self, other: ObservationData) -> ObservationData:
        """Replace matching pending rows with completed observations before appending.

        Matching is performed in canonical model-input space. Each completed incoming
        row resolves at most one pending row, preserving duplicate experiments as
        distinct observations while preventing the normal ask->tell cycle from
        leaving stale ``X_pending`` entries behind.
        """

        torch = _torch()
        if int(self.X.shape[-1]) != int(other.X.shape[-1]):
            raise ValueError("ObservationData feature dimensions must match.")
        if int(self.Y.shape[-1]) != int(other.Y.shape[-1]):
            raise ValueError("ObservationData target dimensions must match.")
        if (self.Yvar is None) != (other.Yvar is None):
            raise ValueError(
                "ObservationData with known Yvar cannot be mixed with observations without Yvar."
            )
        if not bool(self.pending_mask.any()) or not bool(other.completed_mask.any()):
            return self.append(other)

        resolved_x = self.X.clone()
        resolved_y = self.Y.clone()
        resolved_yvar = None if self.Yvar is None else self.Yvar.clone()
        resolved_observed = self.observed_mask.clone()
        resolved_failed = self.failed_mask.clone()
        resolved_pending = self.pending_mask.clone()
        available_pending = self.pending_mask.clone()
        consumed = torch.zeros(
            int(other.X.shape[0]),
            dtype=torch.bool,
            device=other.X.device,
        )

        other_x = other.X.to(self.X)
        tolerance = (
            8.0 * torch.finfo(self.X.dtype).eps if torch.is_floating_point(self.X) else 0.0
        )

        completed_indices = torch.nonzero(other.completed_mask, as_tuple=False).flatten()
        for incoming_index_tensor in completed_indices:
            incoming_index = int(incoming_index_tensor.item())
            pending_indices = torch.nonzero(available_pending, as_tuple=False).flatten()
            if int(pending_indices.numel()) == 0:
                break

            pending_x = resolved_x[pending_indices]
            candidate_x = other_x[incoming_index]
            if torch.is_floating_point(resolved_x):
                matches = torch.isclose(
                    pending_x,
                    candidate_x.unsqueeze(0),
                    rtol=tolerance,
                    atol=tolerance,
                ).all(dim=-1)
            else:
                matches = (pending_x == candidate_x.unsqueeze(0)).all(dim=-1)
            if not bool(matches.any()):
                continue

            match_offset = int(torch.nonzero(matches, as_tuple=False)[0].item())
            existing_index = int(pending_indices[match_offset].item())
            resolved_x[existing_index] = candidate_x
            resolved_y[existing_index] = other.Y[incoming_index].to(resolved_y)
            if resolved_yvar is not None:
                resolved_yvar[existing_index] = other.Yvar[incoming_index].to(resolved_yvar)
            resolved_observed[existing_index] = other.observed_mask[incoming_index].to(
                resolved_observed.device
            )
            resolved_failed[existing_index] = other.failed_mask[incoming_index].to(
                resolved_failed.device
            )
            resolved_pending[existing_index] = other.pending_mask[incoming_index].to(
                resolved_pending.device
            )
            available_pending[existing_index] = False
            consumed[incoming_index] = True

        resolved = ObservationData(
            X=resolved_x,
            Y=resolved_y,
            Yvar=resolved_yvar,
            observed_mask=resolved_observed,
            failed_mask=resolved_failed,
            pending_mask=resolved_pending,
        )
        remaining = ~consumed
        if not bool(remaining.any()):
            return resolved
        return resolved.append(
            ObservationData(
                X=other.X[remaining],
                Y=other.Y[remaining],
                Yvar=None if other.Yvar is None else other.Yvar[remaining],
                observed_mask=other.observed_mask[remaining],
                failed_mask=other.failed_mask[remaining],
                pending_mask=other.pending_mask[remaining],
            )
        )

    def report(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.X.shape[0]),
            "n_completed": int(self.completed_mask.sum().item()),
            "n_success": int(self.success_mask.sum().item()),
            "n_failed": int(self.failed_mask.sum().item()),
            "n_pending": int(self.pending_mask.sum().item()),
            "known_observation_variance": self.Yvar is not None,
            "observed_per_output": [
                int(value)
                for value in self.observed_mask.sum(dim=0).detach().cpu().tolist()
            ],
        }


@dataclass
class ExperimentFailureConfig:
    """Configuration for learning and applying experiment success probability."""

    model_config: Any | None = None
    fit_config: Any | None = None
    min_success_probability: float = 0.5
    eta: float = 0.05
    reduce_q: FailureQReduction = "prod"

    def __post_init__(self) -> None:
        if self.fit_config is None:
            from ..configs import FitConfig

            self.fit_config = FitConfig()
        probability = float(self.min_success_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("min_success_probability must be between 0 and 1.")
        if float(self.eta) <= 0.0:
            raise ValueError("eta must be positive.")
        if self.reduce_q not in {"prod", "min", "mean"}:
            raise ValueError("reduce_q must be 'prod', 'min', or 'mean'.")


__all__ = [
    "ExperimentFailureConfig",
    "FailureQReduction",
    "ObservationData",
    "ObservationRowStatus",
]
