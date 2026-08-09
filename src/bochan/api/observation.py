"""Observation-state handling for physical experiment workflows.

The canonical contract distinguishes four independent facts:

- a target cell can be observed or not observed;
- a completed experiment can succeed or fail;
- a pending experiment has not produced an outcome yet;
- missing target values are never interpreted as experiment failures implicitly.

This module contains no runtime patching. It is consumed directly by the core
API, tabular conversion, and candidate construction paths.
"""

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
    """Canonical tensor representation of experiment observations.

    Args:
        X: Experiment conditions with shape ``[n, d]``.
        Y: Target matrix with shape ``[n, m]`` or ``[n]``. Unobserved target
            cells are represented as NaN in the canonical stored matrix.
        observed_mask: Optional cell-wise observation mask. When omitted, finite
            target cells are treated as observed.
        failed_mask: Optional row mask identifying completed failed experiments.
            Failed rows are excluded from every objective target while remaining
            useful for an experiment-success classifier.
        pending_mask: Optional row mask identifying experiments still running.
            Pending rows are excluded from objective and success-model fitting and
            are exposed through :attr:`pending_X` for acquisition ``X_pending``.
    """

    X: Any
    Y: Any
    observed_mask: Any | None = None
    failed_mask: Any | None = None
    pending_mask: Any | None = None

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

        self.X = X
        self.Y = canonical_y
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
    ) -> ObservationData:
        """Build observations from row status strings.

        Accepted row statuses are exactly ``success``, ``failed``, and
        ``pending``. Target-level missingness remains independent and is inferred
        from finite Y cells unless ``observed_mask`` is supplied.
        """

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
        )

    @property
    def completed_mask(self):
        """Rows whose experiment has finished, whether successful or failed."""

        return ~self.pending_mask

    @property
    def success_mask(self):
        """Completed rows that did not fail experimentally."""

        return self.completed_mask & ~self.failed_mask

    @property
    def objective_row_mask(self):
        """Successful rows containing at least one observed objective."""

        return self.success_mask & self.observed_mask.any(dim=-1)

    @property
    def pending_X(self):
        """Conditions currently running and therefore suitable for ``X_pending``."""

        return self.X[self.pending_mask]

    @property
    def completed_X(self):
        """All finished experiment conditions, including failed experiments."""

        return self.X[self.completed_mask]

    def objective_training_data(self) -> tuple[Any, Any]:
        """Return successful rows with at least one observed target cell."""

        mask = self.objective_row_mask
        if not bool(mask.any()):
            raise ValueError("No successful experiment contains an observed objective value.")
        return self.X[mask], self.Y[mask]

    def output_training_data(self, output_index: int) -> tuple[Any, Any]:
        """Return rows where one output was actually observed."""

        index = int(output_index)
        if index < 0 or index >= int(self.Y.shape[-1]):
            raise IndexError(
                f"output_index={index} is outside [0, {int(self.Y.shape[-1]) - 1}]."
            )
        mask = self.success_mask & self.observed_mask[:, index]
        if not bool(mask.any()):
            raise ValueError(f"Output {index} has no successful observed values.")
        return self.X[mask], self.Y[mask, index : index + 1]

    def success_training_data(self) -> tuple[Any, Any]:
        """Return completed experiments and binary success labels."""

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
        """Return a new observation table with rows from ``other`` appended."""

        torch = _torch()
        if int(self.X.shape[-1]) != int(other.X.shape[-1]):
            raise ValueError("ObservationData feature dimensions must match.")
        if int(self.Y.shape[-1]) != int(other.Y.shape[-1]):
            raise ValueError("ObservationData target dimensions must match.")
        return ObservationData(
            X=torch.cat([self.X, other.X.to(self.X)], dim=0),
            Y=torch.cat([self.Y, other.Y.to(self.Y)], dim=0),
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

    def report(self) -> dict[str, Any]:
        """Return serializable observation counts for diagnostics."""

        return {
            "n_rows": int(self.X.shape[0]),
            "n_completed": int(self.completed_mask.sum().item()),
            "n_success": int(self.success_mask.sum().item()),
            "n_failed": int(self.failed_mask.sum().item()),
            "n_pending": int(self.pending_mask.sum().item()),
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
            from .configs import FitConfig

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
