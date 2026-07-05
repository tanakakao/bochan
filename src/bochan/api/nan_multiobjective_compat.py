"""NaN-safe defaults for partially observed multi-objective targets.

Wide multi-task models intentionally retain the original ``train_Y`` matrix,
including NaNs that represent task values not observed at a given input. Model
fitting converts those cells to long-format observations, but automatic EHVI /
NEHVI defaults operate on the original matrix and therefore need separate
missing-value handling.
"""

from __future__ import annotations

from typing import Any


def _as_finite_objective_matrix(values: Any) -> Any:
    """Validate and return a floating objective matrix with shape ``[n, m]``."""

    import torch

    tensor = values if torch.is_tensor(values) else torch.as_tensor(values)
    if tensor.ndim != 2:
        raise ValueError(
            "Multi-objective values must have shape [n, m]. "
            f"Got shape={tuple(tensor.shape)}."
        )
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError("Multi-objective values must contain at least one row and output.")
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.get_default_dtype())
    return tensor


def make_nan_safe_default_ref_point(values: Any, margin: float = 0.1) -> Any:
    """Create a reference point from finite observations in each objective.

    Each objective may be observed at different input rows. This is sufficient
    for a conservative per-objective reference point, even when no complete
    multi-objective row exists.
    """

    import torch

    tensor = _as_finite_objective_matrix(values)
    finite = torch.isfinite(tensor)
    observed_per_output = finite.any(dim=-2)
    if not bool(observed_per_output.all()):
        missing = torch.where(~observed_per_output)[0].detach().cpu().tolist()
        raise ValueError(
            "Cannot infer a multi-objective reference point because some outputs "
            f"have no finite observations. Missing output indices: {missing}."
        )

    safe_values = torch.where(
        finite,
        tensor,
        torch.full_like(tensor, float("inf")),
    )
    return (safe_values.min(dim=-2).values - float(margin)).detach()


def complete_multiobjective_rows(values: Any) -> Any:
    """Return rows where every objective is finite.

    EHVI partitioning represents an observed Pareto set and therefore requires
    joint objective vectors measured at the same input. Partially observed rows
    cannot be silently imputed without changing the acquisition semantics.
    """

    import torch

    tensor = _as_finite_objective_matrix(values)
    complete = tensor[torch.isfinite(tensor).all(dim=-1)]
    if complete.shape[0] == 0:
        raise ValueError(
            "EHVI requires at least one training row where all objectives are "
            "finite. The wide multi-task model can fit partially observed train_Y, "
            "but an observed Pareto partition cannot be built when every row has "
            "a missing objective. Add at least one complete objective row, provide "
            "an explicit partitioning, or use NEHVI with the model posterior over "
            "X_baseline."
        )
    return complete


def make_nan_safe_partitioning(ref_point: Any, values: Any) -> Any:
    """Build EHVI partitioning from complete finite objective rows only."""

    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    complete = complete_multiobjective_rows(values)
    return FastNondominatedPartitioning(ref_point=ref_point, Y=complete)


def apply_nan_multiobjective_compat() -> None:
    """Install NaN-safe helpers in automatic defaults and engine imports."""

    from . import automatic_multiobjective, engine_defaults

    automatic_multiobjective._make_default_ref_point = make_nan_safe_default_ref_point
    automatic_multiobjective.make_default_ref_point = make_nan_safe_default_ref_point
    automatic_multiobjective._make_partitioning = make_nan_safe_partitioning
    automatic_multiobjective.make_partitioning = make_nan_safe_partitioning

    # engine_defaults imports these callables by name, so update its bound
    # references as well as the source module.
    engine_defaults.make_default_ref_point = make_nan_safe_default_ref_point
    engine_defaults.make_partitioning = make_nan_safe_partitioning


__all__ = [
    "apply_nan_multiobjective_compat",
    "complete_multiobjective_rows",
    "make_nan_safe_default_ref_point",
    "make_nan_safe_partitioning",
]
