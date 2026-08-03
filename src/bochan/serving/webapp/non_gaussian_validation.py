"""Pre-fit target-domain validation for non-Gaussian Web models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


_NON_GAUSSIAN_FAMILIES = (
    "negative_binomial",
    "poisson",
    "gamma",
    "beta",
)


def non_gaussian_family(model_type: str) -> str | None:
    """Return the likelihood family encoded in a public or internal model key."""

    normalized = str(model_type).strip().lower()
    for family in _NON_GAUSSIAN_FAMILIES:
        if normalized == family or normalized.startswith(f"{family}_"):
            return family
    return None


def _row_suffix(series: Any, invalid: Any) -> str:
    """Format a compact sample of invalid DataFrame index values."""

    rows = [str(value) for value in series.index[invalid][:5].tolist()]
    if not rows:
        return ""
    suffix = ", ..." if int(invalid.sum()) > len(rows) else ""
    return f" Invalid rows: {', '.join(rows)}{suffix}."


def validate_non_gaussian_target_frame(
    data: Any,
    target_columns: Sequence[str],
    model_type: str,
) -> None:
    """Validate all observed Web targets before constructing a surrogate model.

    Beta targets must lie strictly inside ``(0, 1)``. Gamma targets must be
    strictly positive. Poisson and Negative Binomial targets must be finite,
    non-negative integer counts. Missing values are ignored here and remain the
    responsibility of the configured target-missing policy.
    """

    family = non_gaussian_family(model_type)
    if family is None:
        return

    import numpy as np
    import pandas as pd

    for target in target_columns:
        if target not in data.columns:
            raise ValueError(f"Unknown target column: {target}")

        series = data[target]
        observed = series.notna()
        numeric = pd.to_numeric(series, errors="coerce")
        non_numeric = observed & numeric.isna()
        if bool(non_numeric.any()):
            raise ValueError(
                f"{target}: {family} regression requires numeric target values."
                f"{_row_suffix(series, non_numeric)}"
            )

        observed_numeric = numeric[observed]
        if observed_numeric.empty:
            raise ValueError(
                f"{target}: {family} regression requires at least one observed target value."
            )

        finite = pd.Series(
            np.isfinite(observed_numeric.to_numpy(dtype=float)),
            index=observed_numeric.index,
        )
        non_finite = observed.copy()
        non_finite.loc[:] = False
        non_finite.loc[finite.index] = ~finite
        if bool(non_finite.any()):
            raise ValueError(
                f"{target}: {family} regression requires finite target values."
                f"{_row_suffix(series, non_finite)}"
            )

        values = observed_numeric.to_numpy(dtype=float)
        if family == "beta":
            invalid_values = (values <= 0.0) | (values >= 1.0)
            requirement = "Beta regression requires every observed target to satisfy 0 < y < 1."
        elif family == "gamma":
            invalid_values = values <= 0.0
            requirement = "Gamma regression requires every observed target to satisfy y > 0."
        else:
            invalid_values = values < 0.0
            if not bool(invalid_values.any()):
                invalid_values = ~np.isclose(
                    values,
                    np.round(values),
                    rtol=0.0,
                    atol=1e-8,
                )
            label = "Poisson" if family == "poisson" else "Negative Binomial"
            requirement = (
                f"{label} regression requires finite non-negative integer counts."
            )

        if bool(invalid_values.any()):
            invalid = observed.copy()
            invalid.loc[:] = False
            invalid.loc[observed_numeric.index] = invalid_values
            raise ValueError(
                f"{target}: {requirement}{_row_suffix(series, invalid)}"
            )


__all__ = [
    "non_gaussian_family",
    "validate_non_gaussian_target_frame",
]
