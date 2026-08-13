"""Named linear-constraint utilities for tabular optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from bochan.api import OptimizeConfig


def merge_named_linear_constraints(
    opt_config: OptimizeConfig | Mapping[str, Any] | None,
    constraints: Sequence[tuple[Any, ...]],
) -> OptimizeConfig | Mapping[str, Any] | None:
    """Merge named linear constraints into mapping or dataclass optimize configs."""

    if not constraints:
        return opt_config
    if opt_config is None:
        return {"constraints": list(constraints)}
    if isinstance(opt_config, Mapping):
        payload = dict(opt_config)
        existing = list(payload.get("constraints") or ())
        payload["constraints"] = [*existing, *constraints]
        return payload

    equalities = list(opt_config.equality_constraints or ())
    inequalities = list(opt_config.inequality_constraints or ())
    for columns, coefficients, operator, rhs in constraints:
        if operator == "=":
            equalities.append((columns, coefficients, rhs))
        elif operator == ">=":
            inequalities.append((columns, coefficients, rhs))
        else:
            inequalities.append(
                (columns, [-float(value) for value in coefficients], -float(rhs))
            )
    return replace(
        opt_config,
        equality_constraints=equalities,
        inequality_constraints=inequalities,
    )


__all__ = ["merge_named_linear_constraints"]
