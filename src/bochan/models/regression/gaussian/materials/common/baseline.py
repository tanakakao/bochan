"""Contracts for deterministic pretrained material baselines.

These contracts describe *what* a pretrained baseline predicts independently of
how a third-party model is loaded. They are intentionally strict about physical
quantity, unit, and aggregation semantics so residual targets are not formed
from incompatible values such as total energy and energy per atom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BaselineAggregation = Literal["total", "per_atom", "intensive", "unspecified"]


def _nonempty_text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True)
class MaterialPropertyContract:
    """Identify one physical target quantity and its representation.

    Args:
        quantity: Stable physical-property identifier such as ``"energy"`` or
            ``"band_gap"``. Comparison is case-insensitive.
        unit: Unit string such as ``"eV"`` or ``"GPa"``. Unit comparison is
            intentionally exact and case-sensitive; no implicit conversion is
            performed.
        aggregation: Whether the value is a total, per-atom, intensive, or
            intentionally unspecified quantity.
    """

    quantity: str
    unit: str
    aggregation: BaselineAggregation = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", _nonempty_text(self.quantity, name="quantity"))
        object.__setattr__(self, "unit", _nonempty_text(self.unit, name="unit"))
        if self.aggregation not in {"total", "per_atom", "intensive", "unspecified"}:
            raise ValueError(
                "aggregation must be 'total', 'per_atom', 'intensive', or 'unspecified'."
            )

    def assert_compatible(self, other: MaterialPropertyContract) -> None:
        """Raise when two property contracts cannot be subtracted safely."""

        if not isinstance(other, MaterialPropertyContract):
            raise TypeError("other must be a MaterialPropertyContract.")
        if self.quantity.casefold() != other.quantity.casefold():
            raise ValueError(
                "Physical quantity mismatch: "
                f"baseline={self.quantity!r}, target={other.quantity!r}."
            )
        if self.unit != other.unit:
            raise ValueError(
                "Unit mismatch: "
                f"baseline={self.unit!r}, target={other.unit!r}. "
                "Bochan does not perform implicit unit conversion for residual baselines."
            )
        if (
            self.aggregation != "unspecified"
            and other.aggregation != "unspecified"
            and self.aggregation != other.aggregation
        ):
            raise ValueError(
                "Aggregation mismatch: "
                f"baseline={self.aggregation!r}, target={other.aggregation!r}."
            )

    def as_dict(self) -> dict[str, str]:
        """Return stable JSON-compatible metadata."""

        return {
            "quantity": self.quantity,
            "unit": self.unit,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True)
class MaterialBaselineSpec:
    """Describe one deterministic pretrained baseline assignment.

    Exactly one of ``output_name`` and ``output_index`` may be supplied. Neither
    is required for a scalar single-output residual model; future multi-baseline
    routing can use either selector without changing the physical-property
    contract.
    """

    family: str
    property: MaterialPropertyContract
    output_name: str | None = None
    output_index: int | None = None
    model_name: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _nonempty_text(self.family, name="family").casefold())
        if not isinstance(self.property, MaterialPropertyContract):
            raise TypeError("property must be a MaterialPropertyContract.")
        if self.output_name is not None:
            object.__setattr__(
                self,
                "output_name",
                _nonempty_text(self.output_name, name="output_name"),
            )
        if self.output_index is not None:
            if isinstance(self.output_index, bool) or not isinstance(self.output_index, int):
                raise TypeError("output_index must be an integer when provided.")
            if self.output_index < 0:
                raise ValueError("output_index must be non-negative when provided.")
        if self.output_name is not None and self.output_index is not None:
            raise ValueError("Specify at most one of output_name and output_index.")
        if self.model_name is not None:
            object.__setattr__(
                self,
                "model_name",
                _nonempty_text(self.model_name, name="model_name"),
            )
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")

    def assert_target_compatible(self, target: MaterialPropertyContract) -> None:
        """Validate that observed targets may be differenced from this baseline."""

        self.property.assert_compatible(target)

    def as_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible metadata for artifacts and APIs."""

        return {
            "family": self.family,
            "property": self.property.as_dict(),
            "output_name": self.output_name,
            "output_index": self.output_index,
            "model_name": self.model_name,
            "enabled": self.enabled,
        }


__all__ = [
    "BaselineAggregation",
    "MaterialBaselineSpec",
    "MaterialPropertyContract",
]
