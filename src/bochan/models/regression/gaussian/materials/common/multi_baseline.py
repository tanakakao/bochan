"""Multiple pretrained baseline contracts for independent material outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP

from .baseline import MaterialBaselineSpec
from .residual import ResidualMaterialGPModel


@dataclass(frozen=True)
class ResolvedBaselineAssignment:
    """Bind one enabled pretrained baseline to exactly one output."""

    output_index: int
    output_name: str
    spec: MaterialBaselineSpec

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible assignment metadata."""

        return {
            "output_index": self.output_index,
            "output_name": self.output_name,
            "baseline": self.spec.as_dict(),
        }


@dataclass(frozen=True)
class MaterialBaselinePlan:
    """Resolve multiple pretrained baselines against a stable output schema.

    Disabled baseline specifications are retained by callers if needed but are
    ignored by the resolved plan. Each enabled baseline must select exactly one
    output either by name or index, except that a selector may be omitted only
    for a single-output schema. Duplicate assignments are rejected.
    """

    output_names: tuple[str, ...]
    assignments: tuple[ResolvedBaselineAssignment, ...]

    @classmethod
    def resolve(
        cls,
        *,
        output_names: Sequence[str],
        baseline_specs: Sequence[MaterialBaselineSpec],
    ) -> MaterialBaselinePlan:
        """Resolve and validate baseline specifications for model-list outputs."""

        names = tuple(str(name) for name in output_names)
        if not names:
            raise ValueError("output_names must contain at least one output.")
        if any(not name.strip() for name in names):
            raise ValueError("output_names must contain non-empty names.")
        if len(set(names)) != len(names):
            raise ValueError("output_names must be unique for baseline routing.")

        resolved: list[ResolvedBaselineAssignment] = []
        occupied: set[int] = set()
        for spec in baseline_specs:
            if not isinstance(spec, MaterialBaselineSpec):
                raise TypeError("baseline_specs must contain MaterialBaselineSpec values.")
            if not spec.enabled:
                continue

            if spec.output_name is not None:
                if spec.output_name not in names:
                    raise ValueError(
                        f"Unknown baseline output_name={spec.output_name!r}; expected one of {names!r}."
                    )
                index = names.index(spec.output_name)
            elif spec.output_index is not None:
                index = int(spec.output_index)
                if index >= len(names):
                    raise ValueError(
                        f"Baseline output_index={index} is out of range for {len(names)} outputs."
                    )
            elif len(names) == 1:
                index = 0
            else:
                raise ValueError(
                    "Each enabled baseline must provide output_name or output_index for multi-output routing."
                )

            if index in occupied:
                raise ValueError(
                    f"Multiple enabled baselines target output {names[index]!r} (index {index})."
                )
            occupied.add(index)
            resolved.append(
                ResolvedBaselineAssignment(
                    output_index=index,
                    output_name=names[index],
                    spec=spec,
                )
            )

        resolved.sort(key=lambda assignment: assignment.output_index)
        return cls(output_names=names, assignments=tuple(resolved))

    @property
    def baseline_output_indices(self) -> tuple[int, ...]:
        """Return output indices that use deterministic pretrained baselines."""

        return tuple(assignment.output_index for assignment in self.assignments)

    @property
    def ordinary_output_indices(self) -> tuple[int, ...]:
        """Return output indices without pretrained baselines."""

        baseline_indices = set(self.baseline_output_indices)
        return tuple(index for index in range(len(self.output_names)) if index not in baseline_indices)

    def assignment_for_output(self, index: int) -> ResolvedBaselineAssignment | None:
        """Return the baseline assignment for one output index, when present."""

        for assignment in self.assignments:
            if assignment.output_index == index:
                return assignment
        return None

    def as_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible plan metadata."""

        return {
            "output_names": list(self.output_names),
            "baseline_output_indices": list(self.baseline_output_indices),
            "ordinary_output_indices": list(self.ordinary_output_indices),
            "assignments": [assignment.as_dict() for assignment in self.assignments],
        }


class MultipleBaselineModelListGP(ModelListGP):
    """Independent-output ModelListGP with validated multiple baseline metadata.

    Residual outputs must be ``ResidualMaterialGPModel`` instances carrying a
    ``baseline_spec`` consistent with the resolved plan. Outputs not assigned a
    pretrained baseline may be any BoTorch model accepted by ``ModelListGP``.
    """

    def __init__(
        self,
        *models: Model,
        output_names: Sequence[str],
        baseline_specs: Sequence[MaterialBaselineSpec],
    ) -> None:
        plan = MaterialBaselinePlan.resolve(
            output_names=output_names,
            baseline_specs=baseline_specs,
        )
        if len(models) != len(plan.output_names):
            raise ValueError(
                "The number of ModelList submodels must match output_names: "
                f"{len(models)} != {len(plan.output_names)}."
            )

        for index, model in enumerate(models):
            assignment = plan.assignment_for_output(index)
            if assignment is None:
                continue
            if not isinstance(model, ResidualMaterialGPModel):
                raise TypeError(
                    f"Output {plan.output_names[index]!r} has a baseline assignment but its "
                    "submodel is not ResidualMaterialGPModel."
                )
            if model.baseline_spec is None:
                raise ValueError(
                    f"Residual submodel for output {plan.output_names[index]!r} must carry baseline_spec."
                )
            if model.baseline_spec != assignment.spec:
                raise ValueError(
                    f"Residual submodel baseline_spec does not match the plan for output "
                    f"{plan.output_names[index]!r}."
                )

        super().__init__(*models)
        self.output_names = plan.output_names
        self.baseline_plan = plan

    @property
    def baseline_metadata(self) -> dict[str, object]:
        """Return resolved multiple-baseline metadata for APIs and artifacts."""

        return self.baseline_plan.as_dict()


__all__ = [
    "MaterialBaselinePlan",
    "MultipleBaselineModelListGP",
    "ResolvedBaselineAssignment",
]
