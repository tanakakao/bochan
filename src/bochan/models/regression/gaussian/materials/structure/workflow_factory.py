"""High-level workflow specification for material MLIP modeling and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from torch import Tensor

from ..common.relaxation import MaterialStructureRelaxer
from .factory import (
    create_relaxation_acquisition_selector,
    create_relaxation_ranker,
)
from .model_factory import (
    MaterialModelMode,
    MaterialModelSpec,
    create_material_model,
)
from .property_factory import MaterialQuantity
from .factory import MaterialMLIPBackend
from .relax_acquisition import MaterialRelaxationAcquisitionSelector
from .relax_rank import MaterialRelaxationRanker

MaterialWorkflowMode = Literal["model_only", "relax_rank", "relax_acquisition"]
SUPPORTED_MATERIAL_WORKFLOW_MODES: tuple[MaterialWorkflowMode, ...] = (
    "model_only",
    "relax_rank",
    "relax_acquisition",
)

_WORKFLOW_MODE_ALIASES = {
    "model_only": "model_only",
    "model-only": "model_only",
    "model": "model_only",
    "relax_rank": "relax_rank",
    "relax-rank": "relax_rank",
    "rank": "relax_rank",
    "relax_acquisition": "relax_acquisition",
    "relax-acquisition": "relax_acquisition",
    "acquisition": "relax_acquisition",
    "bo": "relax_acquisition",
    "al": "relax_acquisition",
}


def normalize_material_workflow_mode(mode: str) -> MaterialWorkflowMode:
    """Normalize a supported material workflow mode."""

    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("workflow_mode must be a non-empty string.")
    normalized = mode.strip().lower()
    resolved = _WORKFLOW_MODE_ALIASES.get(normalized)
    if resolved is None:
        supported = ", ".join(SUPPORTED_MATERIAL_WORKFLOW_MODES)
        raise ValueError(
            f"Unsupported material workflow mode {mode!r}. Supported modes: {supported}."
        )
    return cast(MaterialWorkflowMode, resolved)


@dataclass(frozen=True, slots=True)
class MaterialWorkflowSpec:
    """Serializable canonical identity for one material MLIP workflow."""

    backend: MaterialMLIPBackend | str
    quantity: MaterialQuantity | str
    model_mode: MaterialModelMode | str
    workflow_mode: MaterialWorkflowMode | str = "model_only"

    def __post_init__(self) -> None:
        model_spec = MaterialModelSpec(
            backend=self.backend,
            quantity=self.quantity,
            mode=self.model_mode,
        )
        object.__setattr__(self, "backend", model_spec.backend)
        object.__setattr__(self, "quantity", model_spec.quantity)
        object.__setattr__(self, "model_mode", model_spec.mode)
        object.__setattr__(
            self,
            "workflow_mode",
            normalize_material_workflow_mode(self.workflow_mode),
        )

    @property
    def model_spec(self) -> MaterialModelSpec:
        """Return the Phase 22 model identity represented by this workflow."""

        return MaterialModelSpec(
            backend=self.backend,
            quantity=self.quantity,
            mode=self.model_mode,
        )

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable canonical representation."""

        return {
            "backend": self.backend,
            "quantity": self.quantity,
            "model_mode": self.model_mode,
            "workflow_mode": self.workflow_mode,
        }


@dataclass(frozen=True, slots=True)
class MaterialWorkflow:
    """Construct models and optional relaxation-selection components consistently."""

    spec: MaterialWorkflowSpec
    ranker: MaterialRelaxationRanker | None = None
    acquisition_selector: MaterialRelaxationAcquisitionSelector | None = None

    def create_model(
        self,
        *,
        structures: Any,
        train_X: Tensor | None = None,
        train_Y: Tensor | None = None,
        train_Yvar: Tensor | None = None,
        **backend_kwargs: Any,
    ) -> Any:
        """Create the workflow's configured direct or residual model."""

        return create_material_model(
            self.spec.backend,
            self.spec.quantity,
            self.spec.model_mode,
            structures=structures,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            **backend_kwargs,
        )


def create_material_workflow(
    backend: str,
    quantity: str,
    model_mode: str,
    workflow_mode: str = "model_only",
    /,
    *,
    relaxer: MaterialStructureRelaxer | None = None,
    **relaxer_kwargs: Any,
) -> MaterialWorkflow:
    """Create a backend-consistent material modeling/relaxation workflow.

    ``model_only`` creates no relaxation component. ``relax_rank`` constructs
    the generic posterior ranker, while ``relax_acquisition`` constructs the
    generic BO/AL acquisition selector. Model construction remains deferred to
    :meth:`MaterialWorkflow.create_model` so relaxed structure banks can be used
    in the exact order produced by the relaxation workflow.
    """

    spec = MaterialWorkflowSpec(
        backend=backend,
        quantity=quantity,
        model_mode=model_mode,
        workflow_mode=workflow_mode,
    )

    if spec.workflow_mode == "model_only":
        if relaxer is not None or relaxer_kwargs:
            raise ValueError(
                "model_only workflows do not accept relaxer or relaxer keyword arguments."
            )
        return MaterialWorkflow(spec=spec)

    if spec.workflow_mode == "relax_rank":
        ranker = create_relaxation_ranker(
            spec.backend,
            relaxer=relaxer,
            **relaxer_kwargs,
        )
        return MaterialWorkflow(spec=spec, ranker=ranker)

    selector = create_relaxation_acquisition_selector(
        spec.backend,
        relaxer=relaxer,
        **relaxer_kwargs,
    )
    return MaterialWorkflow(spec=spec, acquisition_selector=selector)


__all__ = [
    "MaterialWorkflow",
    "MaterialWorkflowMode",
    "MaterialWorkflowSpec",
    "SUPPORTED_MATERIAL_WORKFLOW_MODES",
    "create_material_workflow",
    "normalize_material_workflow_mode",
]
