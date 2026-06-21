from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.ordinal import (
    MultiTaskOrdinalGPModel,
    MultiTaskOrdinalMixedGPModel,
)
from bochan.models.ordinal.base import (
    MultiTaskOrdinalGPModel as BaseMultiTaskOrdinalGPModel,
)
from bochan.models.ordinal.base import (
    MultiTaskOrdinalMixedGPModel as BaseMultiTaskOrdinalMixedGPModel,
)


def test_ordinal_multitask_models_are_exported_at_package_level() -> None:
    assert MultiTaskOrdinalGPModel is BaseMultiTaskOrdinalGPModel
    assert MultiTaskOrdinalMixedGPModel is BaseMultiTaskOrdinalMixedGPModel


def test_mixed_ordinal_multitask_registry_resolves_public_model() -> None:
    assert (
        MODEL_REGISTRY["mixed"]["ordinal"]["multitask"]
        is MultiTaskOrdinalMixedGPModel
    )
