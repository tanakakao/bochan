"""Phase 4 compatibility tests for shared structure feature infrastructure."""

from bochan.models.regression.gaussian.deep.structure import (
    _StructureGPFeatureExtractor as OldStructureGPFeatureExtractor,
)
from bochan.models.regression.gaussian.deep.structure import (
    _resolve_structure_input_transform as old_resolve_structure_input_transform,
)
from bochan.models.regression.gaussian.deep.structure import (
    _validate_structure_bank as old_validate_structure_bank,
)
from bochan.models.regression.gaussian.deep.structure import (
    _validate_structure_model_inputs as old_validate_structure_model_inputs,
)
from bochan.models.regression.gaussian.materials.structure.common import (
    _StructureGPFeatureExtractor,
    _resolve_structure_input_transform,
    _validate_structure_bank,
    _validate_structure_model_inputs,
)


def test_structure_common_aliases_preserve_exact_implementation_objects() -> None:
    assert _StructureGPFeatureExtractor is OldStructureGPFeatureExtractor
    assert _resolve_structure_input_transform is old_resolve_structure_input_transform
    assert _validate_structure_bank is old_validate_structure_bank
    assert _validate_structure_model_inputs is old_validate_structure_model_inputs


def test_structure_feature_extractor_keeps_historical_module_path() -> None:
    assert (
        _StructureGPFeatureExtractor.__module__
        == "bochan.models.regression.gaussian.deep.structure"
    )
