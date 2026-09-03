from __future__ import annotations

import importlib
import pickle

import pytest

from bochan.models.regression.gaussian.materials.common.compatibility import (
    LEGACY_MATERIAL_MODEL_PATHS,
    MaterialCompatibilityPath,
    canonical_material_model_paths,
    legacy_material_model_paths,
)


def _resolve(path: str):
    module_name, attribute = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def test_compatibility_path_requires_distinct_module_attribute_paths() -> None:
    with pytest.raises(ValueError, match="module:attribute"):
        MaterialCompatibilityPath("invalid", "module:Class")
    with pytest.raises(ValueError, match="must differ"):
        MaterialCompatibilityPath("module:Class", "module:Class")


def test_compatibility_path_lists_are_stable_and_unique() -> None:
    legacy = legacy_material_model_paths()
    canonical = canonical_material_model_paths()

    assert len(legacy) == len(canonical) == len(LEGACY_MATERIAL_MODEL_PATHS)
    assert len(set(legacy)) == len(legacy)
    assert len(set(canonical)) == len(canonical)
    assert all(item.serialization_protected for item in LEGACY_MATERIAL_MODEL_PATHS)


@pytest.mark.parametrize("item", LEGACY_MATERIAL_MODEL_PATHS)
def test_canonical_and_legacy_paths_resolve_same_class_object(
    item: MaterialCompatibilityPath,
) -> None:
    legacy_class = _resolve(item.legacy)
    canonical_class = _resolve(item.canonical)

    assert canonical_class is legacy_class


@pytest.mark.parametrize("item", LEGACY_MATERIAL_MODEL_PATHS)
def test_pickle_class_round_trip_preserves_historical_module_resolution(
    item: MaterialCompatibilityPath,
) -> None:
    legacy_class = _resolve(item.legacy)
    legacy_module, legacy_name = item.legacy.split(":", maxsplit=1)

    assert legacy_class.__module__ == legacy_module
    assert legacy_class.__name__ == legacy_name
    assert pickle.loads(pickle.dumps(legacy_class)) is legacy_class
