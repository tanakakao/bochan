from __future__ import annotations

import importlib
import subprocess
import sys


def test_materials_package_exposes_neutral_contracts() -> None:
    materials = importlib.import_module("bochan.models.regression.gaussian.materials")
    common = importlib.import_module("bochan.models.regression.gaussian.materials.common")

    assert materials.MaterialEncoder is common.MaterialEncoder
    assert materials.MaterialProcessFusion is common.MaterialProcessFusion
    assert materials.ConcatFusion is common.ConcatFusion
    assert materials.build_material_process_fusion is common.build_material_process_fusion


def test_legacy_composition_contracts_reexport_canonical_objects() -> None:
    from bochan.composition.encoders.base import MaterialEncoder as LegacyMaterialEncoder
    from bochan.composition.encoders.fusion import (
        ConcatFusion as LegacyConcatFusion,
        MaterialProcessFusion as LegacyMaterialProcessFusion,
        build_material_process_fusion as legacy_build_material_process_fusion,
    )
    from bochan.models.regression.gaussian.materials.common import (
        ConcatFusion,
        MaterialEncoder,
        MaterialProcessFusion,
        build_material_process_fusion,
    )

    assert LegacyMaterialEncoder is MaterialEncoder
    assert LegacyConcatFusion is ConcatFusion
    assert LegacyMaterialProcessFusion is MaterialProcessFusion
    assert legacy_build_material_process_fusion is build_material_process_fusion


def test_domain_namespaces_are_importable_without_concrete_models() -> None:
    composition = importlib.import_module(
        "bochan.models.regression.gaussian.materials.composition"
    )
    structure = importlib.import_module(
        "bochan.models.regression.gaussian.materials.structure"
    )

    assert composition.__all__ == []
    assert structure.__all__ == []


def test_canonical_material_import_does_not_load_concrete_encoder_modules() -> None:
    code = """
import sys
import bochan.models.regression.gaussian.materials  # noqa: F401

forbidden = (
    'bochan.composition',
    'bochan.composition.encoders.crabnet',
    'bochan.composition.encoders.roost',
    'bochan.composition.encoders.alignn',
    'bochan.composition.encoders.chgnet',
    'bochan.composition.encoders.m3gnet',
    'bochan.composition.encoders.mace',
)
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit('canonical materials import loaded concrete encoder modules: ' + ', '.join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
