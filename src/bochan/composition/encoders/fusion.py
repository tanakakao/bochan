"""Compatibility imports for canonical material/process fusion contracts."""

from bochan.models.regression.gaussian.materials.common.fusion import (
    ConcatFusion,
    MaterialProcessFusion,
    build_material_process_fusion,
)

__all__ = [
    "ConcatFusion",
    "MaterialProcessFusion",
    "build_material_process_fusion",
]
