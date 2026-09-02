"""Common contracts for material-aware Gaussian models."""

from .base import MaterialEncoder
from .fusion import (
    ConcatFusion,
    MaterialProcessFusion,
    build_material_process_fusion,
)

__all__ = [
    "ConcatFusion",
    "MaterialEncoder",
    "MaterialProcessFusion",
    "build_material_process_fusion",
]
