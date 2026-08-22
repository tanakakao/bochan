"""Material encoders and material/process fusion modules."""

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
