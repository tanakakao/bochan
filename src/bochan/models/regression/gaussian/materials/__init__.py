"""Material-aware Gaussian model infrastructure.

This package owns domain-neutral material contracts and the composition and
structure model namespaces. Concrete model wrappers remain in ``gaussian.deep``
during the staged migration.
"""

from .common import (
    ConcatFusion,
    MaterialEncoder,
    MaterialProcessFusion,
    build_material_process_fusion,
)

__all__ = [
    "ConcatFusion",
    "MaterialEncoder",
    "MaterialProcessFusion",
    "build_material_process_fusion",
]
