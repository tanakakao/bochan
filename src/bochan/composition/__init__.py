"""Core composition domain API."""

from .descriptors import CompositionDescriptorCalculator
from .encoders import (
    ConcatFusion,
    CrabNetEncoder,
    MaterialEncoder,
    MaterialProcessFusion,
    build_material_process_fusion,
)
from .formula import (
    ATOMIC_NUMBERS,
    ATOMIC_WEIGHTS,
    element_order,
    format_formula,
    normalize_composition,
    parse_formula,
)
from .search_space import CompositionSearchSpace
from .simplex import SimplexTransform, TorchSimplexTransform, close_compositions, ilr_basis
from .transformer import CompositionTransformer

__all__ = [
    "ATOMIC_NUMBERS",
    "ATOMIC_WEIGHTS",
    "CompositionDescriptorCalculator",
    "CompositionSearchSpace",
    "CompositionTransformer",
    "ConcatFusion",
    "CrabNetEncoder",
    "MaterialEncoder",
    "MaterialProcessFusion",
    "SimplexTransform",
    "TorchSimplexTransform",
    "close_compositions",
    "build_material_process_fusion",
    "element_order",
    "format_formula",
    "ilr_basis",
    "normalize_composition",
    "parse_formula",
]
