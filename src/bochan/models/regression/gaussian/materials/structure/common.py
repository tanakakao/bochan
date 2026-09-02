"""Canonical structure-feature infrastructure during staged migration.

These aliases intentionally preserve the historical implementation objects and
module paths so structure caches, state-dict behavior, and pickle compatibility
remain unchanged while callers gain a neutral structure namespace.
"""

from ...deep.structure import (
    _StructureGPFeatureExtractor,
    _resolve_structure_input_transform,
    _validate_structure_bank,
    _validate_structure_model_inputs,
)

__all__ = [
    "_StructureGPFeatureExtractor",
    "_resolve_structure_input_transform",
    "_validate_structure_bank",
    "_validate_structure_model_inputs",
]
