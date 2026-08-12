"""Acquisition construction and routing services for the high-level API."""

from .service import (
    build_acquisition,
    build_objective,
    infer_bundle_multi_output,
    is_nsgaii_strategy,
    resolve_acquisition,
    resolve_acquisition_class,
    resolve_input_perturbation_objective,
)

__all__ = [
    "build_acquisition",
    "build_objective",
    "infer_bundle_multi_output",
    "is_nsgaii_strategy",
    "resolve_acquisition",
    "resolve_acquisition_class",
    "resolve_input_perturbation_objective",
]
