from __future__ import annotations

from .hetero_single_output import (
    qHeteroMulticlassExpectedImprovement,
    qHeteroMulticlassProbabilityOfFeasibility,
    qHeteroMulticlassProbabilityOfImprovement,
    qHeteroMulticlassUpperConfidenceBound,
)
from .multi_output import _MultiOutputMulticlassBOBase


class qHeteroMultiOutputMulticlassProbabilityOfFeasibility(_MultiOutputMulticlassBOBase):
    single_output_acqf_cls = qHeteroMulticlassProbabilityOfFeasibility


class qHeteroMultiOutputMulticlassExpectedImprovement(_MultiOutputMulticlassBOBase):
    single_output_acqf_cls = qHeteroMulticlassExpectedImprovement


class qHeteroMultiOutputMulticlassProbabilityOfImprovement(_MultiOutputMulticlassBOBase):
    single_output_acqf_cls = qHeteroMulticlassProbabilityOfImprovement


class qHeteroMultiOutputMulticlassUpperConfidenceBound(_MultiOutputMulticlassBOBase):
    single_output_acqf_cls = qHeteroMulticlassUpperConfidenceBound


__all__ = [
    "qHeteroMultiOutputMulticlassProbabilityOfFeasibility",
    "qHeteroMultiOutputMulticlassExpectedImprovement",
    "qHeteroMultiOutputMulticlassProbabilityOfImprovement",
    "qHeteroMultiOutputMulticlassUpperConfidenceBound",
]
