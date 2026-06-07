from __future__ import annotations

from .hetero_single_output import (
    qHeteroMulticlassBALD,
    qHeteroMulticlassIntegratedPosteriorVarianceProxy,
    qHeteroMulticlassMarginUncertainty,
    qHeteroMulticlassPredictiveEntropy,
    qHeteroMulticlassProbabilityVariance,
)
from .multi_output import _MultiOutputMulticlassAcqBase


class qHeteroMultiOutputMulticlassPredictiveEntropy(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qHeteroMulticlassPredictiveEntropy


class qHeteroMultiOutputMulticlassProbabilityVariance(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qHeteroMulticlassProbabilityVariance


class qHeteroMultiOutputMulticlassMarginUncertainty(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qHeteroMulticlassMarginUncertainty


class qHeteroMultiOutputMulticlassBALD(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qHeteroMulticlassBALD


class qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy(_MultiOutputMulticlassAcqBase):
    single_output_acqf_cls = qHeteroMulticlassIntegratedPosteriorVarianceProxy


__all__ = [
    "qHeteroMultiOutputMulticlassPredictiveEntropy",
    "qHeteroMultiOutputMulticlassProbabilityVariance",
    "qHeteroMultiOutputMulticlassMarginUncertainty",
    "qHeteroMultiOutputMulticlassBALD",
    "qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
]
