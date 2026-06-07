from __future__ import annotations

from .hetero_single_output import (
    qHeteroMulticlassBoundaryVarianceAcquisition,
    qHeteroMulticlassClassEntropyAcquisition,
    qHeteroMulticlassICUAcquisition,
    qHeteroMulticlassJointLatentStraddleAcquisition,
    qHeteroMulticlassLatentStraddleAcquisition,
    qHeteroMulticlassLevelSetUncertainty,
    qHeteroMulticlassProbabilityOfExceedance,
)
from .multi_output import _MultiOutputMulticlassLevelSetBase


class qHeteroMultiOutputMulticlassLatentStraddleAcquisition(_MultiOutputMulticlassLevelSetBase):
    single_output_acqf_cls = qHeteroMulticlassLatentStraddleAcquisition


class qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition(_MultiOutputMulticlassLevelSetBase):
    single_output_acqf_cls = qHeteroMulticlassJointLatentStraddleAcquisition


class qHeteroMultiOutputMulticlassICUAcquisition(_MultiOutputMulticlassLevelSetBase):
    single_output_acqf_cls = qHeteroMulticlassICUAcquisition


class qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition(_MultiOutputMulticlassLevelSetBase):
    single_output_acqf_cls = qHeteroMulticlassBoundaryVarianceAcquisition


class qHeteroMultiOutputMulticlassClassEntropyAcquisition(_MultiOutputMulticlassLevelSetBase):
    single_output_acqf_cls = qHeteroMulticlassClassEntropyAcquisition


class qHeteroMultiOutputMulticlassProbabilityOfExceedance(_MultiOutputMulticlassLevelSetBase):
    single_output_acqf_cls = qHeteroMulticlassProbabilityOfExceedance


class qHeteroMultiOutputMulticlassLevelSetUncertainty(_MultiOutputMulticlassLevelSetBase):
    single_output_acqf_cls = qHeteroMulticlassLevelSetUncertainty


__all__ = [
    "qHeteroMultiOutputMulticlassLatentStraddleAcquisition",
    "qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition",
    "qHeteroMultiOutputMulticlassICUAcquisition",
    "qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition",
    "qHeteroMultiOutputMulticlassClassEntropyAcquisition",
    "qHeteroMultiOutputMulticlassProbabilityOfExceedance",
    "qHeteroMultiOutputMulticlassLevelSetUncertainty",
]
