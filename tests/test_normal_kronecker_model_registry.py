from __future__ import annotations

from botorch.models.multitask import KroneckerMultiTaskGP

from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationGPModel,
)
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationGPModel,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel


def test_normal_regression_kronecker_registry() -> None:
    assert (
        MODEL_REGISTRY["normal"]["regression"]["kronecker"]
        is KroneckerMultiTaskGP
    )
    assert (
        MODEL_REGISTRY["normal"]["multi_objective"]["kronecker"]
        is KroneckerMultiTaskGP
    )


def test_normal_binary_kronecker_registry() -> None:
    assert (
        MODEL_REGISTRY["normal"]["binary"]["kronecker"]
        is KroneckerMultiTaskBinaryClassificationGPModel
    )


def test_normal_ordinal_kronecker_registry() -> None:
    assert (
        MODEL_REGISTRY["normal"]["ordinal"]["kronecker"]
        is KroneckerMultiTaskOrdinalGPModel
    )


def test_normal_multiclass_kronecker_registry() -> None:
    assert (
        MODEL_REGISTRY["normal"]["multiclass"]["kronecker"]
        is KroneckerMultiTaskMulticlassClassificationGPModel
    )
