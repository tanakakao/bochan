from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationMixedGPModel,
)
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationMixedGPModel,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalMixedGPModel
from bochan.models.regression.gaussian import GaussianMixedKroneckerMultiTaskGP


def test_mixed_kronecker_models_are_registered():
    assert MODEL_REGISTRY["mixed"]["regression"]["kronecker"] is GaussianMixedKroneckerMultiTaskGP
    assert MODEL_REGISTRY["mixed"]["multi_objective"]["kronecker"] is GaussianMixedKroneckerMultiTaskGP
    assert (
        MODEL_REGISTRY["mixed"]["binary"]["kronecker"]
        is KroneckerMultiTaskBinaryClassificationMixedGPModel
    )
    assert (
        MODEL_REGISTRY["mixed"]["multiclass"]["kronecker"]
        is KroneckerMultiTaskMulticlassClassificationMixedGPModel
    )
    assert (
        MODEL_REGISTRY["mixed"]["ordinal"]["kronecker"]
        is KroneckerMultiTaskOrdinalMixedGPModel
    )
