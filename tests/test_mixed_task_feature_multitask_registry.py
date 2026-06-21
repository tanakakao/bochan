from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.classification.binary.base import MultiTaskBinaryClassificationMixedGPModel
from bochan.models.classification.multiclass.base import MultiTaskMulticlassClassificationMixedGPModel
from bochan.models.ordinal.base import MultiTaskOrdinalMixedGPModel
from bochan.models.regression.gaussian import MixedMultiTaskGP


def test_mixed_multitask_registry_entries():
    assert MODEL_REGISTRY["mixed"]["regression"]["multitask"] is MixedMultiTaskGP
    assert MODEL_REGISTRY["mixed"]["multi_objective"]["multitask"] is MixedMultiTaskGP
    assert MODEL_REGISTRY["mixed"]["binary"]["multitask"] is MultiTaskBinaryClassificationMixedGPModel
    assert MODEL_REGISTRY["mixed"]["multiclass"]["multitask"] is MultiTaskMulticlassClassificationMixedGPModel
    assert MODEL_REGISTRY["mixed"]["ordinal"]["multitask"] is MultiTaskOrdinalMixedGPModel
