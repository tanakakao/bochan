from __future__ import annotations

import inspect

import pytest

from bochan.models.classification.multiclass import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
)
from bochan.models.classification.multiclass.deep import (
    DeepKernelMulticlassClassificationGPModel,
    DeepKernelMixedMulticlassClassificationGPModel,
    MulticlassDeepGPModel,
    MulticlassMixedDeepGPModel,
)
from bochan.models.classification.multiclass.high_dim import (
    PCAMulticlassClassificationGPModel,
    PCAMulticlassClassificationMixedGPModel,
    REMBOMulticlassClassificationGPModel,
    REMBOMulticlassClassificationMixedGPModel,
)


@pytest.mark.parametrize(
    "model_cls",
    [
        MulticlassClassificationGPModel,
        MulticlassClassificationMixedGPModel,
        DeepKernelMulticlassClassificationGPModel,
        DeepKernelMixedMulticlassClassificationGPModel,
        MulticlassDeepGPModel,
        MulticlassMixedDeepGPModel,
        PCAMulticlassClassificationGPModel,
        PCAMulticlassClassificationMixedGPModel,
        REMBOMulticlassClassificationGPModel,
        REMBOMulticlassClassificationMixedGPModel,
    ],
)
def test_multiclass_make_mll_accepts_beta(model_cls) -> None:
    parameter = inspect.signature(model_cls.make_mll).parameters["beta"]
    assert parameter.default == 1.0
