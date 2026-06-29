from __future__ import annotations

import inspect

from bochan.models.classification.multiclass.high_dim import (
    PCAMulticlassClassificationGPModel,
    PCAMulticlassClassificationMixedGPModel,
    REMBOMulticlassClassificationGPModel,
    REMBOMulticlassClassificationMixedGPModel,
)


def test_projected_multiclass_models_accept_beta() -> None:
    for model_cls in (
        PCAMulticlassClassificationGPModel,
        PCAMulticlassClassificationMixedGPModel,
        REMBOMulticlassClassificationGPModel,
        REMBOMulticlassClassificationMixedGPModel,
    ):
        parameter = inspect.signature(model_cls.make_mll).parameters["beta"]
        assert parameter.default == 1.0
