from __future__ import annotations

import pytest
import torch
from botorch.models.model import Model

from bochan.acquisition.multiclass.bayesian_optimization import (
    qMultiOutputMulticlassExpectedImprovement,
    qMultiOutputMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassProbabilityOfImprovement,
    qMultiOutputMulticlassUpperConfidenceBound,
)


class _DummyModel(Model):
    @property
    def num_outputs(self) -> int:
        return 2

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize(
    ("acqf_cls", "kwargs"),
    [
        (qMultiOutputMulticlassProbabilityOfFeasibility, {}),
        (qMultiOutputMulticlassExpectedImprovement, {"best_f": 0.0}),
        (qMultiOutputMulticlassProbabilityOfImprovement, {"best_f": 0.0}),
        (qMultiOutputMulticlassUpperConfidenceBound, {}),
    ],
)
def test_target_probability_acquisitions_default_to_class_zero(acqf_cls, kwargs):
    acqf = acqf_cls(model=_DummyModel(), **kwargs)

    assert acqf.target_class == 0
    assert acqf.output_target_classes is None


def test_explicit_output_target_classes_take_precedence():
    acqf = qMultiOutputMulticlassExpectedImprovement(
        model=_DummyModel(),
        output_target_classes=[1, 0],
        best_f=torch.zeros(2),
    )

    assert acqf.target_class is None
    assert acqf.output_target_classes == [1, 0]
