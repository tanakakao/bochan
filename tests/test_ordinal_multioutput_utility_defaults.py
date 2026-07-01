from types import SimpleNamespace

import torch

from bochan.acquisition.ordinal.bayesian_optimization import (
    infer_multioutput_ordinal_utility_values,
    qMultiOutputOrdinalExpectedHypervolumeImprovement,
    qMultiOutputOrdinalNParEGO,
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
)
from bochan.api.acquisition_registry import resolve_acqf_cls


def test_infers_shared_utilities_from_wrapper_submodel_labels():
    model = SimpleNamespace(
        models=[
            SimpleNamespace(train_targets=torch.tensor([0, 1, 2])),
            SimpleNamespace(train_targets=torch.tensor([0, 2, 1])),
        ]
    )

    utility_values = infer_multioutput_ordinal_utility_values(model)

    assert torch.equal(utility_values, torch.tensor([0.0, 1.0, 2.0]))


def test_infers_per_output_utilities_for_different_class_counts():
    model = SimpleNamespace(
        models=[
            SimpleNamespace(train_targets=torch.tensor([0, 1, 2])),
            SimpleNamespace(train_targets=torch.tensor([0, 1, 2, 3])),
        ]
    )

    utility_values = infer_multioutput_ordinal_utility_values(model)

    assert isinstance(utility_values, list)
    assert [value.numel() for value in utility_values] == [3, 4]


def test_registry_resolves_ordinal_bo_aliases_to_defaulting_constructors():
    expected = {
        "ehvi": qMultiOutputOrdinalExpectedHypervolumeImprovement,
        "nehvi": qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
        "nparego": qMultiOutputOrdinalNParEGO,
    }

    for alias, constructor in expected.items():
        resolved = resolve_acqf_cls(
            alias,
            task_type="ordinal",
            model_type="base",
            multi_output=True,
        )
        assert resolved is constructor
