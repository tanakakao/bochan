import inspect
from types import SimpleNamespace

import torch

from bochan.acquisition.ordinal.bayesian_optimization import (
    _infer_multioutput_ordinal_train_y,
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


def test_infers_raw_train_y_from_wrapper_submodels():
    model = SimpleNamespace(
        models=[
            SimpleNamespace(train_targets=torch.tensor([0, 1, 2])),
            SimpleNamespace(train_targets=torch.tensor([2, 1, 0])),
        ]
    )

    train_y = _infer_multioutput_ordinal_train_y(model)

    assert torch.equal(
        train_y,
        torch.tensor([[0, 2], [1, 1], [2, 0]]),
    )


def test_infers_raw_train_y_from_correlated_model():
    model = SimpleNamespace(
        train_targets=torch.tensor([[0, 1], [1, 2], [2, 0]])
    )

    train_y = _infer_multioutput_ordinal_train_y(model)

    assert torch.equal(train_y, model.train_targets)


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


def test_ordinal_bo_wrappers_expose_automatic_context_parameters():
    ehvi = inspect.signature(qMultiOutputOrdinalExpectedHypervolumeImprovement)
    nehvi = inspect.signature(qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement)
    nparego = inspect.signature(qMultiOutputOrdinalNParEGO)

    assert {"ref_point", "partitioning"} <= set(ehvi.parameters)
    assert {"ref_point", "X_baseline"} <= set(nehvi.parameters)
    assert {"ref_point", "X_baseline", "best_f", "train_Y"} <= set(
        nparego.parameters
    )
