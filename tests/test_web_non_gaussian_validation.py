from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from bochan.serving.webapp.non_gaussian_validation import (
    non_gaussian_family,
    validate_non_gaussian_target_frame,
)
from bochan.serving.webapp.workflows import _validate_non_gaussian_web_targets


@pytest.mark.parametrize(
    ("model_type", "family"),
    [
        ("beta_base", "beta"),
        ("beta_multitask", "beta"),
        ("gamma_wide_multitask", "gamma"),
        ("poisson_deepkernel", "poisson"),
        ("negative_binomial_multitask", "negative_binomial"),
        ("base", None),
    ],
)
def test_non_gaussian_family(model_type: str, family: str | None) -> None:
    assert non_gaussian_family(model_type) == family


@pytest.mark.parametrize(
    ("model_type", "values"),
    [
        ("beta_base", [0.1, 0.5, 0.9]),
        ("gamma_multitask", [0.1, 1.0, 4.0]),
        ("poisson_base", [0, 1, 10]),
        ("negative_binomial_multitask", [0, 2, 15]),
    ],
)
def test_valid_non_gaussian_targets_pass(
    model_type: str,
    values: list[float],
) -> None:
    data = pd.DataFrame({"target": values})
    validate_non_gaussian_target_frame(data, ["target"], model_type)


@pytest.mark.parametrize(
    ("model_type", "values", "message"),
    [
        ("beta_base", [0.0, 0.5], r"0 < y < 1"),
        ("beta_multitask", [0.2, 1.0], r"0 < y < 1"),
        ("gamma_base", [0.0, 1.0], r"y > 0"),
        ("gamma_multitask", [-0.1, 1.0], r"y > 0"),
        ("poisson_base", [-1.0, 2.0], "non-negative integer counts"),
        ("poisson_multitask", [1.5, 2.0], "non-negative integer counts"),
        (
            "negative_binomial_base",
            [-1.0, 2.0],
            "non-negative integer counts",
        ),
        (
            "negative_binomial_multitask",
            [1.25, 2.0],
            "non-negative integer counts",
        ),
    ],
)
def test_invalid_non_gaussian_targets_fail_before_fit(
    model_type: str,
    values: list[float],
    message: str,
) -> None:
    data = pd.DataFrame({"target": values}, index=[10, 11])
    with pytest.raises(ValueError, match=message) as error:
        validate_non_gaussian_target_frame(data, ["target"], model_type)
    assert "target:" in str(error.value)
    assert "Invalid rows:" in str(error.value)


@pytest.mark.parametrize(
    "model_type",
    ["beta_base", "gamma_base", "poisson_base", "negative_binomial_base"],
)
def test_non_numeric_and_infinite_targets_are_rejected(model_type: str) -> None:
    with pytest.raises(ValueError, match="numeric target values"):
        validate_non_gaussian_target_frame(
            pd.DataFrame({"target": [1.0, "bad"]}),
            ["target"],
            model_type,
        )
    with pytest.raises(ValueError, match="finite target values"):
        validate_non_gaussian_target_frame(
            pd.DataFrame({"target": [1.0, float("inf")]}),
            ["target"],
            model_type,
        )


def test_missing_values_are_left_to_target_missing_policy() -> None:
    validate_non_gaussian_target_frame(
        pd.DataFrame({"target": [0.2, None, 0.8]}),
        ["target"],
        "beta_multitask",
    )


def test_gaussian_models_are_not_domain_restricted() -> None:
    validate_non_gaussian_target_frame(
        pd.DataFrame({"target": [-3.5, 0.0, 8.2]}),
        ["target"],
        "base",
    )


def test_web_workflow_wrapper_validates_store_data() -> None:
    class Store:
        def get(self, dataset_id: str) -> SimpleNamespace:
            assert dataset_id == "dataset-1"
            return SimpleNamespace(data=pd.DataFrame({"yield": [0.2, 1.0]}))

    request = SimpleNamespace(
        dataset_id="dataset-1",
        target_column="yield",
        target_columns=["yield"],
        direction="maximize",
        directions={},
        model_type="beta_base",
    )

    with pytest.raises(ValueError, match=r"0 < y < 1"):
        _validate_non_gaussian_web_targets(request, Store())
