from __future__ import annotations

import pytest
from botorch.acquisition.logei import (
    qLogNoisyExpectedImprovement,
    qLogProbabilityOfFeasibility,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO

from bochan.acquisition.regression.bayesian_optimization import (
    qMultiOutputRegressionNParEGO,
)
from bochan.api.registry.acquisition import available_acqf_names, resolve_acqf_cls


def test_log_nei_alias_resolves_to_botorch() -> None:
    resolved = resolve_acqf_cls(
        "lognei",
        task_type="regression",
        multi_output=False,
    )

    assert resolved is qLogNoisyExpectedImprovement


def test_log_pof_alias_resolves_to_botorch() -> None:
    resolved = resolve_acqf_cls(
        "logpof",
        task_type="regression",
        multi_output=True,
    )

    assert resolved is qLogProbabilityOfFeasibility


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("logehvi", qLogExpectedHypervolumeImprovement),
        ("lognehvi", qLogNoisyExpectedHypervolumeImprovement),
        ("lognparego", qLogNParEGO),
    ],
)
def test_multiobjective_log_aliases_resolve_to_botorch(name, expected) -> None:
    resolved = resolve_acqf_cls(
        name,
        task_type="regression",
        multi_output=True,
    )

    assert resolved is expected


def test_regular_regression_nparego_short_name_resolves_to_native_class() -> None:
    resolved = resolve_acqf_cls(
        "nparego",
        task_type="regression",
        multi_output=True,
    )

    assert resolved is qMultiOutputRegressionNParEGO


def test_regular_nparego_requires_multi_output() -> None:
    with pytest.raises(ValueError, match="multi-output"):
        resolve_acqf_cls(
            "nparego",
            task_type="regression",
            multi_output=False,
        )


@pytest.mark.parametrize(
    "name",
    ["logehvi", "lognehvi", "lognparego"],
)
def test_multiobjective_log_aliases_require_multi_output(name) -> None:
    with pytest.raises(ValueError, match="multi-output"):
        resolve_acqf_cls(
            name,
            task_type="regression",
            multi_output=False,
        )


@pytest.mark.parametrize(
    "name",
    ["lognei", "logpof", "logehvi", "lognehvi", "lognparego"],
)
def test_log_short_aliases_reject_classification_task(name) -> None:
    with pytest.raises(ValueError, match="regression / hybrid"):
        resolve_acqf_cls(
            name,
            task_type="binary",
            multi_output=True,
        )


def test_canonical_log_names_can_be_resolved_directly() -> None:
    assert resolve_acqf_cls("qLogNoisyExpectedImprovement") is qLogNoisyExpectedImprovement
    assert resolve_acqf_cls("qLogProbabilityOfFeasibility") is qLogProbabilityOfFeasibility
    assert (
        resolve_acqf_cls("qLogExpectedHypervolumeImprovement")
        is qLogExpectedHypervolumeImprovement
    )
    assert (
        resolve_acqf_cls("qLogNoisyExpectedHypervolumeImprovement")
        is qLogNoisyExpectedHypervolumeImprovement
    )


def test_log_aliases_are_exposed_by_registry_listing() -> None:
    names = set(available_acqf_names())

    assert {"lognei", "logpof", "logehvi", "lognehvi", "lognparego"} <= names
