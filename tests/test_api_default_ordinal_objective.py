from __future__ import annotations

import torch

from bochan.acquisition.objective import OrdinalExpectedUtilityMCObjective
from bochan.acquisition.ordinal.bayesian_optimization.utility_acquisitions import (
    qOrdinalExpectedImprovement,
)
from bochan.api.configs import (
    AcquisitionConfig,
    DataContext,
    ModelBundle,
    ModelConfig,
)
from bochan.api.engine_defaults import resolve_acquisition_defaults


class _OrdinalLikelihood:
    def __init__(self) -> None:
        self.cutpoints = torch.tensor([-0.5, 0.5], dtype=torch.double)


class _OrdinalModel:
    def __init__(self) -> None:
        self.ordinal_likelihood = _OrdinalLikelihood()


def _make_bundle() -> ModelBundle:
    model = _OrdinalModel()
    return ModelBundle(
        model=model,
        train_X=torch.zeros(3, 2, dtype=torch.double),
        train_Y=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double),
        model_config=ModelConfig(task_type="ordinal"),
        task_type="ordinal",
        model_type="base",
    )


def test_ordinal_ei_gets_default_expected_utility_objective() -> None:
    config = AcquisitionConfig(
        name="ei",
        acqf_cls=qOrdinalExpectedImprovement,
    )

    resolved, _ = resolve_acquisition_defaults(
        _make_bundle(),
        config,
        DataContext(),
    )

    assert isinstance(resolved.objective, OrdinalExpectedUtilityMCObjective)
    assert torch.equal(
        resolved.objective.utility_values,
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    )


def test_explicit_ordinal_objective_is_not_overwritten() -> None:
    explicit = object()
    config = AcquisitionConfig(
        name="ei",
        acqf_cls=qOrdinalExpectedImprovement,
        objective=explicit,
    )

    resolved, _ = resolve_acquisition_defaults(
        _make_bundle(),
        config,
        DataContext(),
    )

    assert resolved.objective is explicit


def test_non_utility_ordinal_acquisition_is_not_modified() -> None:
    class OtherOrdinalAcquisition:
        pass

    config = AcquisitionConfig(
        name="straddle",
        acqf_cls=OtherOrdinalAcquisition,
    )

    resolved, _ = resolve_acquisition_defaults(
        _make_bundle(),
        config,
        DataContext(),
    )

    assert resolved.objective is None
