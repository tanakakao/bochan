from types import SimpleNamespace

import pytest
import torch

from bochan.api.configs import OptimizeConfig
from bochan.api.optimizer.dispatch import optimize_candidates
from bochan.models.multifidelity.optimization import (
    enumerate_discrete_fidelities_into_opt_config,
)


class _Model:
    fidelity_features = (2,)
    target_fidelities = {2: 1.0}


class _Acq:
    model = _Model()


def test_optimize_config_validates_fidelity_values():
    config = OptimizeConfig(fidelity_values=[0.25, 0.5, 1.0])
    assert config.fidelity_values == (0.25, 0.5, 1.0)

    with pytest.raises(ValueError, match="must not be empty"):
        OptimizeConfig(fidelity_values=[])
    with pytest.raises(ValueError, match="duplicates"):
        OptimizeConfig(fidelity_values=[0.5, 0.5])


def test_enumerates_discrete_fidelity_for_normal_input():
    config = OptimizeConfig(fidelity_values=[0.25, 1.0])
    resolved = enumerate_discrete_fidelities_into_opt_config(
        config,
        model=_Model(),
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
    )
    assert resolved.fixed_features_list == [{2: 0.25}, {2: 1.0}]


def test_crosses_categorical_assignments_with_fidelity_values():
    config = OptimizeConfig(
        fidelity_values=[0.25, 1.0],
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
    )
    resolved = enumerate_discrete_fidelities_into_opt_config(
        config,
        model=_Model(),
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
    )
    assert resolved.fixed_features_list == [
        {1: 0.0, 2: 0.25},
        {1: 0.0, 2: 1.0},
        {1: 1.0, 2: 0.25},
        {1: 1.0, 2: 1.0},
    ]


def test_discrete_fidelity_overrides_target_fixing_during_dispatch():
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.zeros(1, 3, dtype=torch.double), torch.tensor(0.0)

    config = OptimizeConfig(
        fidelity_values=[0.25, 1.0],
        ensure_unique_candidates=False,
    )
    optimize_candidates(
        _Acq(),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
        config,
        base_optimize_candidates=backend,
    )
    resolved = captured["config"]
    assert resolved.fixed_features is None
    assert resolved.fixed_features_list == [{2: 0.25}, {2: 1.0}]
    assert "mixed" in str(resolved.optimizer)


def test_rejects_fidelity_value_outside_bounds():
    config = OptimizeConfig(fidelity_values=[0.25, 1.25])
    with pytest.raises(ValueError, match="within bounds"):
        enumerate_discrete_fidelities_into_opt_config(
            config,
            model=_Model(),
            bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
        )


def test_rejects_non_multifidelity_model():
    config = OptimizeConfig(fidelity_values=[0.25, 1.0])
    with pytest.raises(ValueError, match="multi-fidelity model"):
        enumerate_discrete_fidelities_into_opt_config(
            config,
            model=SimpleNamespace(),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        )
