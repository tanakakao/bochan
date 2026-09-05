from types import SimpleNamespace

import pytest
import torch

from bochan.api.configs import OptimizeConfig
from bochan.api.optimizer.dispatch import optimize_candidates
from bochan.models.multifidelity.optimization import (
    merge_target_fidelities_into_opt_config,
    prepare_continuous_fidelity_optimization,
)


class _Model:
    fidelity_features = (2,)
    target_fidelities = {2: 1.0}


class _MultiModel:
    fidelity_features = (2, 3)
    target_fidelities = {2: 1.0, 3: 1.0}


class _Acq:
    model = _Model()


def _bounds():
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.double,
    )


def _multi_bounds():
    return torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=torch.double,
    )


def test_optimize_config_rejects_discrete_and_continuous_fidelity_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        OptimizeConfig(
            fidelity_values=[0.25, 1.0],
            optimize_fidelity=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        OptimizeConfig(
            fidelity_assignments=[{2: 0.25}],
            optimize_fidelity=True,
        )


def test_continuous_fidelity_mode_skips_target_fixing():
    config = OptimizeConfig(optimize_fidelity=True)
    prepared = prepare_continuous_fidelity_optimization(
        config,
        model=_Model(),
        bounds=_bounds(),
    )
    resolved = merge_target_fidelities_into_opt_config(prepared, model=_Model())

    assert resolved.optimize_fidelity is True
    assert resolved.fixed_features is None
    assert resolved.fixed_features_list is None


def test_multidimensional_continuous_mode_leaves_all_fidelities_free():
    config = OptimizeConfig(optimize_fidelity=True)
    prepared = prepare_continuous_fidelity_optimization(
        config,
        model=_MultiModel(),
        bounds=_multi_bounds(),
    )
    resolved = merge_target_fidelities_into_opt_config(prepared, model=_MultiModel())

    assert resolved.optimize_fidelity is True
    assert resolved.fixed_features is None
    assert resolved.fixed_features_list is None


def test_normal_joint_optimization_leaves_fidelity_free():
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.3, 0.4, 0.37]], dtype=torch.double), torch.tensor(1.0)

    candidates, _ = optimize_candidates(
        _Acq(),
        _bounds(),
        OptimizeConfig(
            optimize_fidelity=True,
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    resolved = captured["config"]
    assert resolved.fixed_features is None
    assert resolved.fixed_features_list is None
    assert resolved.optimizer == "optimize_acqf"
    assert candidates[0, 2].item() == pytest.approx(0.37)


def test_multidimensional_joint_optimization_leaves_both_fidelities_free():
    captured = {}
    AcqType = type("qMultiFidelityKnowledgeGradient", (), {})
    acqf = AcqType()
    acqf.model = _MultiModel()

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.3, 0.4, 0.37, 0.62]], dtype=torch.double), torch.tensor(1.0)

    candidates, _ = optimize_candidates(
        acqf,
        _multi_bounds(),
        OptimizeConfig(
            optimize_fidelity=True,
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    resolved = captured["config"]
    assert resolved.fixed_features is None
    assert resolved.fixed_features_list is None
    assert resolved.optimizer == "optimize_acqf"
    assert candidates[0, 2].item() == pytest.approx(0.37)
    assert candidates[0, 3].item() == pytest.approx(0.62)


def test_mixed_joint_optimization_keeps_categories_fixed_but_fidelity_free():
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.3, 1.0, 0.42]], dtype=torch.double), torch.tensor(1.0)

    optimize_candidates(
        _Acq(),
        _bounds(),
        OptimizeConfig(
            optimize_fidelity=True,
            fixed_features_list=[{1: 0.0}, {1: 1.0}],
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    resolved = captured["config"]
    assert resolved.fixed_features_list == [{1: 0.0}, {1: 1.0}]
    assert all(2 not in assignment for assignment in resolved.fixed_features_list)
    assert "mixed" in str(resolved.optimizer)


def test_multidimensional_mixed_joint_optimization_keeps_all_fidelities_free():
    captured = {}
    AcqType = type("qMultiFidelityKnowledgeGradient", (), {})
    acqf = AcqType()
    acqf.model = _MultiModel()

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.3, 1.0, 0.42, 0.65]], dtype=torch.double), torch.tensor(1.0)

    optimize_candidates(
        acqf,
        _multi_bounds(),
        OptimizeConfig(
            optimize_fidelity=True,
            fixed_features_list=[{1: 0.0}, {1: 1.0}],
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    resolved = captured["config"]
    assert resolved.fixed_features_list == [{1: 0.0}, {1: 1.0}]
    assert all(2 not in assignment and 3 not in assignment for assignment in resolved.fixed_features_list)
    assert "mixed" in str(resolved.optimizer)


def test_continuous_fidelity_rejects_global_fidelity_fix():
    config = OptimizeConfig(
        optimize_fidelity=True,
        fixed_features={2: 0.5},
    )
    with pytest.raises(ValueError, match="conflicts with fixed_features"):
        prepare_continuous_fidelity_optimization(
            config,
            model=_Model(),
            bounds=_bounds(),
        )


def test_multidimensional_continuous_fidelity_rejects_any_global_fidelity_fix():
    config = OptimizeConfig(
        optimize_fidelity=True,
        fixed_features={3: 0.5},
    )
    with pytest.raises(ValueError, match="conflicts with fixed_features"):
        prepare_continuous_fidelity_optimization(
            config,
            model=_MultiModel(),
            bounds=_multi_bounds(),
        )


def test_continuous_fidelity_rejects_list_fidelity_fix():
    config = OptimizeConfig(
        optimize_fidelity=True,
        fixed_features_list=[{1: 0.0, 2: 0.5}],
    )
    with pytest.raises(ValueError, match="fixed_features_list"):
        prepare_continuous_fidelity_optimization(
            config,
            model=_Model(),
            bounds=_bounds(),
        )


def test_continuous_fidelity_rejects_non_multifidelity_model():
    config = OptimizeConfig(optimize_fidelity=True)
    with pytest.raises(ValueError, match="multi-fidelity model"):
        prepare_continuous_fidelity_optimization(
            config,
            model=SimpleNamespace(),
            bounds=_bounds(),
        )


def test_multifidelity_acquisition_accepts_continuous_fidelity_mode():
    AcqType = type("qMultiFidelityKnowledgeGradient", (), {})
    acqf = AcqType()
    acqf.model = _Model()
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.2, 0.0, 0.33]], dtype=torch.double), torch.tensor(0.0)

    optimize_candidates(
        acqf,
        _bounds(),
        OptimizeConfig(
            optimize_fidelity=True,
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    assert captured["config"].optimize_fidelity is True
    assert captured["config"].fixed_features is None


def test_multifidelity_acquisition_requires_query_fidelity_mode():
    AcqType = type("qMultiFidelityMaxValueEntropy", (), {})
    acqf = AcqType()
    acqf.model = _Model()

    with pytest.raises(ValueError, match="require OptimizeConfig.fidelity_values"):
        optimize_candidates(
            acqf,
            _bounds(),
            OptimizeConfig(ensure_unique_candidates=False),
            base_optimize_candidates=lambda **kwargs: (
                torch.zeros(1, 3, dtype=torch.double),
                torch.tensor(0.0),
            ),
        )


def test_wrapped_acquisition_resolves_multifidelity_model():
    wrapped = SimpleNamespace(base_acqf=SimpleNamespace(model=_Model()))
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.zeros(1, 3, dtype=torch.double), torch.tensor(0.0)

    optimize_candidates(
        wrapped,
        _bounds(),
        OptimizeConfig(
            optimize_fidelity=True,
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    assert captured["config"].optimize_fidelity is True
    assert captured["config"].fixed_features is None
