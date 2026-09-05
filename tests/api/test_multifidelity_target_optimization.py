from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import OptimizeConfig
from bochan.api.optimizer.dispatch import optimize_candidates
from bochan.models.multifidelity import (
    merge_target_fidelities_into_opt_config,
    target_fidelity_fixed_features,
)


class _Acqf:
    def __init__(self, model):
        self.model = model


class _Model:
    def __init__(self, targets=None):
        self.target_fidelities = targets


def test_target_fidelity_fixed_features_reads_model_targets():
    model = _Model({2: 1.0})

    assert target_fidelity_fixed_features(model) == {2: 1.0}


def test_target_fidelity_fixed_features_is_noop_without_targets():
    assert target_fidelity_fixed_features(SimpleNamespace()) == {}


def test_merge_target_fidelity_preserves_other_fixed_features():
    config = OptimizeConfig(
        fixed_features={0: 0.25},
        ensure_unique_candidates=False,
    )

    resolved = merge_target_fidelities_into_opt_config(
        config,
        model=_Model({2: 1.0}),
    )

    assert resolved.fixed_features == {0: 0.25, 2: 1.0}
    assert config.fixed_features == {0: 0.25}


def test_merge_target_fidelity_rejects_conflicting_fixed_feature():
    config = OptimizeConfig(fixed_features={2: 0.5})

    with pytest.raises(ValueError, match="conflicts with the model target fidelity"):
        merge_target_fidelities_into_opt_config(
            config,
            model=_Model({2: 1.0}),
        )


def test_merge_target_fidelity_rejects_conflicting_mixed_assignment():
    config = OptimizeConfig(
        fixed_features_list=[{1: 0.0, 2: 0.5}],
    )

    with pytest.raises(ValueError, match="fixed_features_list conflicts"):
        merge_target_fidelities_into_opt_config(
            config,
            model=_Model({2: 1.0}),
        )


def test_dispatch_applies_target_fidelity_to_normal_optimizer():
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.2, 0.4, 1.0]]), torch.tensor(0.0)

    config = OptimizeConfig(ensure_unique_candidates=False)
    optimize_candidates(
        acqf=_Acqf(_Model({2: 1.0})),
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        config=config,
        base_optimize_candidates=backend,
    )

    assert captured["config"].fixed_features == {2: 1.0}


def test_dispatch_combines_target_fidelity_with_mixed_categories():
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.2, 0.0, 0.4, 1.0]]), torch.tensor(0.0)

    config = OptimizeConfig(
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        ensure_unique_candidates=False,
    )
    optimize_candidates(
        acqf=_Acqf(_Model({3: 1.0})),
        bounds=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        ),
        config=config,
        base_optimize_candidates=backend,
    )

    resolved = captured["config"]
    assert resolved.fixed_features == {3: 1.0}
    assert resolved.fixed_features_list == [{1: 0.0}, {1: 1.0}]


def test_dispatch_is_noop_for_non_multifidelity_models():
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.2, 0.4]]), torch.tensor(0.0)

    config = OptimizeConfig(
        fixed_features={0: 0.2},
        ensure_unique_candidates=False,
    )
    optimize_candidates(
        acqf=_Acqf(SimpleNamespace()),
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        config=config,
        base_optimize_candidates=backend,
    )

    assert captured["config"].fixed_features == {0: 0.2}
