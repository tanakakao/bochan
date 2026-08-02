import json

import pytest
import torch

from bochan.inspection import FeatureGroup, FeatureImportanceConfig, compute_feature_importance


class Posterior:
    def __init__(self, mean):
        self.mean = mean


class LinearModel:
    def posterior(self, X):
        return Posterior(3 * X[:, :1])


def test_config_defaults_are_independent_and_validated():
    first, second = FeatureImportanceConfig(), FeatureImportanceConfig()
    assert first.predictive_methods == ["permutation"]
    assert first.diagnostic_methods == ["auto"]
    assert first.predictive_methods is not second.predictive_methods
    with pytest.raises(ValueError):
        FeatureImportanceConfig(predictive_methods=["shap"])
    with pytest.raises(ValueError):
        FeatureImportanceConfig(diagnostic_methods=["unknown"])
    with pytest.raises(ValueError):
        FeatureImportanceConfig(n_repeats=0)


def test_raw_permutation_is_reproducible_and_does_not_mutate_input():
    X = torch.arange(40, dtype=torch.float).reshape(20, 2)
    y = 3 * X[:, 0]
    original = X.clone()
    kwargs = dict(model=LinearModel(), X=X, y=y, feature_names=["signal", "noise"])
    result = compute_feature_importance(**kwargs, config=FeatureImportanceConfig(n_repeats=4))
    repeated = compute_feature_importance(**kwargs, config=FeatureImportanceConfig(n_repeats=4))
    entries = result.output.predictive_methods["permutation"].entries
    assert entries["signal"].importance.mean > entries["noise"].importance.mean
    assert torch.equal(
        entries["signal"].importance.values,
        repeated.output.predictive_methods["permutation"].entries["signal"].importance.values,
    )
    assert len(entries["signal"].importance.values) == 4
    assert torch.equal(X, original)
    json.dumps(result.to_dict())


def test_joint_group_validation_and_metadata():
    X = torch.randn(12, 3)
    y = 3 * X[:, 0]
    config = FeatureImportanceConfig(n_repeats=2, feature_groups=[FeatureGroup("coords", (0, 1))])
    result = compute_feature_importance(model=LinearModel(), X=X, y=y, config=config)
    entry = result.output.predictive_methods["permutation"].entries["coords"]
    assert entry.metadata["permutation_strategy"] == "joint_row_permutation"
    with pytest.raises(ValueError):
        compute_feature_importance(
            model=LinearModel(), X=X, y=y, config=FeatureImportanceConfig(feature_groups=[FeatureGroup("bad", (3,))])
        )
