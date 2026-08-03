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


class HybridMulticlassModel:
    """Expose class probabilities separately from one-column objective output."""

    def class_probs_list(self, X, output_indices=None):
        assert output_indices in (None, [0])
        logits = torch.stack(
            [
                -4.0 * (X[:, 0] - 0.1).square(),
                -4.0 * (X[:, 0] - 0.5).square(),
                -4.0 * (X[:, 0] - 0.9).square(),
            ],
            dim=-1,
        )
        return [torch.softmax(logits, dim=-1)]


class HybridMulticlassPredictor:
    """Mimic Web hybrid prediction, whose default mean is expected utility."""

    def __init__(self):
        self.model = HybridMulticlassModel()

    def predict(self, X, return_result=False):
        expected_utility = X[:, :1]
        return Posterior(expected_utility) if return_result else expected_utility


class ScalarMulticlassModel:
    def posterior(self, X):
        return Posterior(X[:, :1])


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


def test_hybrid_multiclass_importance_uses_all_class_probabilities():
    X = torch.tensor(
        [
            [0.05, 0.0],
            [0.15, 1.0],
            [0.45, 0.0],
            [0.55, 1.0],
            [0.85, 0.0],
            [0.95, 1.0],
        ],
        dtype=torch.double,
    )
    y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.double)
    predictor = HybridMulticlassPredictor()

    result = compute_feature_importance(
        model=predictor.model,
        predictor=predictor,
        X=X,
        y=y,
        task_type="multiclass",
        feature_names=["signal", "noise"],
        config=FeatureImportanceConfig(n_repeats=3, diagnostic_methods=[]),
    )

    method = result.output.predictive_methods["permutation"]
    assert "multiclass_log_loss" in method.baseline_metrics
    assert method.entries["signal"].importance.mean > method.entries["noise"].importance.mean
    json.dumps(result.to_dict())


def test_multiclass_importance_reports_probability_width_mismatch_clearly():
    X = torch.arange(12, dtype=torch.double).reshape(6, 2)
    y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.double)

    with pytest.raises(ValueError, match="probability columns"):
        compute_feature_importance(
            model=ScalarMulticlassModel(),
            X=X,
            y=y,
            task_type="multiclass",
            config=FeatureImportanceConfig(n_repeats=1, diagnostic_methods=[]),
        )
