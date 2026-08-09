"""Feature-importance regression tests for one-to-many input transforms."""

import torch

from bochan.inspection import FeatureImportanceConfig, compute_feature_importance


class Posterior:
    """Minimal posterior exposing only a mean tensor."""

    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean


class ExpandedLinearModel:
    """Mimic InputPerturbation by returning four predictions per input row."""

    def posterior(self, X: torch.Tensor) -> Posterior:
        mean = 3.0 * X[:, :1]
        expanded = mean.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 1)
        return Posterior(expanded)


class ExpandedMulticlassModel:
    """Return four flattened probability vectors per nominal input row."""

    def class_probs_list(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
    ) -> list[torch.Tensor]:
        assert output_indices in (None, [0])
        logits = torch.stack(
            [
                -8.0 * (X[:, 0] - 0.1).square(),
                -8.0 * (X[:, 0] - 0.5).square(),
                -8.0 * (X[:, 0] - 0.9).square(),
            ],
            dim=-1,
        )
        probability = torch.softmax(logits, dim=-1)
        expanded = probability.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)
        return [expanded]


def test_regression_importance_aggregates_expanded_prediction_rows() -> None:
    """RMSE must compare one aggregated prediction with each nominal target."""

    X = torch.arange(20, dtype=torch.double).reshape(10, 2) / 20.0
    y = 3.0 * X[:, 0]

    result = compute_feature_importance(
        model=ExpandedLinearModel(),
        X=X,
        y=y,
        feature_names=["signal", "noise"],
        config=FeatureImportanceConfig(n_repeats=2, diagnostic_methods=[]),
    )

    method = result.output.predictive_methods["permutation"]
    assert method.baseline_metrics["rmse"] < 1e-12
    assert method.entries["signal"].importance.mean > method.entries["noise"].importance.mean


def test_multiclass_importance_aggregates_expanded_probability_rows() -> None:
    """Class probabilities must be averaged per nominal row before log loss."""

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

    result = compute_feature_importance(
        model=ExpandedMulticlassModel(),
        X=X,
        y=y,
        task_type="multiclass",
        feature_names=["signal", "noise"],
        config=FeatureImportanceConfig(n_repeats=2, diagnostic_methods=[]),
    )

    method = result.output.predictive_methods["permutation"]
    assert "multiclass_log_loss" in method.baseline_metrics
    assert method.entries["signal"].importance.mean > method.entries["noise"].importance.mean
