from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from bochan.acquisition.regression.active_learning import (
    qRegressionBALD,
    qRegressionPosteriorVariance,
    qRegressionPredictiveEntropy,
)
from bochan.api.acquisition_registry import resolve_acqf_cls


class _HomoskedasticGaussianModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation_noise_calls: list[bool] = []

    def posterior(self, X: torch.Tensor, observation_noise: bool = False):
        self.observation_noise_calls.append(bool(observation_noise))
        x = X[..., :1]
        latent_variance = 1.1 - (x - 0.5).square()
        noise_variance = 0.2
        variance = latent_variance + (noise_variance if observation_noise else 0.0)
        return SimpleNamespace(
            mean=torch.zeros_like(variance),
            variance=variance,
        )


def test_regression_active_learning_short_names_resolve_to_distinct_classes() -> None:
    resolved = {
        name: resolve_acqf_cls(name, task_type="regression", model_type="base", multi_output=False)
        for name in ("variance", "predictive_entropy", "BALD")
    }

    assert resolved["variance"] is qRegressionPosteriorVariance
    assert resolved["predictive_entropy"] is qRegressionPredictiveEntropy
    assert resolved["BALD"] is qRegressionBALD
    assert len(set(resolved.values())) == 3


def test_single_output_predictive_entropy_uses_observation_predictive_variance() -> None:
    model = _HomoskedasticGaussianModel()
    acquisition = qRegressionPredictiveEntropy(model)
    X = torch.tensor([[[0.25]]], dtype=torch.double)

    value = acquisition(X)

    assert torch.isfinite(value).all()
    assert model.observation_noise_calls == [True]


def test_homoskedastic_gaussian_local_uncertainty_scores_have_same_argmax() -> None:
    X = torch.tensor([[[0.1]], [[0.5]], [[0.9]]], dtype=torch.double)
    scores = []
    for acquisition_cls in (
        qRegressionPosteriorVariance,
        qRegressionPredictiveEntropy,
        qRegressionBALD,
    ):
        acquisition = acquisition_cls(_HomoskedasticGaussianModel())
        scores.append(acquisition(X))

    argmaxes = [int(torch.argmax(score)) for score in scores]
    assert argmaxes == [1, 1, 1]
    assert not torch.allclose(scores[0], scores[1])
    assert not torch.allclose(scores[0], scores[2])


def test_web_ui_explains_regression_active_learning_equivalence() -> None:
    source = (Path(__file__).parents[1] / "web" / "src" / "pages" / "OptimizePage.tsx").read_text(encoding="utf-8")

    assert "等分散Gaussian回帰" in source
    assert "同じ候補順位になるのが正常" in source
    assert "NIPV" in source
