from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match in {path}, found {count}.")
    file_path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


replace_once(
    "src/bochan/acquisition/regression/active_learning/_pointwise.py",
    '''class qRegressionPredictiveEntropy(_RegressionActiveLearningBase):
    """Regression predictive-entropy acquisition for Gaussian predictive marginals."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        var, Xt = self._posterior_variance_score(X)
        entropy = 0.5 * torch.log(
''',
    '''class qRegressionPredictiveEntropy(_RegressionActiveLearningBase):
    """Regression predictive entropy of the noisy Gaussian observation."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        _, var, Xt = self._posterior_mean_variance(X, observation_noise=True)
        entropy = 0.5 * torch.log(
''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''  const projectedModel = modelType === "pca" || modelType === "rembo";
  const sequentialForced = q > 1 && (
''',
    '''  const projectedModel = modelType === "pca" || modelType === "rembo";
  const regressionLocalUncertaintyEquivalent = (
    acquisitionFamily === "active_learning"
    && homogeneousTask
    && taskTypes[0] === "regression"
    && ["variance", "predictive_entropy", "bald"].includes(acquisition.toLowerCase())
  );
  const sequentialForced = q > 1 && (
''',
)

replace_once(
    "web/src/pages/OptimizePage.tsx",
    '''          <p className="settings-note">
            {searchMethod === "nsgaii" && "NSGA-II選択時は、内部的にNSGA-II用のベクトル獲得戦略へ切り替えます。"}
            {searchMethod !== "nsgaii" && acquisitionFamily === "bayesian_optimization" && "目的値の改善を狙って候補を選びます。"}
            {acquisitionFamily === "active_learning" && "予測不確実性を減らすために情報量の高い候補を選びます。"}
            {acquisitionFamily === "level_set_estimation" && "設定した境界や目標付近を重点的に探索します。"}
          </p>
''',
    '''          <p className="settings-note">
            {searchMethod === "nsgaii" && "NSGA-II選択時は、内部的にNSGA-II用のベクトル獲得戦略へ切り替えます。"}
            {searchMethod !== "nsgaii" && acquisitionFamily === "bayesian_optimization" && "目的値の改善を狙って候補を選びます。"}
            {acquisitionFamily === "active_learning" && "予測不確実性を減らすために情報量の高い候補を選びます。"}
            {acquisitionFamily === "level_set_estimation" && "設定した境界や目標付近を重点的に探索します。"}
          </p>
          {regressionLocalUncertaintyEquivalent && (
            <p className="settings-note">
              標準の等分散Gaussian回帰では、Variance・Predictive Entropy・BALDはposterior varianceの単調変換になるため、
              同じ候補順位になるのが正常です。異なる観点で実験点を選びたい場合は、領域全体の不確実性低減を評価するNIPVを使用してください。
            </p>
          )}
''',
)

Path("tests/test_regression_active_learning_equivalence.py").write_text(
    '''from __future__ import annotations

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
    source = (
        Path(__file__).parents[1] / "web" / "src" / "pages" / "OptimizePage.tsx"
    ).read_text(encoding="utf-8")

    assert "等分散Gaussian回帰" in source
    assert "同じ候補順位になるのが正常" in source
    assert "NIPV" in source
''',
    encoding="utf-8",
    newline="\n",
)

print("Applied regression active-learning clarity changes.")
