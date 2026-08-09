"""Regression tests for InputPerturbation shape handling in regression LSE."""

from __future__ import annotations

import pytest
import torch
from botorch.models.model import Model
from torch import Tensor

from bochan.acquisition.regression.levelset_estimation import (
    MultiOutputRegressionLevelSetScoreObjective,
    RegressionLevelSetScoreObjective,
    qMultiOutputRegressionStraddle,
    qRegressionBoundaryVariance,
    qRegressionICU,
    qRegressionStraddle,
)


class _Posterior:
    """Minimal posterior exposing deterministic mean and variance tensors."""

    def __init__(self, mean: Tensor, variance: Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _EvalOnlyPerturbation(torch.nn.Module):
    """Mimic an eval-only one-to-many transform with n_w=4."""

    is_one_to_many = True
    transform_on_train = False

    def forward(self, X: Tensor) -> Tensor:
        return X.repeat_interleave(4, dim=-2)

    def preprocess_transform(self, X: Tensor) -> Tensor:
        # Keep the candidate count nominal while still mimicking one-to-one
        # preprocessing such as Normalize.
        return X + 0.0


class _NominalPosteriorModel(Model):
    """Return nominal q posterior moments while transform_inputs expands q."""

    def __init__(self, *, num_outputs: int = 1) -> None:
        super().__init__()
        self._num_outputs = int(num_outputs)
        self.input_transform = _EvalOnlyPerturbation()
        self.train_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def transform_inputs(
        self,
        X: Tensor,
        input_transform: torch.nn.Module | None = None,
    ) -> Tensor:
        transform = self.input_transform if input_transform is None else input_transform
        return transform(X)

    def posterior(self, X: Tensor, **_: object) -> _Posterior:
        base = X[..., :1]
        mean = torch.cat(
            [base + 0.1 * index for index in range(self._num_outputs)],
            dim=-1,
        )
        variance = torch.full_like(mean, 0.04)
        return _Posterior(mean, variance)


class _ExpandedPosteriorModel(_NominalPosteriorModel):
    """Return q*n_w posterior moments, matching external tree estimators."""

    def posterior(self, X: Tensor, **_: object) -> _Posterior:
        expanded = self.transform_inputs(X)
        base = expanded[..., :1]
        mean = torch.cat(
            [base + 0.1 * index for index in range(self._num_outputs)],
            dim=-1,
        )
        variance = torch.full_like(mean, 0.04)
        return _Posterior(mean, variance)


@pytest.mark.parametrize(
    "acqf_cls",
    [qRegressionStraddle, qRegressionICU, qRegressionBoundaryVariance],
)
def test_single_output_lse_accepts_nominal_posterior_with_expanded_transform(
    acqf_cls: type,
) -> None:
    """Reported q=1 / n_w=4 shape mismatch must not reach LSE scoring."""

    model = _NominalPosteriorModel()
    X = torch.linspace(0.05, 0.95, 256, dtype=torch.double).reshape(256, 1, 1)
    kwargs: dict[str, object] = {
        "model": model,
        "threshold": 0.5,
        "n_w": 4,
        "X_observed": model.train_X,
    }
    if acqf_cls is qRegressionBoundaryVariance:
        kwargs["tau"] = 0.2

    acqf = acqf_cls(**kwargs)
    score = acqf(X)

    assert score.shape == (256,)
    assert torch.isfinite(score).all()


def test_single_output_straddle_keeps_already_expanded_posterior_contract() -> None:
    """External estimators exposing q*n_w rows must retain perturbation scoring."""

    model = _ExpandedPosteriorModel()
    X = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.double)
    acqf = qRegressionStraddle(
        model=model,
        threshold=0.5,
        n_w=4,
        X_observed=model.train_X,
    )

    score = acqf(X)

    assert score.shape == (2,)
    assert torch.isfinite(score).all()


def test_multi_output_straddle_accepts_nominal_posterior_with_expanded_transform() -> None:
    """The same nominal/expanded alignment must apply to multi-output LSE."""

    model = _NominalPosteriorModel(num_outputs=2)
    X = torch.linspace(0.1, 0.9, 16, dtype=torch.double).reshape(16, 1, 1)
    acqf = qMultiOutputRegressionStraddle(
        model=model,
        thresholds=[0.4, 0.6],
        output_weights=[1.0, 1.0],
        output_reduction="weighted_mean",
        n_w=4,
        X_observed=model.train_X,
    )

    score = acqf(X)

    assert score.shape == (16,)
    assert torch.isfinite(score).all()


@pytest.mark.parametrize(
    "objective_cls",
    [RegressionLevelSetScoreObjective, MultiOutputRegressionLevelSetScoreObjective],
)
def test_lse_risk_objective_accepts_already_aggregated_nominal_q(
    objective_cls: type,
) -> None:
    """VaR/CVaR objectives must not reject a wrapper-aggregated q score."""

    X = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.double)
    score = torch.tensor([[0.3], [0.7]], dtype=torch.double)
    objective = objective_cls(n_w=4, risk_type="cvar", alpha=0.5)

    actual = objective(score, X=X)

    torch.testing.assert_close(actual, score)
