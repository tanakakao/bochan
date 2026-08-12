from __future__ import annotations

import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import nn

from bochan.api import BayesianOptimizer
from bochan.api.configs import FitConfig, ModelConfig, MultiOutputConfig
from bochan.api.observation import ExperimentFailureConfig, ObservationData


class _RecordingModel(Model):
    def __init__(self, train_X, train_Y, **kwargs) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=train_X.dtype))
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_inputs = (train_X,)
        self.train_targets = train_Y.squeeze(-1)

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X, **kwargs):
        mean = X[..., :1] * 0.0 + self.anchor
        q = int(X.shape[-2])
        covariance = torch.eye(q, dtype=X.dtype, device=X.device).expand(
            *X.shape[:-2], q, q
        )
        return GPyTorchPosterior(MultivariateNormal(mean.squeeze(-1), covariance))


class _RecordingSuccessModel(_RecordingModel):
    def probability_posterior(self, X, **kwargs):
        return self.posterior(X, **kwargs)


def _recording_config(task_type: str = "regression") -> ModelConfig:
    return ModelConfig(
        task_type=task_type,
        model_factory=lambda **kwargs: _RecordingModel(**kwargs),
        outcome_transform=False,
    )


def test_hybrid_submodels_fit_only_their_observed_rows() -> None:
    X = torch.arange(6, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, float("nan")],
            [float("nan"), 30.0],
            [4.0, 40.0],
            [5.0, float("nan")],
            [float("nan"), 60.0],
        ],
        dtype=torch.double,
    )
    model_config = ModelConfig(
        task_type="hybrid",
        model_type="base",
        outcome_transform=False,
        multi_output_config=MultiOutputConfig(
            output_configs=[
                _recording_config("regression"),
                _recording_config("regression"),
            ],
            output_names=["strength", "conductivity"],
            use_hybrid=True,
        ),
    )
    bo = BayesianOptimizer(model_config, FitConfig(skip_fit=True))
    bo.fit(X, Y)

    sub_bundles = bo.bundle.metadata["sub_bundles"]
    torch.testing.assert_close(
        sub_bundles[0].train_X.squeeze(-1),
        torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=torch.double),
    )
    torch.testing.assert_close(
        sub_bundles[1].train_X.squeeze(-1),
        torch.tensor([0.0, 2.0, 3.0, 5.0], dtype=torch.double),
    )
    assert bo.bundle.metadata["partial_observation"] is True
    assert bo.bundle.metadata["observed_per_output"] == [4, 4]
    assert torch.isnan(bo.train_Y).any()
    assert torch.isnan(bo.model.train_Y).any()


def test_hybrid_regression_binary_ordinal_use_independent_observation_rows() -> None:
    X = torch.arange(6, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(
        [
            [10.0, 1.0, float("nan")],
            [11.0, float("nan"), 0.0],
            [float("nan"), 0.0, 1.0],
            [13.0, 1.0, 2.0],
            [14.0, float("nan"), 2.0],
            [float("nan"), 0.0, float("nan")],
        ],
        dtype=torch.double,
    )
    model_config = ModelConfig(
        task_type="hybrid",
        model_type="base",
        outcome_transform=False,
        multi_output_config=MultiOutputConfig(
            output_configs=[
                _recording_config("regression"),
                _recording_config("binary"),
                _recording_config("ordinal"),
            ],
            output_names=["strength", "crack", "grade"],
            use_hybrid=True,
        ),
    )
    bo = BayesianOptimizer(model_config, FitConfig(skip_fit=True))
    bo.fit(X, Y)

    sub_bundles = bo.bundle.metadata["sub_bundles"]
    assert [bundle.task_type for bundle in sub_bundles] == [
        "regression",
        "binary",
        "ordinal",
    ]
    assert bo.bundle.metadata["observed_per_output"] == [4, 4, 4]
    torch.testing.assert_close(
        sub_bundles[0].train_X.squeeze(-1),
        torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=torch.double),
    )
    torch.testing.assert_close(
        sub_bundles[1].train_X.squeeze(-1),
        torch.tensor([0.0, 2.0, 3.0, 5.0], dtype=torch.double),
    )
    torch.testing.assert_close(
        sub_bundles[2].train_X.squeeze(-1),
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double),
    )
    assert torch.isnan(bo.model.train_Y).any()


def test_failed_rows_are_excluded_from_objective_but_used_by_success_model() -> None:
    X = torch.arange(5, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(
        [[1.0], [float("nan")], [3.0], [float("nan")], [5.0]],
        dtype=torch.double,
    )
    observations = ObservationData.from_status(
        X,
        Y,
        status=["success", "failed", "success", "pending", "success"],
    )
    objective_config = ModelConfig(
        task_type="regression",
        model_factory=lambda **kwargs: _RecordingModel(**kwargs),
        outcome_transform=False,
    )
    failure_model_config = ModelConfig(
        task_type="binary",
        model_factory=lambda **kwargs: _RecordingSuccessModel(**kwargs),
        outcome_transform=False,
    )
    failure_config = ExperimentFailureConfig(
        model_config=failure_model_config,
        fit_config=FitConfig(skip_fit=True),
    )
    bo = BayesianOptimizer(objective_config, FitConfig(skip_fit=True))
    bo.fit_observations(observations, failure_config=failure_config)

    torch.testing.assert_close(
        bo.bundle.train_X.squeeze(-1),
        torch.tensor([0.0, 2.0, 4.0], dtype=torch.double),
    )
    torch.testing.assert_close(
        bo.failure_bundle.train_X.squeeze(-1),
        torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.double),
    )
    torch.testing.assert_close(
        bo.failure_bundle.train_Y.squeeze(-1),
        torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.double),
    )
    context = bo._resolve_data_context(None)
    torch.testing.assert_close(context.X_pending, X[3:4])


def test_kronecker_missing_targets_are_rejected_without_imputation() -> None:
    X = torch.rand(5, 2, dtype=torch.double)
    Y = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, float("nan")],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ],
        dtype=torch.double,
    )
    bo = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_type="kronecker",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )

    try:
        bo.fit(X, Y)
    except ValueError as exc:
        assert "Kronecker" in str(exc)
        assert "multitask" in str(exc)
    else:
        raise AssertionError("Kronecker partial observations must not be silently imputed.")


def test_wide_multitask_keeps_nan_cells_as_unobserved() -> None:
    X = torch.rand(6, 2, dtype=torch.double)
    Y = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, float("nan")],
            [float("nan"), 3.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ],
        dtype=torch.double,
    )
    bo = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_type="multitask",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )
    bo.fit(X, Y)

    assert type(bo.model).__name__ == "WideMultiTaskGP"
    assert torch.isnan(bo.train_Y).sum().item() == 2
    assert torch.isnan(bo.model.train_Y_wide).sum().item() == 2
