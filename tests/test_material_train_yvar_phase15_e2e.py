from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import nn

from bochan.api import BayesianOptimizer
from bochan.api.acquisition.provenance import candidate_acquisition_diagnostics
from bochan.api.configs import (
    AcquisitionConfig,
    FitConfig,
    ModelConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OptimizeConfig,
)
from bochan.api.observation import ExperimentFailureConfig, ObservationData


class _RecordingModel(Model):
    def __init__(
        self,
        train_X,
        train_Y,
        train_Yvar=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=train_X.dtype))
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
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


class _ProbeAcquisition(AcquisitionFunction):
    def __init__(self, model) -> None:
        super().__init__(model=model)

    def forward(self, X):
        return X[..., 0].mean(dim=-1)


def _recording_config(task_type: str = "regression") -> ModelConfig:
    return ModelConfig(
        task_type=task_type,
        model_factory=lambda **kwargs: _RecordingModel(**kwargs),
        outcome_transform=False,
    )


def _material_model_config() -> ModelConfig:
    return ModelConfig(
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


def _failure_config() -> ExperimentFailureConfig:
    return ExperimentFailureConfig(
        model_config=ModelConfig(
            task_type="binary",
            model_factory=lambda **kwargs: _RecordingSuccessModel(**kwargs),
            outcome_transform=False,
        ),
        fit_config=FitConfig(skip_fit=True),
    )


def _probe_factory(*, bundle, config, data_context):
    return _ProbeAcquisition(bundle.model)


def _scalar_acquisition(name: str, output: int) -> AcquisitionConfig:
    return AcquisitionConfig(
        name=name,
        acqf_factory=_probe_factory,
        objective_config=ObjectiveConfig(mode="scalar", output=output),
    )


def _vector_acquisition(name: str) -> AcquisitionConfig:
    return AcquisitionConfig(
        name=name,
        acqf_factory=_probe_factory,
        objective_config=ObjectiveConfig(mode="multi_output"),
    )


def _install_deterministic_candidate_optimizer(monkeypatch) -> None:
    counter = {"value": 0}

    def _fake_optimize_candidates(*, acqf, bounds, config):
        counter["value"] += 1
        value = 0.1 * counter["value"]
        candidate = torch.tensor([[value]], dtype=torch.double)
        return candidate, torch.tensor(value, dtype=torch.double)

    monkeypatch.setattr(
        "bochan.api.optimizer.optimize_candidates",
        _fake_optimize_candidates,
    )


def test_phase15_partial_yvar_failure_pending_ask_tell_compare_lifecycle(monkeypatch):
    _install_deterministic_candidate_optimizer(monkeypatch)

    X = torch.tensor([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]], dtype=torch.double)
    Y = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, float("nan")],
            [float("nan"), 12.0],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [6.0, 15.0],
        ],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [
            [0.01, 0.04],
            [0.01, float("nan")],
            [float("nan"), 0.04],
            [float("nan"), float("nan")],
            [float("nan"), float("nan")],
            [0.01, 0.04],
        ],
        dtype=torch.double,
    )
    observations = ObservationData.from_status(
        X,
        Y,
        Yvar=Yvar,
        status=["success", "success", "success", "failed", "pending", "success"],
    )

    bo = BayesianOptimizer(_material_model_config(), FitConfig(skip_fit=True))
    bo.fit_observations(observations, failure_config=_failure_config())

    assert bo.observations.report() == {
        "n_rows": 6,
        "n_completed": 5,
        "n_success": 4,
        "n_failed": 1,
        "n_pending": 1,
        "observed_per_output": [3, 3],
        "known_observation_variance": True,
    }
    assert bo.failure_model is not None
    assert bo.bundle.metadata["partial_observation"] is True
    sub_bundles = bo.bundle.metadata["sub_bundles"]
    assert sub_bundles[0].model.train_Yvar is not None
    assert sub_bundles[1].model.train_Yvar is not None
    torch.testing.assert_close(
        bo._resolve_data_context(None).X_pending,
        torch.tensor([[0.8]], dtype=torch.double),
    )

    first = bo.ask(
        _scalar_acquisition("strength_before_tell", 0),
        OptimizeConfig(q=1),
        return_result=True,
    )
    first_diagnostics = candidate_acquisition_diagnostics(first)
    assert first_diagnostics is not None
    assert first_diagnostics["training_rows"] == 4
    assert first_diagnostics["baseline_rows"] == 3
    assert first_diagnostics["baseline_filtered"] is True
    assert first_diagnostics["partial_observation"] is True
    assert first_diagnostics["objective_output_indices"] == [0]
    assert first_diagnostics["known_observation_variance"] is True
    assert first_diagnostics["failed_rows"] == 1
    assert first_diagnostics["pending_rows"] == 1
    assert type(first.acqf).__name__ == "ExperimentSuccessWeightedAcquisition"

    bo.tell(
        torch.tensor([[0.8]], dtype=torch.double),
        torch.tensor([[5.0, float("nan")]], dtype=torch.double),
        torch.tensor([[0.01, float("nan")]], dtype=torch.double),
        status="success",
        refit=True,
    )

    assert bo.observations.report() == {
        "n_rows": 6,
        "n_completed": 6,
        "n_success": 5,
        "n_failed": 1,
        "n_pending": 0,
        "observed_per_output": [4, 3],
        "known_observation_variance": True,
    }
    assert bo._resolve_data_context(None).X_pending is None

    second = bo.ask(
        _scalar_acquisition("strength_after_tell", 0),
        OptimizeConfig(q=1),
        return_result=True,
    )
    second_diagnostics = candidate_acquisition_diagnostics(second)
    assert second_diagnostics is not None
    assert second_diagnostics["training_rows"] == 5
    assert second_diagnostics["baseline_rows"] == 4
    assert second_diagnostics["pending_rows"] == 0
    assert second_diagnostics["known_observation_variance"] is True

    preserved_first = candidate_acquisition_diagnostics(first)
    assert preserved_first is not None
    assert preserved_first["training_rows"] == 4
    assert preserved_first["baseline_rows"] == 3
    assert preserved_first["pending_rows"] == 1

    compared = bo.compare_acquisitions(
        [
            _scalar_acquisition("strength_compare", 0),
            _scalar_acquisition("conductivity_compare", 1),
            _vector_acquisition("vector_compare"),
        ],
        OptimizeConfig(q=1),
    )
    strength = candidate_acquisition_diagnostics(compared["strength_compare"])
    conductivity = candidate_acquisition_diagnostics(compared["conductivity_compare"])
    vector = candidate_acquisition_diagnostics(compared["vector_compare"])

    assert strength is not None
    assert conductivity is not None
    assert vector is not None
    assert strength["baseline_rows"] == 4
    assert strength["objective_output_indices"] == [0]
    assert conductivity["baseline_rows"] == 3
    assert conductivity["objective_output_indices"] == [1]
    assert vector["baseline_rows"] == 5
    assert vector["objective_output_indices"] is None
    assert vector["baseline_filtered"] is False


def test_phase15_complete_known_yvar_remains_acquisition_baseline_noop(monkeypatch):
    _install_deterministic_candidate_optimizer(monkeypatch)

    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.double)
    Yvar = torch.tensor([[0.01, 0.04], [0.01, 0.04], [0.01, 0.04]], dtype=torch.double)
    observations = ObservationData.from_status(
        X,
        Y,
        Yvar=Yvar,
        status=["success", "success", "success"],
    )

    bo = BayesianOptimizer(_material_model_config(), FitConfig(skip_fit=True))
    bo.fit_observations(observations)
    result = bo.ask(
        _scalar_acquisition("complete_strength", 0),
        OptimizeConfig(q=1),
        return_result=True,
    )

    diagnostics = candidate_acquisition_diagnostics(result)
    assert diagnostics is not None
    assert diagnostics["training_rows"] == 3
    assert diagnostics["baseline_rows"] == 3
    assert diagnostics["baseline_filtered"] is False
    assert diagnostics["partial_observation"] is False
    assert diagnostics["known_observation_variance"] is True
    assert diagnostics["failed_rows"] == 0
    assert diagnostics["pending_rows"] == 0
