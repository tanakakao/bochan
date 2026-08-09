from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.transforms.input import InputPerturbation

from bochan.models.classification.binary.external import (
    LightGBMBinaryEnsembleModel,
    NGBoostBinaryEnsembleModel,
    RandomForestBinaryClassificationModel,
)
from bochan.models.external.common import _check_one_to_one_input_transform
from bochan.models.regression.external import (
    LightGBMEnsembleModel,
    NGBoostEnsembleModel,
    RandomForestMixedRegressorModel,
    RandomForestRegressorModel,
)
from bochan.models.transforms.input import build_input_transform


class _FakeRegressor:
    def __init__(self, bias: float = 0.0) -> None:
        self.bias = float(bias)
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 0] + self.bias


class _FakeTree(_FakeRegressor):
    pass


class _FakeForest:
    def __init__(self, biases: tuple[float, ...] = (0.0, 0.5)) -> None:
        self.estimators_ = [_FakeTree(bias) for bias in biases]
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        return self


class _FakeProbabilityMember:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = float(offset)
        self.classes_ = np.array([0, 1], dtype=int)
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X, y, **kwargs):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.fit_kwargs = dict(kwargs)
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        p1 = np.clip(0.15 + 0.7 * X[:, 0] + self.offset, 0.01, 0.99)
        return np.column_stack([1.0 - p1, p1])


class _FakeProbabilityForest:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1], dtype=int)
        self.estimators_ = [
            _FakeProbabilityMember(offset=-0.05),
            _FakeProbabilityMember(offset=0.05),
        ]
        self.fit_X: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None

    def fit(self, X, y, **kwargs):
        del kwargs
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.classes_ = np.array([0, 1], dtype=int)
        return self


def _regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.25], [0.5], [0.75], [1.0]],
        dtype=torch.double,
    )
    Y = (1.5 * X + 0.1).clone()
    return X, Y


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    return X, Y


def _perturbation_transform(
    train_X: torch.Tensor,
    *,
    n_w: int = 4,
    categorical_idx: list[int] | None = None,
):
    bounds = torch.stack(
        [train_X.min(dim=0).values, train_X.max(dim=0).values],
        dim=0,
    )
    return build_input_transform(
        train_X=train_X,
        bounds=bounds,
        perturbation=True,
        categorical_idx=categorical_idx,
        n_w=n_w,
        std=0.05,
        normalize=True,
    )


def _regression_model(kind: str, train_X: torch.Tensor, train_Y: torch.Tensor, transform):
    if kind == "random_forest":
        estimator = _FakeForest()
        model = RandomForestRegressorModel(
            train_X=train_X,
            train_Y=train_Y,
            estimator=estimator,
            input_transform=transform,
        )
        fit_estimators = [estimator]
    elif kind == "lightgbm":
        estimators = [_FakeRegressor(0.0), _FakeRegressor(0.5)]
        model = LightGBMEnsembleModel(
            train_X=train_X,
            train_Y=train_Y,
            estimators=estimators,
            bootstrap=False,
            input_transform=transform,
        )
        fit_estimators = estimators
    elif kind == "ngboost":
        estimators = [_FakeRegressor(0.0), _FakeRegressor(0.5)]
        model = NGBoostEnsembleModel(
            train_X=train_X,
            train_Y=train_Y,
            estimators=estimators,
            bootstrap=False,
            input_transform=transform,
        )
        fit_estimators = estimators
    else:  # pragma: no cover - parametrization controls this
        raise AssertionError(kind)
    return model, fit_estimators


@pytest.mark.parametrize("kind", ["random_forest", "lightgbm", "ngboost"])
def test_tree_regression_ensembles_perturb_only_posterior_rows(kind: str) -> None:
    """Fitting stays nominal while posterior evaluation expands every candidate."""
    torch.manual_seed(0)
    train_X, train_Y = _regression_data()
    n_w = 4
    transform = _perturbation_transform(train_X, n_w=n_w)
    model, fit_estimators = _regression_model(kind, train_X, train_Y, transform)

    model.fit()

    for estimator in fit_estimators:
        assert estimator.fit_X is not None
        assert estimator.fit_X.shape[0] == train_X.shape[0]

    X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    posterior = model.posterior(X)
    assert posterior.mean.shape == torch.Size([X.shape[0] * n_w, 1])
    assert torch.isfinite(posterior.mean).all()


@pytest.mark.parametrize("kind", ["random_forest", "lightgbm", "ngboost"])
def test_tree_binary_ensembles_support_eval_only_input_perturbation(kind: str) -> None:
    """Probability posteriors retain one row per perturbation for all tree families."""
    torch.manual_seed(0)
    train_X, train_Y = _binary_data()
    n_w = 3
    transform = _perturbation_transform(train_X, n_w=n_w)

    if kind == "random_forest":
        estimator = _FakeProbabilityForest()
        model = RandomForestBinaryClassificationModel(
            train_X=train_X,
            train_Y=train_Y,
            estimator=estimator,
            input_transform=transform,
        )
        fit_estimators = [estimator]
    elif kind == "lightgbm":
        estimators = [
            _FakeProbabilityMember(offset=-0.05),
            _FakeProbabilityMember(offset=0.05),
        ]
        model = LightGBMBinaryEnsembleModel(
            train_X=train_X,
            train_Y=train_Y,
            estimators=estimators,
            bootstrap=False,
            input_transform=transform,
        )
        fit_estimators = estimators
    else:
        estimators = [
            _FakeProbabilityMember(offset=-0.05),
            _FakeProbabilityMember(offset=0.05),
        ]
        model = NGBoostBinaryEnsembleModel(
            train_X=train_X,
            train_Y=train_Y,
            estimators=estimators,
            bootstrap=False,
            input_transform=transform,
        )
        fit_estimators = estimators

    model.fit()
    for estimator in fit_estimators:
        assert estimator.fit_X is not None
        assert estimator.fit_X.shape[0] == train_X.shape[0]

    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(X)
    assert posterior.mean.shape == torch.Size([X.shape[0] * n_w, 1])
    assert torch.all((posterior.mean >= 0.0) & (posterior.mean <= 1.0))


@pytest.mark.parametrize("kind", ["lightgbm", "ngboost"])
def test_boosting_validation_rows_are_not_perturbation_expanded(kind: str) -> None:
    """Fit-time validation must use preprocessing only, not eval perturbations."""
    torch.manual_seed(0)
    train_X, train_Y = _regression_data()
    transform = _perturbation_transform(train_X, n_w=5)
    model, estimators = _regression_model(kind, train_X, train_Y, transform)
    X_val = torch.tensor([[0.1], [0.9]], dtype=torch.double)
    Y_val = torch.tensor([[0.2], [1.4]], dtype=torch.double)

    model.fit(X_val=X_val, Y_val=Y_val)

    for estimator in estimators:
        if kind == "lightgbm":
            eval_set = estimator.fit_kwargs["eval_set"]
            val_X, val_Y = eval_set[0]
        else:
            val_X = estimator.fit_kwargs["X_val"]
            val_Y = estimator.fit_kwargs["Y_val"]
        assert np.asarray(val_X).shape[0] == X_val.shape[0]
        assert np.asarray(val_Y).shape[0] == Y_val.shape[0]


def test_mixed_random_forest_perturbs_only_continuous_features() -> None:
    """Mixed tree models keep categorical values valid through perturbation expansion."""
    torch.manual_seed(0)
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.25, 1.0],
            [0.5, 2.0],
            [0.75, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.2).clone()
    transform = _perturbation_transform(train_X, n_w=4, categorical_idx=[1])
    estimator = _FakeForest()
    model = RandomForestMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        estimator=estimator,
        input_transform=transform,
    ).fit()

    posterior = model.posterior(torch.tensor([[0.4, 1.0]], dtype=torch.double))
    assert posterior.mean.shape == torch.Size([4, 1])
    assert torch.isfinite(posterior.mean).all()


def test_external_eval_only_one_to_many_contract_is_model_name_independent() -> None:
    """External estimator safety is based on transform timing, not model-name prefixes."""
    train_X, _ = _regression_data()
    transform = _perturbation_transform(train_X, n_w=3)

    _check_one_to_one_input_transform(transform, model_name="TabPFN")
    _check_one_to_one_input_transform(transform, model_name="Future external estimator")


def test_training_time_one_to_many_transform_is_rejected_for_external_models() -> None:
    """External estimators must never expand X without expanding fit targets."""
    perturbation = InputPerturbation(
        perturbation_set=torch.tensor([[0.0], [0.05]], dtype=torch.double),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        transform_on_train=True,
        transform_on_eval=True,
    )

    with pytest.raises(UnsupportedError, match="only for evaluation"):
        _check_one_to_one_input_transform(
            perturbation,
            model_name="Random Forest",
        )


def test_web_random_forest_cvar_runs_with_input_perturbation() -> None:
    """Web RF BO must compose finite-ensemble uncertainty with input-risk aggregation."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastapi")
    pytest.importorskip("sklearn")

    from bochan.desktop.services import DatasetStore, build_dataset_record
    from bochan.serving.webapp.app import RegressionRunRequest
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).numpy()
    data = pd.DataFrame({"x": x, "y": 1.0 - (x - 0.65) ** 2})
    record = build_dataset_record(
        data=data,
        name="tree-perturbation.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)

    request = RegressionRunRequest(
        dataset_id=record.dataset_id,
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        model_type="random_forest",
        model_kwargs={
            "n_estimators": 8,
            "random_state": 0,
            "web_target_settings": [
                {
                    "target": "y",
                    "task_type": "regression",
                    "optimize": True,
                    "direction": "maximize",
                    "goal": "none",
                    "value": None,
                }
            ],
        },
        fit_maxiter=4,
        normalize=True,
        outcome_transform=True,
        input_perturbation=True,
        n_w=4,
        perturbation_std=0.05,
        search_space=[
            {
                "name": "x",
                "type": "numeric",
                "lower": 0.0,
                "upper": 1.0,
                "fixed": False,
            }
        ],
        acquisition={
            "name": "EI",
            "beta": 2.0,
            "acqf_kwargs": {
                "web_family": "bayesian_optimization",
                "web_risk_type": "cvar",
                "web_risk_alpha": 0.5,
            },
        },
        optimizer={
            "name": "ga",
            "q": 1,
            "num_restarts": 1,
            "raw_samples": 8,
            "sequential": True,
        },
    )

    result = run_regression_web_workflow(request, store)

    assert len(result["candidates"]) == 1
    assert result["model_type"] == "random_forest"
    assert result["metadata"]["input_perturbation_risk_type"] == "cvar"
    assert result["metadata"]["input_perturbation_risk_enabled"] is True
