from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

import bochan.fit  # noqa: F401
from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    DataContext,
    FitConfig,
    ModelBundle,
    ModelConfig,
    MultiOutputConfig,
)
import bochan.api.engine_defaults as engine_defaults
from bochan.models.regression.gaussian.high_dim import (
    SaasGaussianMixedGPModel,
    SaasGaussianGPModel,
)
from bochan.models.regression.gaussian.kronecker_multitask import (
    GaussianMixedKroneckerMultiTaskGP,
    GaussianKroneckerMultiTaskGP,
)
from bochan.models.regression.gaussian.multifidelity import (
    WideMixedMultiFidelityGP,
    WideMultiFidelityGP,
)
from bochan.models.regression.gaussian.robust import (
    HeteroscedasticGaussianMixedGPModel,
    HeteroscedasticGaussianGPModel,
    RobustRelevancePursuitGaussianMixedGPModel,
    RobustRelevancePursuitGaussianGPModel,
)
from bochan.models.multitask.wide import WideMultiTaskGP


class _NeedsBestF:
    def __init__(self, model, best_f) -> None:
        self.model = model
        self.best_f = best_f


class _NeedsEHVIContext:
    def __init__(self, model, ref_point, partitioning) -> None:
        self.model = model
        self.ref_point = ref_point
        self.partitioning = partitioning


class _NeedsRefPoint:
    def __init__(self, model, ref_point, X_baseline=None) -> None:
        self.model = model
        self.ref_point = ref_point
        self.X_baseline = X_baseline


def _make_bundle(
    *,
    task_type: str = "regression",
    train_Y: torch.Tensor | None = None,
) -> ModelBundle:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    if train_Y is None:
        train_Y = torch.tensor([[1.0], [3.0], [2.0]], dtype=torch.double)
    config = ModelConfig(
        task_type=task_type,
        model_type="base",
        outcome_transform=False,
    )
    return ModelBundle(
        model=SimpleNamespace(),
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        task_type=task_type,
        model_type="base",
    )


def test_two_column_train_y_enables_empty_multi_output_config() -> None:
    config = ModelConfig(
        task_type="regression",
        model_type="base",
        outcome_transform=False,
    )
    train_Y = torch.zeros(4, 2, dtype=torch.double)

    resolved = engine_defaults.resolve_multi_output_model_config(config, train_Y)

    assert config.multi_output_config is None
    assert isinstance(resolved.multi_output_config, MultiOutputConfig)


def test_explicit_multi_output_config_is_preserved() -> None:
    explicit = MultiOutputConfig(output_names=["a", "b"])
    config = ModelConfig(
        task_type="regression",
        model_type="base",
        outcome_transform=False,
        multi_output_config=explicit,
    )

    resolved = engine_defaults.resolve_multi_output_model_config(
        config,
        torch.zeros(3, 2, dtype=torch.double),
    )

    assert resolved is config
    assert resolved.multi_output_config is explicit


def test_bayesian_optimizer_fit_uses_automatic_multi_output_config(monkeypatch) -> None:
    captured = {}

    def fake_build_model(*, train_X, train_Y, config, model_registry=None):
        captured["config"] = config
        return ModelBundle(
            model=SimpleNamespace(),
            train_X=train_X,
            train_Y=train_Y,
            model_config=config,
            task_type=str(config.task_type),
            model_type=str(config.model_type),
            metadata={"multi_output": config.multi_output_config is not None},
        )

    monkeypatch.setattr(
        "bochan.api.optimizer._build_partial_objective_bundle",
        fake_build_model,
    )
    monkeypatch.setattr(
        "bochan.api.optimizer.fit_model",
        lambda bundle, config: bundle,
    )

    optimizer = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )
    optimizer.fit(
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.double),
    )

    assert isinstance(captured["config"].multi_output_config, MultiOutputConfig)
    assert isinstance(optimizer.model_config.multi_output_config, MultiOutputConfig)


def test_regression_ei_computes_best_f_from_observed_values() -> None:
    bundle = _make_bundle()
    config = AcquisitionConfig(name="ei", acqf_cls=_NeedsBestF)

    context = engine_defaults.resolve_acquisition_data_context(
        bundle,
        config,
        DataContext(),
    )

    torch.testing.assert_close(context.best_f, torch.tensor(3.0, dtype=torch.double))


def test_binary_ei_uses_binary_best_f_helper(monkeypatch) -> None:
    bundle = _make_bundle(
        task_type="binary",
        train_Y=torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.double),
    )
    expected = torch.tensor(0.73, dtype=torch.double)
    calls = []

    def fake_compute_binary_best_f(model, train_X, **kwargs):
        calls.append((model, train_X, kwargs))
        return expected

    monkeypatch.setattr(
        "bochan.acquisition.binary.bayesian_optimization.compute_binary_best_f",
        fake_compute_binary_best_f,
    )

    context = engine_defaults.resolve_acquisition_data_context(
        bundle,
        AcquisitionConfig(name="pi", acqf_cls=_NeedsBestF),
        DataContext(),
    )

    assert len(calls) == 1
    assert context.best_f is expected


def test_explicit_best_f_is_not_overwritten(monkeypatch) -> None:
    bundle = _make_bundle()
    explicit = torch.tensor(9.0, dtype=torch.double)
    monkeypatch.setattr(
        engine_defaults,
        "compute_best_f",
        lambda *args, **kwargs: pytest.fail("automatic best_f must not run"),
    )

    context = engine_defaults.resolve_acquisition_data_context(
        bundle,
        AcquisitionConfig(name="ei", acqf_cls=_NeedsBestF),
        DataContext(best_f=explicit),
    )

    assert context.best_f is explicit


def test_ehvi_computes_ref_point_and_partitioning(monkeypatch) -> None:
    train_Y = torch.tensor(
        [[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]],
        dtype=torch.double,
    )
    bundle = _make_bundle(train_Y=train_Y)
    sentinel = object()
    captured = {}

    def fake_partitioning(ref_point, values):
        captured["ref_point"] = ref_point
        captured["values"] = values
        return sentinel

    monkeypatch.setattr(engine_defaults, "make_partitioning", fake_partitioning)

    context = engine_defaults.resolve_acquisition_data_context(
        bundle,
        AcquisitionConfig(name="ehi", acqf_cls=_NeedsEHVIContext),
        DataContext(),
    )

    torch.testing.assert_close(
        context.ref_point,
        torch.tensor([-0.1, 1.9], dtype=torch.double),
    )
    torch.testing.assert_close(captured["values"], train_Y)
    assert captured["ref_point"] is context.ref_point
    assert context.partitioning is sentinel


@pytest.mark.parametrize("name", ["nehvi", "nehi", "nparego"])
def test_nehvi_and_nparego_compute_ref_point_without_partitioning(name: str) -> None:
    train_Y = torch.tensor(
        [[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]],
        dtype=torch.double,
    )
    bundle = _make_bundle(train_Y=train_Y)

    context = engine_defaults.resolve_acquisition_data_context(
        bundle,
        AcquisitionConfig(name=name, acqf_cls=_NeedsRefPoint),
        DataContext(),
    )

    torch.testing.assert_close(
        context.ref_point,
        torch.tensor([-0.1, 1.9], dtype=torch.double),
    )
    assert context.partitioning is None


def test_explicit_ref_point_and_partitioning_are_preserved(monkeypatch) -> None:
    bundle = _make_bundle(
        train_Y=torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.double)
    )
    ref_point = torch.tensor([-5.0, -6.0], dtype=torch.double)
    partitioning = object()
    monkeypatch.setattr(
        engine_defaults,
        "observed_multiobjective_values",
        lambda *args, **kwargs: pytest.fail(
            "automatic multi-objective defaults must not run"
        ),
    )

    context = engine_defaults.resolve_acquisition_data_context(
        bundle,
        AcquisitionConfig(name="ehi", acqf_cls=_NeedsEHVIContext),
        DataContext(ref_point=ref_point, partitioning=partitioning),
    )

    assert context.ref_point is ref_point
    assert context.partitioning is partitioning


class _VariadicEHVI:
    def __init__(self, model, *args, **kwargs) -> None:
        self.model = model
        self.args = args
        self.kwargs = kwargs


def test_variadic_wrapper_receives_defaults_through_acqf_kwargs(monkeypatch) -> None:
    train_Y = torch.tensor(
        [[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]],
        dtype=torch.double,
    )
    bundle = _make_bundle(train_Y=train_Y)
    partitioning = object()
    monkeypatch.setattr(
        engine_defaults,
        "make_partitioning",
        lambda ref_point, values: partitioning,
    )

    resolved, context = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(name="ehi", acqf_cls=_VariadicEHVI),
        DataContext(),
    )

    torch.testing.assert_close(
        resolved.acqf_kwargs["ref_point"],
        torch.tensor([-0.1, 1.9], dtype=torch.double),
    )
    assert resolved.acqf_kwargs["partitioning"] is partitioning
    assert context.ref_point is None
    assert context.partitioning is None


def test_acqf_kwargs_best_f_has_priority_over_data_context(monkeypatch) -> None:
    bundle = _make_bundle()
    explicit = torch.tensor(7.0, dtype=torch.double)
    monkeypatch.setattr(
        engine_defaults,
        "compute_best_f",
        lambda *args, **kwargs: pytest.fail("automatic best_f must not run"),
    )

    resolved, context = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="ei",
            acqf_cls=_NeedsBestF,
            acqf_kwargs={"best_f": explicit},
        ),
        DataContext(best_f=torch.tensor(9.0, dtype=torch.double)),
    )

    assert resolved.acqf_kwargs["best_f"] is explicit
    assert context.best_f is None


@pytest.mark.parametrize(
    "model_cls",
    [
        SingleTaskGP,
        MixedSingleTaskGP,
        GaussianKroneckerMultiTaskGP,
        GaussianMixedKroneckerMultiTaskGP,
        WideMultiTaskGP,
        WideMultiFidelityGP,
        WideMixedMultiFidelityGP,
        SaasGaussianGPModel,
        SaasGaussianMixedGPModel,
        RobustRelevancePursuitGaussianGPModel,
        RobustRelevancePursuitGaussianMixedGPModel,
        HeteroscedasticGaussianGPModel,
        HeteroscedasticGaussianMixedGPModel,
    ],
)
def test_exact_regression_models_define_make_mll_directly(model_cls) -> None:
    assert "make_mll" in model_cls.__dict__
    assert callable(model_cls.__dict__["make_mll"])


def test_single_task_make_mll_returns_exact_marginal_log_likelihood() -> None:
    train_X = torch.linspace(0.0, 1.0, 4, dtype=torch.double).unsqueeze(-1)
    train_Y = train_X.square()
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)

    mll = model.make_mll()

    assert isinstance(mll, ExactMarginalLogLikelihood)
    assert mll.model is model
    assert mll.likelihood is model.likelihood
