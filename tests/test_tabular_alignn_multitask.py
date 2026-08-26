from __future__ import annotations

import pandas as pd
import pytest
import torch
from gpytorch.kernels import MultitaskKernel
from torch import Tensor, nn

from bochan.api import DataContext, MultiOutputConfig
from bochan.models.regression.gaussian.deep import (
    ALIGNNMixedMultiTaskDKLModel,
    ALIGNNMixedMultiTaskGPModel,
    ALIGNNMultiTaskDKLModel,
    ALIGNNMultiTaskGPModel,
)
from bochan.tabular import TabularBayesianOptimizer


class FakeALIGNN(nn.Module):
    """Small injected ALIGNN-like encoder for dependency-free tests."""

    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_projection = nn.Linear(3, output_dim)
        self.alignn_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.gcn_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.double()

    def encode(self, graph: Tensor) -> Tensor:
        x = torch.tanh(self.input_projection(graph))
        for layer in self.alignn_layers:
            x = torch.tanh(layer(x))
        for layer in self.gcn_layers:
            x = torch.tanh(layer(x))
        return x


class FakeGraphBuilder:
    def build_many(self, structures: tuple[object, ...]) -> tuple[Tensor, ...]:
        return tuple(
            torch.tensor(
                [1.0 + index, 0.2 + 0.1 * index, 0.1 + 0.2 * index],
                dtype=torch.double,
            )
            for index, _ in enumerate(structures)
        )


def _catalog() -> dict[str, object]:
    return {"alpha": object(), "beta": object(), "gamma": object()}


def _frame(*, mixed: bool) -> pd.DataFrame:
    data: dict[str, object] = {
        "phase": ["alpha", "beta", "gamma", "alpha", "beta", "gamma", "alpha", "beta"],
        "temperature": [900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 925.0, 1125.0],
        "pressure": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 0.9, 1.7],
        "strength": [100.0, 115.0, 123.0, 132.0, 141.0, 150.0, 108.0, 145.0],
        "conductivity": [2.1, 2.4, 2.2, 2.7, 2.6, 3.0, 2.3, 2.8],
    }
    if mixed:
        data["furnace"] = ["A", "B", "A", "B", "A", "B", "A", "B"]
    return pd.DataFrame(data)


MODEL_CASES = [
    ("alignn_multitask", ALIGNNMultiTaskGPModel, False, False),
    ("alignn_multitask_dkl", ALIGNNMultiTaskDKLModel, False, True),
    ("alignn_multitask", ALIGNNMixedMultiTaskGPModel, True, False),
    ("alignn_multitask_dkl", ALIGNNMixedMultiTaskDKLModel, True, True),
]


def _optimizer(
    model_type: str,
    *,
    mixed: bool,
    dkl: bool,
    target_cols: list[str] | str | None = None,
    cross_validation: bool = False,
) -> TabularBayesianOptimizer:
    input_cols = ["temperature", "phase", "pressure"]
    categorical_cols: list[str] = []
    if mixed:
        input_cols = ["temperature", "furnace", "phase", "pressure"]
        categorical_cols = ["furnace"]
    model_kwargs: dict[str, object] = {"encoder": FakeALIGNN(), "latent_dim": 3}
    if dkl:
        model_kwargs["encoder_training"] = "partial"
    return TabularBayesianOptimizer(
        task_type="multi_objective",
        model_type=model_type,
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        target_cols=target_cols or ["strength", "conductivity"],
        structure_col="phase",
        structure_catalog=_catalog(),
        structure_graph_builder=FakeGraphBuilder(),
        bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
        model_kwargs=model_kwargs,
        fit_config={"skip_fit": True},
        cross_validation=cross_validation,
        cv_config={"splitter": "kfold", "n_splits": 2, "shuffle": False},
    )


@pytest.mark.parametrize(
    ("model_type", "expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_alignn_multitask_family_builds_one_correlated_model(
    model_type: str,
    expected_cls: type,
    mixed: bool,
    dkl: bool,
) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(model_type, mixed=mixed, dkl=dkl).fit(_frame(mixed=mixed))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    model = bundle.model
    assert isinstance(model, expected_cls)
    assert bundle.model_type == model_type
    assert bundle.task_type == "multi_objective"
    assert bundle.model_config.multi_output_config is None
    assert model.num_outputs == 2
    assert model.num_tasks == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    assert model.task_covar_module is model.deepkernel.covar_module.task_covar_module
    assert model.task_kernel is model.task_covar_module
    assert model.structure_graphs is optimizer.structure.structure_graphs

    encoder_parameters = list(model.material_encoder.parameters())
    assert encoder_parameters
    if dkl:
        assert any(parameter.requires_grad for parameter in encoder_parameters)
        assert model.structure_feature_cache_enabled is False
    else:
        assert not any(parameter.requires_grad for parameter in encoder_parameters)
        assert model.structure_feature_cache_enabled is True

    if mixed:
        assert bundle.input_type == "mixed"
        assert bundle.cat_dims == [2]
        assert model.cat_dims == [2]
        assert model.categorical_process_dim == 1
        assert optimizer.dataset.cat_dims == [0, 2]
        assert optimizer.dataset.category_maps["furnace"] == {"A": 0, "B": 1}
    else:
        assert bundle.input_type == "normal"
        assert not bundle.cat_dims

    raw = optimizer.dataset.X[:2].detach().clone().requires_grad_(True)
    posterior = model.posterior(raw)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), raw)

    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(gradient).all()
    continuous_dims = sorted(set(range(raw.shape[-1])) - set(bundle.cat_dims or []))
    continuous_process_dims = [index for index in continuous_dims if index != 0]
    assert continuous_process_dims
    assert gradient[:, continuous_process_dims].abs().sum() > 0


@pytest.mark.parametrize(
    ("model_type", "_expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_alignn_multitask_family_optimizes_multiobjective_nehvi(
    model_type: str,
    _expected_cls: type,
    mixed: bool,
    dkl: bool,
) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(model_type, mixed=mixed, dkl=dkl).fit(_frame(mixed=mixed))
    assert optimizer.train_X is not None
    assert optimizer.train_Y is not None

    ref_point = optimizer.train_Y.min(dim=0).values - 0.1
    candidates, acq_value = optimizer.candidate(
        acq_name="nehvi",
        q=1,
        objective_mode="multi_output",
        objective_outputs=["strength", "conductivity"],
        objective_directions=["maximize", "maximize"],
        data_context=DataContext(
            X_baseline=optimizer.train_X,
            Y_baseline=optimizer.train_Y,
            ref_point=ref_point,
        ),
        structure_ids=["alpha", "beta", "gamma"],
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    assert candidates.loc[0, "phase"] in {"alpha", "beta", "gamma"}
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1200.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    if mixed:
        assert candidates.loc[0, "furnace"] in {"A", "B"}
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_alignn_multitask_rejects_single_target_and_points_to_independent_model() -> None:
    with pytest.raises(ValueError, match="Use model_type='alignn_gp'"):
        _optimizer(
            "alignn_multitask",
            mixed=False,
            dkl=False,
            target_cols="strength",
        )


def test_alignn_multitask_rejects_explicit_multi_output_config() -> None:
    with pytest.raises(ValueError, match="keep wide targets in one model"):
        TabularBayesianOptimizer(
            task_type="multi_objective",
            model_type="alignn_multitask",
            multi_output_config=MultiOutputConfig(output_names=["strength", "conductivity"]),
            input_cols=["temperature", "phase", "pressure"],
            target_cols=["strength", "conductivity"],
            structure_col="phase",
            structure_catalog=_catalog(),
            structure_graph_builder=FakeGraphBuilder(),
            bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
            model_kwargs={"encoder": FakeALIGNN(), "latent_dim": 3},
            fit_config={"skip_fit": True},
        ).fit(_frame(mixed=False))


def test_alignn_multitask_array_fit_uses_wide_y_and_default_names() -> None:
    X = [
        [0.0, 900.0, 0.8],
        [1.0, 950.0, 1.0],
        [2.0, 1000.0, 1.2],
        [0.0, 1050.0, 1.4],
        [1.0, 1100.0, 1.6],
        [2.0, 1150.0, 1.8],
    ]
    y = [
        [100.0, 2.1],
        [115.0, 2.4],
        [123.0, 2.2],
        [132.0, 2.7],
        [141.0, 2.6],
        [150.0, 3.0],
    ]
    optimizer = TabularBayesianOptimizer(
        task_type="multi_objective",
        model_type="alignn_multitask",
        input_cols=[0, 1, 2],
        structure_col=0,
        structure_catalog={0: object(), 1: object(), 2: object()},
        structure_graph_builder=FakeGraphBuilder(),
        bounds={1: [850.0, 1200.0], 2: [0.5, 2.0]},
        model_kwargs={"encoder": FakeALIGNN(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    ).fit(X, y)
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ALIGNNMultiTaskGPModel)
    assert bundle.model.num_outputs == 2
    assert optimizer.dataset.target_names == ["y0", "y1"]
    assert bundle.model_config.multi_output_config is None


def test_alignn_multitask_cross_validation_preserves_correlated_model() -> None:
    optimizer = _optimizer(
        "alignn_multitask",
        mixed=False,
        dkl=False,
        cross_validation=True,
    ).fit(_frame(mixed=False))

    result = optimizer.cross_validation_result_
    assert result is not None
    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, ALIGNNMultiTaskGPModel)
    assert isinstance(bundle.model.deepkernel.covar_module, MultitaskKernel)
    assert bundle.model_config.multi_output_config is None
