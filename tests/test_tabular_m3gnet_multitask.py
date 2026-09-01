from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from gpytorch.kernels import MultitaskKernel
from torch import Tensor, nn

from bochan.api import DataContext, MultiOutputConfig
from bochan.composition import M3GNetEncoder
from bochan.models.regression.gaussian.deep import (
    M3GNetMixedMultiTaskDKLModel,
    M3GNetMixedMultiTaskGPModel,
    M3GNetMultiTaskDKLModel,
    M3GNetMultiTaskGPModel,
)
from bochan.tabular import TabularBayesianOptimizer

pytest.importorskip("pymatgen")


class FakeGraph:
    def __init__(self, structure: Any) -> None:
        self.frac_coords = torch.as_tensor(structure.frac_coords, dtype=torch.float32)
        self.pbc_offset = torch.zeros((1, 3), dtype=torch.float32)
        self.pbc_offshift = torch.zeros((1, 3), dtype=torch.float32)
        self.pos = self.frac_coords.clone()

    def to(self, device: Any) -> FakeGraph:
        self.frac_coords = self.frac_coords.to(device)
        self.pbc_offset = self.pbc_offset.to(device)
        self.pbc_offshift = self.pbc_offshift.to(device)
        self.pos = self.pos.to(device)
        return self


class FakeGraphConverter:
    def get_graph(self, structure: Any) -> tuple[FakeGraph, Tensor, None]:
        lattice = torch.as_tensor(structure.lattice.matrix, dtype=torch.float32)
        return FakeGraph(structure), lattice, None


class FakeM3GNet(nn.Module):
    def __init__(self, output_dim: int = 4, n_blocks: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.is_intensive = True
        self.include_state = False
        self.embedding = nn.Linear(3, output_dim)
        self.graph_layers = nn.ModuleList(
            nn.Linear(output_dim, output_dim) for _ in range(n_blocks)
        )
        self.final_layer = nn.Linear(output_dim, 1)
        self.feature_dict: dict[str, Tensor] = {}

    def forward(self, g: FakeGraph, state_attr: Tensor | None = None) -> Tensor:
        assert state_attr is None
        features = torch.tanh(self.embedding(g.frac_coords.mean(dim=0)))
        for layer in self.graph_layers:
            features = features + torch.tanh(layer(features))
        self.feature_dict = {"readout": features.unsqueeze(0)}
        return self.final_layer(features).squeeze(-1)


def _material_encoder() -> M3GNetEncoder:
    return M3GNetEncoder(
        encoder=FakeM3GNet(),
        graph_converter=FakeGraphConverter(),
    )


def _structure(scale: float) -> dict[str, object]:
    return {
        "lattice_mat": [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
        "cartesian": False,
    }


def _catalog() -> dict[str, dict[str, object]]:
    return {
        "alpha": _structure(5.20),
        "beta": _structure(5.35),
        "gamma": _structure(5.50),
    }


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
    ("m3gnet_multitask", M3GNetMultiTaskGPModel, False, False),
    ("m3gnet_multitask_dkl", M3GNetMultiTaskDKLModel, False, True),
    ("m3gnet_multitask", M3GNetMixedMultiTaskGPModel, True, False),
    ("m3gnet_multitask_dkl", M3GNetMixedMultiTaskDKLModel, True, True),
]


def _optimizer(
    model_type: str,
    *,
    mixed: bool,
    dkl: bool,
    target_cols: list[str] | str | None = None,
) -> TabularBayesianOptimizer:
    input_cols = ["temperature", "phase", "pressure"]
    categorical_cols: list[str] = []
    if mixed:
        input_cols = ["temperature", "furnace", "phase", "pressure"]
        categorical_cols = ["furnace"]
    model_kwargs: dict[str, object] = {
        "encoder": _material_encoder(),
        "latent_dim": 3,
    }
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
        bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
        model_kwargs=model_kwargs,
        fit_config={"skip_fit": True},
    )


@pytest.mark.parametrize(
    ("model_type", "expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_m3gnet_multitask_family_builds_one_correlated_model(
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
    assert model.structures is optimizer.structure.structures

    encoder_parameters = list(model.material_encoder.parameters())
    assert encoder_parameters
    if dkl:
        assert any(parameter.requires_grad for parameter in encoder_parameters)
        assert not model.structure_feature_cache_enabled
        assert not model.material_encoder.encoder.final_layer.weight.requires_grad
    else:
        assert not any(parameter.requires_grad for parameter in encoder_parameters)
        assert model.structure_feature_cache_enabled

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
    assert torch.isfinite(gradient).all()
    continuous_dims = sorted(set(range(raw.shape[-1])) - set(bundle.cat_dims or []))
    continuous_process_dims = [index for index in continuous_dims if index != 0]
    assert gradient[:, continuous_process_dims].abs().sum() > 0

    subset = model.posterior(raw.detach(), output_indices=[1])
    assert subset.mean.shape == torch.Size([2, 1])
    torch.testing.assert_close(subset.mean, posterior.mean.detach()[..., 1:2])
    torch.testing.assert_close(subset.variance, posterior.variance.detach()[..., 1:2])


@pytest.mark.parametrize(
    ("model_type", "_expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_m3gnet_multitask_family_optimizes_multiobjective_nehvi(
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


def test_m3gnet_multitask_rejects_single_target_and_points_to_independent_model() -> None:
    with pytest.raises(ValueError, match="Use model_type='m3gnet_gp'"):
        _optimizer(
            "m3gnet_multitask",
            mixed=False,
            dkl=False,
            target_cols="strength",
        )


def test_m3gnet_multitask_rejects_explicit_multi_output_config() -> None:
    with pytest.raises(ValueError, match="keep wide targets in one model"):
        TabularBayesianOptimizer(
            task_type="multi_objective",
            model_type="m3gnet_multitask",
            multi_output_config=MultiOutputConfig(
                output_names=["strength", "conductivity"]
            ),
            input_cols=["temperature", "phase", "pressure"],
            target_cols=["strength", "conductivity"],
            structure_col="phase",
            structure_catalog=_catalog(),
            bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
            model_kwargs={"encoder": _material_encoder(), "latent_dim": 3},
            fit_config={"skip_fit": True},
        ).fit(_frame(mixed=False))


def test_m3gnet_multitask_array_fit_uses_wide_y_and_default_names() -> None:
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
        model_type="m3gnet_multitask",
        input_cols=[0, 1, 2],
        structure_col=0,
        structure_catalog={
            0: _structure(5.20),
            1: _structure(5.35),
            2: _structure(5.50),
        },
        bounds={1: [850.0, 1200.0], 2: [0.5, 2.0]},
        model_kwargs={"encoder": _material_encoder(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    ).fit(X, y)
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, M3GNetMultiTaskGPModel)
    assert bundle.model.num_outputs == 2
    assert optimizer.dataset.target_names == ["y0", "y1"]
    assert bundle.model_config.multi_output_config is None
