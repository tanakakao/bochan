from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import pytest
import torch
from gpytorch.kernels import MultitaskKernel
from torch import Tensor, nn

from bochan.api import DataContext, MultiOutputConfig
from bochan.models.regression.gaussian.deep import (
    CHGNetMixedMultiTaskDKLModel,
    CHGNetMixedMultiTaskGPModel,
    CHGNetMultiTaskDKLModel,
    CHGNetMultiTaskGPModel,
)
from bochan.tabular import TabularBayesianOptimizer

pytest.importorskip("pymatgen")


class FakeCrystalGraph:
    def __init__(self, structure: Any) -> None:
        self.lattice = torch.tensor(
            [
                float(structure.lattice.a) / 10.0,
                float(len(structure)) / 10.0,
                float(structure.frac_coords.sum()) / max(len(structure), 1),
            ],
            dtype=torch.float32,
        )

    def to(self, device: str = "cpu") -> FakeCrystalGraph:
        self.lattice = self.lattice.to(device)
        return self


class FakeGraphConverter:
    def __call__(self, structure: Any) -> FakeCrystalGraph:
        return FakeCrystalGraph(structure)


class FakeCHGNet(nn.Module):
    def __init__(self, output_dim: int = 4, n_conv: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.atom_embedding = nn.Linear(3, output_dim)
        self.atom_conv_layers = nn.ModuleList(
            nn.Linear(output_dim, output_dim) for _ in range(n_conv)
        )
        self.mlp = nn.Linear(output_dim, 1)
        self.graph_converter = FakeGraphConverter()

    def forward(
        self,
        graphs: Sequence[FakeCrystalGraph],
        *,
        task: str = "e",
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor]:
        assert task == "e"
        features = torch.stack([graph.lattice for graph in graphs])
        features = torch.tanh(self.atom_embedding(features))
        for layer in self.atom_conv_layers:
            features = features + torch.tanh(layer(features))
        result = {"e": self.mlp(features).squeeze(-1)}
        if return_crystal_feas:
            result["crystal_fea"] = features
        return result


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
    ("chgnet_multitask", CHGNetMultiTaskGPModel, False, False),
    ("chgnet_multitask_dkl", CHGNetMultiTaskDKLModel, False, True),
    ("chgnet_multitask", CHGNetMixedMultiTaskGPModel, True, False),
    ("chgnet_multitask_dkl", CHGNetMixedMultiTaskDKLModel, True, True),
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
    model_kwargs: dict[str, object] = {"encoder": FakeCHGNet(), "latent_dim": 3}
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
def test_chgnet_multitask_family_builds_one_correlated_model(
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
def test_chgnet_multitask_family_optimizes_multiobjective_nehvi(
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


def test_chgnet_multitask_rejects_single_target_and_points_to_independent_model() -> None:
    with pytest.raises(ValueError, match="Use model_type='chgnet_gp'"):
        _optimizer(
            "chgnet_multitask",
            mixed=False,
            dkl=False,
            target_cols="strength",
        )


def test_chgnet_multitask_rejects_explicit_multi_output_config() -> None:
    with pytest.raises(ValueError, match="keep wide targets in one model"):
        TabularBayesianOptimizer(
            task_type="multi_objective",
            model_type="chgnet_multitask",
            multi_output_config=MultiOutputConfig(output_names=["strength", "conductivity"]),
            input_cols=["temperature", "phase", "pressure"],
            target_cols=["strength", "conductivity"],
            structure_col="phase",
            structure_catalog=_catalog(),
            bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
            model_kwargs={"encoder": FakeCHGNet(), "latent_dim": 3},
            fit_config={"skip_fit": True},
        ).fit(_frame(mixed=False))


def test_chgnet_multitask_array_fit_uses_wide_y_and_default_names() -> None:
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
        model_type="chgnet_multitask",
        input_cols=[0, 1, 2],
        structure_col=0,
        structure_catalog={0: _structure(5.20), 1: _structure(5.35), 2: _structure(5.50)},
        bounds={1: [850.0, 1200.0], 2: [0.5, 2.0]},
        model_kwargs={"encoder": FakeCHGNet(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    ).fit(X, y)
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, CHGNetMultiTaskGPModel)
    assert bundle.model.num_outputs == 2
    assert optimizer.dataset.target_names == ["y0", "y1"]
    assert bundle.model_config.multi_output_config is None
