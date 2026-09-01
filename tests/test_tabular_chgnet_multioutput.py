from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import pytest
import torch
from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor, nn

from bochan.api import MultiOutputConfig
from bochan.models.regression.gaussian.deep import (
    CHGNetDKLModel,
    CHGNetGPModel,
    CHGNetMixedDKLModel,
    CHGNetMixedGPModel,
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
        "phase": ["alpha", "beta", "gamma", "alpha", "beta", "gamma"],
        "temperature": [900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0],
        "pressure": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        "strength": [100.0, 115.0, 123.0, 132.0, 141.0, 150.0],
        "conductivity": [2.1, 2.4, 2.2, 2.7, 2.6, 3.0],
    }
    if mixed:
        data["furnace"] = ["A", "B", "A", "B", "A", "B"]
    return pd.DataFrame(data)


def _optimizer(
    model_type: str = "chgnet_gp",
    *,
    mixed: bool = False,
) -> TabularBayesianOptimizer:
    input_cols = ["temperature", "phase", "pressure"]
    categorical_cols: list[str] = []
    if mixed:
        input_cols = ["temperature", "furnace", "phase", "pressure"]
        categorical_cols = ["furnace"]
    model_kwargs: dict[str, object] = {"encoder": FakeCHGNet(), "latent_dim": 3}
    if model_type == "chgnet_dkl":
        model_kwargs["encoder_training"] = "partial"
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type=model_type,
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        target_cols=["strength", "conductivity"],
        structure_col="phase",
        structure_catalog=_catalog(),
        bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
        model_kwargs=model_kwargs,
        fit_config={"skip_fit": True},
    )


def test_chgnet_gp_multi_output_auto_builds_independent_model_list() -> None:
    optimizer = _optimizer().fit(_frame(mixed=False))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert bundle.task_type == "multi_objective"
    assert bundle.model_config.multi_output_config is not None
    assert bundle.model_config.multi_output_config.output_names == [
        "strength",
        "conductivity",
    ]
    assert len(bundle.model.models) == 2
    assert all(isinstance(model, CHGNetGPModel) for model in bundle.model.models)
    assert bundle.model.models[0].material_encoder is not bundle.model.models[1].material_encoder
    assert bundle.model.models[0].structures is bundle.model.models[1].structures

    X, _ = optimizer._prediction_input(
        pd.DataFrame(
            {
                "phase": ["alpha", "gamma"],
                "temperature": [975.0, 1075.0],
                "pressure": [1.1, 1.5],
            }
        )
    )
    posterior = bundle.model.posterior(X)
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])


def test_chgnet_mixed_multi_output_preserves_process_categories() -> None:
    optimizer = _optimizer(mixed=True).fit(_frame(mixed=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert bundle.input_type == "mixed"
    assert bundle.cat_dims == [2]
    assert all(isinstance(model, CHGNetMixedGPModel) for model in bundle.model.models)
    assert all(model.cat_dims == [2] for model in bundle.model.models)
    assert optimizer.dataset.feature_names == [
        "phase",
        "temperature",
        "furnace",
        "pressure",
    ]
    assert optimizer.dataset.cat_dims == [0, 2]
    assert optimizer.dataset.category_maps["furnace"] == {"A": 0, "B": 1}


def test_chgnet_dkl_multi_output_uses_independent_trainable_encoders() -> None:
    optimizer = _optimizer("chgnet_dkl").fit(_frame(mixed=False))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    first, second = bundle.model.models
    assert isinstance(first, CHGNetDKLModel)
    assert isinstance(second, CHGNetDKLModel)
    assert first.material_encoder is not second.material_encoder
    assert first.trainable_encoder_layers == 1
    assert second.trainable_encoder_layers == 1
    first_params = {id(parameter) for parameter in first.material_encoder.parameters()}
    second_params = {id(parameter) for parameter in second.material_encoder.parameters()}
    assert first_params.isdisjoint(second_params)
    assert any(parameter.requires_grad for parameter in first.material_encoder.parameters())
    assert any(parameter.requires_grad for parameter in second.material_encoder.parameters())


def test_chgnet_mixed_dkl_multi_output_uses_mixed_submodels() -> None:
    optimizer = _optimizer("chgnet_dkl", mixed=True).fit(_frame(mixed=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert all(isinstance(model, CHGNetMixedDKLModel) for model in bundle.model.models)
    assert all(model.cat_dims == [2] for model in bundle.model.models)


def test_chgnet_multi_output_candidate_keeps_structure_and_category_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = _optimizer(mixed=True).fit(_frame(mixed=True))
    captured: dict[str, object] = {}

    def fake_candidate(
        acq_config: Any,
        opt_config: Any,
        *,
        data_context: Any = None,
        bounds: Any = None,
        return_result: bool = False,
    ) -> tuple[Tensor, Tensor]:
        captured["opt_config"] = opt_config
        return (
            torch.tensor([[1.0, 1025.0, 1.0, 1.25]], dtype=torch.double),
            torch.tensor(0.8, dtype=torch.double),
        )

    monkeypatch.setattr(optimizer.bo, "candidate", fake_candidate)
    candidates, acq_value = optimizer.candidate(
        acq_name="nehvi",
        q=1,
        objective_mode="multi_output",
        objective_outputs=["strength", "conductivity"],
        objective_directions=["maximize", "maximize"],
        structure_ids=["beta", "gamma"],
    )

    opt_config = captured["opt_config"]
    assert opt_config.fixed_features_list is not None
    assert {entry[0] for entry in opt_config.fixed_features_list} == {1.0, 2.0}
    assert {entry[2] for entry in opt_config.fixed_features_list} == {0.0, 1.0}
    assert candidates.loc[0, "phase"] == "beta"
    assert candidates.loc[0, "furnace"] == "B"
    assert float(acq_value) == pytest.approx(0.8)


def test_chgnet_rejects_explicit_user_multi_output_config() -> None:
    with pytest.raises(ValueError, match="derives independent multi-output structure automatically"):
        TabularBayesianOptimizer(
            task_type="multi_objective",
            model_type="chgnet_gp",
            multi_output_config=MultiOutputConfig(output_names=["strength", "conductivity"]),
            input_cols=["temperature", "phase", "pressure"],
            target_cols=["strength", "conductivity"],
            structure_col="phase",
            structure_catalog=_catalog(),
            bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
            model_kwargs={"encoder": FakeCHGNet(), "latent_dim": 3},
            fit_config={"skip_fit": True},
        ).fit(_frame(mixed=False))


def test_chgnet_multi_output_array_fit_derives_default_output_names() -> None:
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
        task_type="regression",
        model_type="chgnet_gp",
        input_cols=[0, 1, 2],
        structure_col=0,
        structure_catalog={0: _structure(5.20), 1: _structure(5.35), 2: _structure(5.50)},
        bounds={1: [850.0, 1200.0], 2: [0.5, 2.0]},
        model_kwargs={"encoder": FakeCHGNet(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    ).fit(X, y)
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert optimizer.dataset.target_names == ["y0", "y1"]
    assert bundle.model_config.multi_output_config.output_names == ["y0", "y1"]
