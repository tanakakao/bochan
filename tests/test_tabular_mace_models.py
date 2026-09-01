from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import MultitaskKernel
from torch import Tensor, nn

pytest.importorskip("mace")

from bochan.api import ModelConfig
from bochan.api.acquisition.defaults import resolve_multi_output_model_config
from bochan.composition import MACEEncoder
from bochan.models.regression.gaussian.deep import (
    MACEDKLModel,
    MACEGPModel,
    MACEMixedGPModel,
    MACEMixedMultiTaskDKLModel,
    MACEMixedMultiTaskGPModel,
    MACEMultiTaskDKLModel,
    MACEMultiTaskGPModel,
)
from bochan.tabular import TabularBayesianOptimizer


class FakeDescriptorLinear(nn.Linear):
    def __init__(self, width: int) -> None:
        super().__init__(width, width, bias=False)
        self.irreps_out = f"{width}x0e + {width}x1o"


class FakeProduct(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = FakeDescriptorLinear(width)
        self.scale = nn.Parameter(torch.ones(()))


class FakeMACE(nn.Module):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor([14], dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(5.0, dtype=torch.float32))
        self.register_buffer("num_interactions", torch.tensor(2, dtype=torch.int64))
        self.heads = ["Default"]
        self.node_embedding = nn.Linear(3, width, bias=False)
        self.radial_embedding = nn.Linear(1, width, bias=False)
        self.spherical_harmonics = nn.Identity()
        self.interactions = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(2)]
        )
        self.products = nn.ModuleList([FakeProduct(width) for _ in range(2)])
        self.readouts = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])

    def forward(
        self,
        data: dict[str, Tensor],
        *,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> dict[str, Tensor]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        positions = data["positions"]
        first = self.products[0].scale * torch.tanh(self.node_embedding(positions))
        equivariant = torch.cat([positions, positions], dim=-1)
        final = self.products[1].scale * torch.tanh(self.interactions[-1](first))
        node_feats = torch.cat([first, equivariant, final], dim=-1)
        return {
            "node_feats": node_feats,
            "energy": self.readouts[-1](final).sum(),
        }


class BatchBuilder:
    def __call__(self, structure: dict[str, object]) -> dict[str, Tensor]:
        lattice = torch.tensor(structure["lattice_mat"], dtype=torch.float32)
        coords = torch.tensor(structure["coords"], dtype=torch.float32)
        positions = coords if bool(structure.get("cartesian", False)) else coords @ lattice
        return {"positions": positions}


def _material_encoder() -> MACEEncoder:
    return MACEEncoder(FakeMACE(), batch_builder=BatchBuilder())


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


def _frame(*, mixed: bool = False, multi_output: bool = False) -> pd.DataFrame:
    data: dict[str, object] = {
        "phase": ["alpha", "beta", "gamma", "alpha", "beta", "gamma"],
        "temperature": [900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0],
        "pressure": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        "property": [0.4, 0.8, 1.1, 0.9, 1.4, 1.8],
    }
    if mixed:
        data["furnace"] = ["A", "B", "A", "B", "A", "B"]
    if multi_output:
        data["strength"] = [100.0, 115.0, 123.0, 132.0, 141.0, 150.0]
        data["conductivity"] = [2.1, 2.4, 2.2, 2.7, 2.6, 3.0]
    return pd.DataFrame(data)


def _optimizer(
    model_type: str = "mace_gp",
    *,
    mixed: bool = False,
    multi_output: bool = False,
    encoder_training: str | None = None,
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
    if encoder_training is not None:
        model_kwargs["encoder_training"] = encoder_training
    target_cols: str | list[str] = "property"
    if multi_output:
        target_cols = ["strength", "conductivity"]
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type=model_type,
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        target_cols=target_cols,
        structure_col="phase",
        structure_catalog=_catalog(),
        bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
        model_kwargs=model_kwargs,
        fit_config={"skip_fit": True},
    )


def test_tabular_mace_gp_builds_structure_first_contract() -> None:
    optimizer = _optimizer().fit(_frame())
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, MACEGPModel)
    assert optimizer.dataset.feature_names == ["phase", "temperature", "pressure"]
    assert optimizer.dataset.cat_dims == [0]
    assert optimizer.model_config.cat_dims == []
    assert optimizer.model_config.pass_cat_dims is False
    assert optimizer.structure.structure_ids == ("alpha", "beta", "gamma")
    assert bundle.model.structures == optimizer.structure.structures
    torch.testing.assert_close(
        optimizer.dataset.X[:, 0],
        torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.double),
    )


def test_tabular_mace_mixed_resolves_process_categories() -> None:
    optimizer = _optimizer(mixed=True).fit(_frame(mixed=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, MACEMixedGPModel)
    assert bundle.input_type == "mixed"
    assert bundle.cat_dims == [2]
    assert bundle.model.cat_dims == [2]
    assert bundle.model.continuous_process_dims == (1, 3)
    assert optimizer.dataset.feature_names == [
        "phase",
        "temperature",
        "furnace",
        "pressure",
    ]
    assert optimizer.dataset.cat_dims == [0, 2]
    assert optimizer.dataset.category_maps["furnace"] == {"A": 0, "B": 1}


def test_tabular_mace_dkl_maps_encoder_training_policy() -> None:
    optimizer = _optimizer(
        "mace_dkl",
        encoder_training="partial",
    ).fit(_frame())
    model = optimizer.bo.bundle.model

    assert isinstance(model, MACEDKLModel)
    assert model.trainable_encoder_layers == 1
    assert not model.structure_feature_cache_enabled
    assert model.material_encoder.encoder.products[-1].scale.requires_grad
    assert not model.material_encoder.encoder.readouts[-1].weight.requires_grad


def test_mace_gp_multi_output_auto_builds_independent_model_list() -> None:
    optimizer = _optimizer(multi_output=True).fit(_frame(multi_output=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert bundle.task_type == "multi_objective"
    assert bundle.model_config.multi_output_config.output_names == [
        "strength",
        "conductivity",
    ]
    assert len(bundle.model.models) == 2
    assert all(isinstance(model, MACEGPModel) for model in bundle.model.models)
    assert bundle.model.models[0].material_encoder is not bundle.model.models[1].material_encoder
    assert bundle.model.models[0].structures is bundle.model.models[1].structures


def test_mace_mixed_multi_output_uses_mixed_submodels() -> None:
    optimizer = _optimizer(mixed=True, multi_output=True).fit(
        _frame(mixed=True, multi_output=True)
    )
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert all(isinstance(model, MACEMixedGPModel) for model in bundle.model.models)
    assert all(model.cat_dims == [2] for model in bundle.model.models)


@pytest.mark.parametrize(
    ("model_type", "mixed", "expected_type"),
    [
        ("mace_multitask", False, MACEMultiTaskGPModel),
        ("mace_multitask", True, MACEMixedMultiTaskGPModel),
        ("mace_multitask_dkl", False, MACEMultiTaskDKLModel),
        ("mace_multitask_dkl", True, MACEMixedMultiTaskDKLModel),
    ],
)
def test_mace_correlated_multitask_tabular_routing(
    model_type: str,
    mixed: bool,
    expected_type: type[nn.Module],
) -> None:
    encoder_training = "partial" if model_type.endswith("_dkl") else None
    optimizer = _optimizer(
        model_type,
        mixed=mixed,
        multi_output=True,
        encoder_training=encoder_training,
    ).fit(_frame(mixed=mixed, multi_output=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, expected_type)
    assert bundle.task_type == "multi_objective"
    assert bundle.model_config.multi_output_config is None
    assert bundle.model.num_outputs == 2
    assert isinstance(bundle.model.deepkernel.covar_module, MultitaskKernel)
    if mixed:
        assert bundle.cat_dims == [2]
        assert bundle.model.cat_dims == [2]
    if model_type.endswith("_dkl"):
        assert bundle.model.trainable_encoder_layers == 1
        assert not bundle.model.structure_feature_cache_enabled


def test_high_level_resolver_keeps_mace_multitask_wide() -> None:
    config = ModelConfig(
        task_type="multi_objective",
        model_type="mace_multitask",
    )
    train_Y = torch.zeros(5, 2, dtype=torch.double)

    resolved = resolve_multi_output_model_config(config, train_Y)

    assert resolved is config
    assert resolved.multi_output_config is None


def test_mace_structure_candidate_enumeration_keeps_process_categories(
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
        acq_name="logei",
        q=1,
        structure_ids=["beta", "gamma"],
    )

    opt_config = captured["opt_config"]
    assert opt_config.fixed_features_list is not None
    assert {entry[0] for entry in opt_config.fixed_features_list} == {1.0, 2.0}
    assert {entry[2] for entry in opt_config.fixed_features_list} == {0.0, 1.0}
    assert candidates.loc[0, "phase"] == "beta"
    assert candidates.loc[0, "furnace"] == "B"
    assert float(acq_value) == pytest.approx(0.8)


def test_mace_multitask_rejects_single_target() -> None:
    with pytest.raises(ValueError, match="Use model_type='mace_gp'"):
        _optimizer("mace_multitask").fit(_frame())


def test_mace_frozen_model_rejects_encoder_training() -> None:
    with pytest.raises(ValueError, match="freezes the MACE structure encoder"):
        _optimizer("mace_gp", encoder_training="partial").fit(_frame())
