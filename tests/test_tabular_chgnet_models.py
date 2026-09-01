from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from bochan.models.regression.gaussian.deep import (
    CHGNetDKLModel,
    CHGNetGPModel,
    CHGNetMixedDKLModel,
    CHGNetMixedGPModel,
)
from bochan.tabular import TabularBayesianOptimizer, TabularDataConfig

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
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, structure: Any) -> FakeCrystalGraph:
        self.calls += 1
        return FakeCrystalGraph(structure)


class FakeCHGNet(nn.Module):
    """Small differentiable CHGNet-like backbone for tabular tests."""

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


def _structure(scale: float, element: str = "Si") -> dict[str, object]:
    return {
        "lattice_mat": [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": [element, element],
        "cartesian": False,
    }


def _catalog() -> dict[str, dict[str, object]]:
    return {
        "alpha": _structure(5.20),
        "beta": _structure(5.35),
        "gamma": _structure(5.50),
    }


def _frame(*, mixed: bool = False) -> pd.DataFrame:
    data: dict[str, object] = {
        "phase": ["alpha", "beta", "gamma", "alpha", "beta", "gamma"],
        "temperature": [900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0],
        "pressure": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        "property": [0.4, 0.8, 1.1, 0.9, 1.4, 1.8],
    }
    if mixed:
        data["furnace"] = ["A", "B", "A", "B", "A", "A"]
        data["atmosphere"] = ["air", "N2", "air", "N2", "Ar", "Ar"]
    return pd.DataFrame(data)


def _optimizer(
    model_type: str = "chgnet_gp",
    *,
    mixed: bool = False,
    encoder_training: str | None = None,
    input_type: str | None = None,
) -> TabularBayesianOptimizer:
    model_kwargs: dict[str, object] = {
        "encoder": FakeCHGNet(),
        "latent_dim": 3,
    }
    if encoder_training is not None:
        model_kwargs["encoder_training"] = encoder_training
    model_config = None if input_type is None else {"input_type": input_type}
    input_cols = ["temperature", "phase", "pressure"]
    categorical_cols = None
    if mixed:
        input_cols = [
            "temperature",
            "furnace",
            "phase",
            "pressure",
            "atmosphere",
        ]
        categorical_cols = ["furnace", "atmosphere"]
    return TabularBayesianOptimizer(
        model_config=model_config,
        task_type="regression",
        model_type=model_type,
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        target_cols="property",
        structure_col="phase",
        structure_catalog=_catalog(),
        bounds={
            "temperature": [850.0, 1200.0],
            "pressure": [0.5, 2.0],
        },
        model_kwargs=model_kwargs,
        fit_config={"skip_fit": True},
    )


def test_tabular_chgnet_gp_builds_raw_structure_index_contract() -> None:
    optimizer = _optimizer().fit(_frame())
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, CHGNetGPModel)
    assert optimizer.dataset.feature_names == ["phase", "temperature", "pressure"]
    assert optimizer.dataset.cat_dims == [0]
    assert optimizer.model_config.cat_dims == []
    assert optimizer.model_config.pass_cat_dims is False
    assert optimizer.structure.structure_ids == ("alpha", "beta", "gamma")
    assert optimizer.structure.structures == tuple(_catalog().values())
    assert bundle.model.structures == optimizer.structure.structures
    assert optimizer.dataset.category_maps["phase"] == {
        "alpha": 0,
        "beta": 1,
        "gamma": 2,
    }
    torch.testing.assert_close(
        optimizer.dataset.X[:, 0],
        torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.double),
    )


def test_tabular_chgnet_defaults_deepkernel_fit_config() -> None:
    optimizer = TabularBayesianOptimizer(
        model_type="chgnet_gp",
        input_cols=["phase", "temperature", "pressure"],
        target_cols="property",
        structure_col="phase",
        structure_catalog=_catalog(),
        bounds={
            "temperature": [850.0, 1200.0],
            "pressure": [0.5, 2.0],
        },
        model_kwargs={"encoder": FakeCHGNet(), "latent_dim": 3},
    )

    assert optimizer.fit_config is not None
    assert optimizer.fit_config.fit_func is not None
    assert optimizer.fit_config.fit_func.__name__ == "fit_deepkernel_mll"


def test_tabular_chgnet_prediction_reuses_structure_catalog_mapping() -> None:
    optimizer = _optimizer().fit(_frame())

    X, index = optimizer._prediction_input(
        pd.DataFrame(
            {
                "phase": ["gamma", "alpha"],
                "temperature": [1025.0, 1075.0],
                "pressure": [1.3, 1.5],
            },
            index=[7, 8],
        )
    )

    assert index.tolist() == [7, 8]
    torch.testing.assert_close(X[:, 0], torch.tensor([2.0, 0.0], dtype=torch.double))


def test_tabular_chgnet_fit_override_reapplies_structure_first_layout() -> None:
    optimizer = _optimizer()
    override = TabularDataConfig(
        input_cols=["pressure", "temperature", "phase"],
        target_cols="property",
        bounds={
            "temperature": [850.0, 1200.0],
            "pressure": [0.5, 2.0],
        },
    )

    optimizer.fit(_frame(), data_config=override)

    assert optimizer.dataset.feature_names == ["phase", "pressure", "temperature"]
    assert isinstance(optimizer.bo.bundle.model, CHGNetGPModel)


def test_tabular_chgnet_candidates_enumerate_structures_and_decode_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = _optimizer().fit(_frame())
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
        candidate = torch.tensor([[1.0, 1030.0, 1.35]], dtype=torch.double)
        return candidate, torch.tensor(0.7, dtype=torch.double)

    monkeypatch.setattr(optimizer.bo, "candidate", fake_candidate)
    candidates, acq_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        structure_ids=["beta", "gamma"],
    )

    opt_config = captured["opt_config"]
    assert opt_config.fixed_features_list == [{0: 1.0}, {0: 2.0}]
    assert candidates.loc[0, "phase"] == "beta"
    assert candidates.loc[0, "temperature"] == pytest.approx(1030.0)
    assert candidates.loc[0, "pressure"] == pytest.approx(1.35)
    assert float(acq_value) == pytest.approx(0.7)


def test_tabular_chgnet_dkl_maps_encoder_training_policy() -> None:
    optimizer = _optimizer("chgnet_dkl", encoder_training="partial").fit(_frame())
    model = optimizer.bo.bundle.model

    assert isinstance(model, CHGNetDKLModel)
    assert model.trainable_encoder_layers == 1
    assert any(parameter.requires_grad for parameter in model.material_encoder.parameters())
    assert not model.structure_feature_cache_enabled


def test_tabular_chgnet_mixed_resolves_process_categories() -> None:
    optimizer = _optimizer(mixed=True).fit(_frame(mixed=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, CHGNetMixedGPModel)
    assert bundle.input_type == "mixed"
    assert optimizer.dataset.feature_names == [
        "phase",
        "temperature",
        "furnace",
        "pressure",
        "atmosphere",
    ]
    assert optimizer.dataset.cat_dims == [0, 2, 4]
    assert optimizer.model_config.cat_dims == [2, 4]
    assert bundle.cat_dims == [2, 4]
    assert bundle.model.cat_dims == [2, 4]
    assert bundle.model.continuous_process_dims == (1, 3)
    assert bundle.model.categorical_process_dim == 2
    assert optimizer.dataset.category_maps["furnace"] == {"A": 0, "B": 1}
    assert optimizer.dataset.category_maps["atmosphere"] == {
        "air": 0,
        "N2": 1,
        "Ar": 2,
    }


def test_tabular_chgnet_mixed_dkl_maps_encoder_training_policy() -> None:
    optimizer = _optimizer(
        "chgnet_dkl",
        mixed=True,
        encoder_training="full",
    ).fit(_frame(mixed=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, CHGNetMixedDKLModel)
    assert bundle.model.trainable_encoder_layers == "all"
    assert bundle.cat_dims == [2, 4]
    assert not bundle.model.structure_feature_cache_enabled


def test_tabular_chgnet_mixed_candidates_preserve_observed_category_tuples(
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
        candidate = torch.tensor(
            [[1.0, 1030.0, 1.0, 1.35, 2.0]],
            dtype=torch.double,
        )
        return candidate, torch.tensor(0.7, dtype=torch.double)

    monkeypatch.setattr(optimizer.bo, "candidate", fake_candidate)
    candidates, _ = optimizer.candidate(
        acq_name="logei",
        q=1,
        structure_ids=["beta", "gamma"],
    )

    fixed = captured["opt_config"].fixed_features_list
    expected_category_assignments = {
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 2.0),
    }
    assert len(fixed) == 6
    assert {entry[0] for entry in fixed} == {1.0, 2.0}
    assert {(entry[2], entry[4]) for entry in fixed} == expected_category_assignments
    assert all(set(entry) == {0, 2, 4} for entry in fixed)
    assert candidates.loc[0, "phase"] == "beta"
    assert candidates.loc[0, "furnace"] == "B"
    assert candidates.loc[0, "atmosphere"] == "Ar"


def test_tabular_chgnet_rejects_alignn_graph_builder() -> None:
    class FakeGraphBuilder:
        def build_many(self, structures: tuple[object, ...]) -> tuple[object, ...]:
            return structures

    with pytest.raises(ValueError, match="ALIGNN-specific"):
        TabularBayesianOptimizer(
            model_type="chgnet_gp",
            input_cols=["phase", "temperature", "pressure"],
            target_cols="property",
            structure_col="phase",
            structure_catalog=_catalog(),
            structure_graph_builder=FakeGraphBuilder(),
            bounds={
                "temperature": [850.0, 1200.0],
                "pressure": [0.5, 2.0],
            },
            model_kwargs={"encoder": FakeCHGNet(), "latent_dim": 3},
            fit_config={"skip_fit": True},
        )


def test_tabular_chgnet_mixed_rejects_normal_input_type_override() -> None:
    with pytest.raises(ValueError, match="input_type='mixed'"):
        _optimizer(mixed=True, input_type="normal").fit(_frame(mixed=True))
