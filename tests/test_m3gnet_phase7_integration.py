"""Phase-7 integration closure for M3GNet structure-aware optimization."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from bochan.api import DataContext
from bochan.composition import M3GNetEncoder
from bochan.models.regression.gaussian.deep import M3GNetGPModel
from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular.structure.scaling import optimize_structure_alternating

pytest.importorskip("pymatgen")


class CountingGraph:
    """Minimal MatGL-like graph with device-transfer support."""

    def __init__(self, structure: Any) -> None:
        self.frac_coords = torch.as_tensor(structure.frac_coords, dtype=torch.float32)
        self.pbc_offset = torch.zeros((1, 3), dtype=torch.float32)
        self.pbc_offshift = torch.zeros((1, 3), dtype=torch.float32)
        self.pos = self.frac_coords.clone()

    def to(self, device: Any) -> CountingGraph:
        self.frac_coords = self.frac_coords.to(device)
        self.pbc_offset = self.pbc_offset.to(device)
        self.pbc_offshift = self.pbc_offshift.to(device)
        self.pos = self.pos.to(device)
        return self


class CountingGraphConverter:
    """Graph converter that records how often structures are encoded."""

    def __init__(self) -> None:
        self.calls = 0

    def get_graph(self, structure: Any) -> tuple[CountingGraph, Tensor, None]:
        self.calls += 1
        lattice = torch.as_tensor(structure.lattice.matrix, dtype=torch.float32)
        return CountingGraph(structure), lattice, None


class FakeM3GNet(nn.Module):
    """Small differentiable intensive M3GNet stand-in."""

    def __init__(self, output_dim: int = 4, n_blocks: int = 2) -> None:
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

    def forward(self, g: CountingGraph, state_attr: Tensor | None = None) -> Tensor:
        assert state_attr is None
        features = torch.tanh(self.embedding(g.frac_coords.mean(dim=0)))
        for layer in self.graph_layers:
            features = features + torch.tanh(layer(features))
        self.feature_dict = {"readout": features.unsqueeze(0)}
        return self.final_layer(features).squeeze(-1)


def _material_encoder(*, converter: CountingGraphConverter | None = None) -> M3GNetEncoder:
    return M3GNetEncoder(
        encoder=FakeM3GNet(),
        graph_converter=converter or CountingGraphConverter(),
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


def _catalog(n_structures: int = 3) -> dict[str, dict[str, object]]:
    return {
        f"s{index}": _structure(5.20 + 0.03 * index)
        for index in range(n_structures)
    }


def _single_output_frame(n_structures: int = 3, *, mixed: bool = False) -> pd.DataFrame:
    rows = []
    for index, phase in enumerate(_catalog(n_structures)):
        row: dict[str, object] = {
            "phase": phase,
            "temperature": 900.0 + 15.0 * index,
            "pressure": 0.8 + 0.05 * index,
            "property": 0.2 + 0.08 * index,
        }
        if mixed:
            row["furnace"] = "A" if index % 2 == 0 else "B"
            row["atmosphere"] = "air" if index % 2 == 0 else "N2"
        rows.append(row)
    return pd.DataFrame(rows)


def _single_output_optimizer(
    n_structures: int = 3,
    *,
    mixed: bool = False,
    converter: CountingGraphConverter | None = None,
) -> TabularBayesianOptimizer:
    input_cols = ["phase", "temperature", "pressure"]
    categorical_cols: list[str] = []
    if mixed:
        input_cols = ["temperature", "furnace", "phase", "pressure", "atmosphere"]
        categorical_cols = ["furnace", "atmosphere"]
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type="m3gnet_gp",
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        target_cols="property",
        structure_col="phase",
        structure_catalog=_catalog(n_structures),
        bounds={"temperature": [850.0, 1150.0], "pressure": [0.5, 2.0]},
        model_kwargs={
            "encoder": _material_encoder(converter=converter),
            "latent_dim": 3,
        },
        fit_config={"skip_fit": True},
    ).fit(_single_output_frame(n_structures, mixed=mixed))


def test_large_m3gnet_structure_bank_routes_q1_to_generic_alternating_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = _single_output_optimizer(12, mixed=True)
    captured: dict[str, object] = {}

    def fake_candidate(
        acq_config: Any,
        opt_config: Any,
        *,
        data_context: Any = None,
        bounds: Any = None,
        return_result: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del acq_config, data_context, bounds, return_result
        captured["opt_config"] = opt_config
        return (
            torch.tensor([[11.0, 1000.0, 1.0, 1.2, 1.0]], dtype=torch.double),
            torch.tensor(0.5, dtype=torch.double),
        )

    monkeypatch.setattr(optimizer.bo, "candidate", fake_candidate)
    candidates, acq_value = optimizer.candidate(acq_name="logei", q=1)
    config = captured["opt_config"]

    assert config.optimizer is optimize_structure_alternating
    assert config.fixed_features_list is None
    assert config.optimizer_kwargs["structure_dim"] == 0
    assert config.optimizer_kwargs["structure_values"] == [float(i) for i in range(12)]
    assert config.optimizer_kwargs["process_fixed_features_list"] == [
        {2: 0.0, 4: 0.0},
        {2: 1.0, 4: 1.0},
    ]
    assert candidates.loc[0, "phase"] == "s11"
    assert candidates.loc[0, "furnace"] == "B"
    assert candidates.loc[0, "atmosphere"] == "N2"
    assert float(acq_value) == pytest.approx(0.5)


def test_large_m3gnet_structure_bank_keeps_exact_enumeration_for_batch_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = _single_output_optimizer(12, mixed=True)
    captured: dict[str, object] = {}

    def fake_candidate(
        acq_config: Any,
        opt_config: Any,
        *,
        data_context: Any = None,
        bounds: Any = None,
        return_result: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del acq_config, data_context, bounds, return_result
        captured["opt_config"] = opt_config
        return (
            torch.tensor(
                [
                    [0.0, 950.0, 0.0, 1.0, 0.0],
                    [1.0, 1000.0, 1.0, 1.2, 1.0],
                ],
                dtype=torch.double,
            ),
            torch.tensor(0.5, dtype=torch.double),
        )

    monkeypatch.setattr(optimizer.bo, "candidate", fake_candidate)
    optimizer.candidate(acq_name="logei", q=2)
    config = captured["opt_config"]

    assert config.optimizer is not optimize_structure_alternating
    assert config.fixed_features_list is not None
    assert len(config.fixed_features_list) == 24


def test_m3gnet_single_objective_ucb_optimizes_structure_and_process() -> None:
    torch.manual_seed(0)
    optimizer = _single_output_optimizer(3)

    candidates, acq_value = optimizer.candidate(
        acq_name="UCB",
        acqf_kwargs={"beta": 0.2},
        q=1,
        structure_ids=["s0", "s1", "s2"],
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    assert candidates.loc[0, "phase"] in {"s0", "s1", "s2"}
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1150.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_m3gnet_independent_multioutput_optimizes_nehvi() -> None:
    torch.manual_seed(0)
    frame = _single_output_frame(6)
    frame["strength"] = [100.0, 108.0, 116.0, 125.0, 134.0, 145.0]
    frame["conductivity"] = [2.1, 2.3, 2.2, 2.6, 2.5, 2.9]
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="m3gnet_gp",
        input_cols=["phase", "temperature", "pressure"],
        target_cols=["strength", "conductivity"],
        structure_col="phase",
        structure_catalog=_catalog(6),
        bounds={"temperature": [850.0, 1150.0], "pressure": [0.5, 2.0]},
        model_kwargs={"encoder": _material_encoder(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    ).fit(frame)

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
        structure_ids=list(_catalog(6)),
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    assert candidates.loc[0, "phase"] in set(_catalog(6))
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1150.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_frozen_m3gnet_cache_is_rebuilt_after_state_load() -> None:
    converter = CountingGraphConverter()
    optimizer = _single_output_optimizer(12, converter=converter)
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model
    assert isinstance(model, M3GNetGPModel)
    extractor = model.m3gnet_feature_extractor
    extractor.clear_material_feature_cache()
    converter.calls = 0
    test_X = optimizer.dataset.X[:3]

    first = extractor(test_X)
    calls_after_first = converter.calls
    second = extractor(test_X)

    assert calls_after_first == 12
    assert converter.calls == calls_after_first
    torch.testing.assert_close(first, second)

    state = model.state_dict()
    model.load_state_dict(state)
    assert extractor.material_feature_cache is None
    third = extractor(test_X)

    assert converter.calls == calls_after_first + 12
    torch.testing.assert_close(first, third)
