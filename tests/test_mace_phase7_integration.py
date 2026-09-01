"""Phase-7 integration closure for MACE structure-aware optimization."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("mace")

from bochan.api import DataContext
from bochan.composition import MACEEncoder
from bochan.models.regression.gaussian.deep import (
    MACEGPModel,
    MACEMixedMultiTaskGPModel,
)
from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular.structure.scaling import optimize_structure_alternating


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
    """Small differentiable MACE stand-in with the descriptor metadata bochan needs."""

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


class CountingBatchBuilder:
    """Convert mapping structures to MACE-like batches and record encoding work."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, structure: dict[str, object]) -> dict[str, Tensor]:
        self.calls += 1
        lattice = torch.tensor(structure["lattice_mat"], dtype=torch.float32)
        coords = torch.tensor(structure["coords"], dtype=torch.float32)
        positions = coords if bool(structure.get("cartesian", False)) else coords @ lattice
        return {"positions": positions}


def _material_encoder(*, batch_builder: CountingBatchBuilder | None = None) -> MACEEncoder:
    return MACEEncoder(
        FakeMACE(),
        batch_builder=batch_builder or CountingBatchBuilder(),
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
    batch_builder: CountingBatchBuilder | None = None,
) -> TabularBayesianOptimizer:
    input_cols = ["phase", "temperature", "pressure"]
    categorical_cols: list[str] = []
    if mixed:
        input_cols = ["temperature", "furnace", "phase", "pressure", "atmosphere"]
        categorical_cols = ["furnace", "atmosphere"]
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type="mace_gp",
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        target_cols="property",
        structure_col="phase",
        structure_catalog=_catalog(n_structures),
        bounds={"temperature": [850.0, 1150.0], "pressure": [0.5, 2.0]},
        model_kwargs={
            "encoder": _material_encoder(batch_builder=batch_builder),
            "latent_dim": 3,
        },
        fit_config={"skip_fit": True},
    ).fit(_single_output_frame(n_structures, mixed=mixed))


def _multi_output_frame(*, mixed: bool = False) -> pd.DataFrame:
    frame = _single_output_frame(6, mixed=mixed)
    frame["strength"] = [100.0, 108.0, 116.0, 125.0, 134.0, 145.0]
    frame["conductivity"] = [2.1, 2.3, 2.2, 2.6, 2.5, 2.9]
    return frame


def test_large_mace_structure_bank_routes_q1_to_generic_alternating_backend(
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


def test_large_mace_structure_bank_keeps_exact_enumeration_for_batch_q(
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


def test_mace_single_objective_logei_optimizes_structure_and_process() -> None:
    torch.manual_seed(0)
    optimizer = _single_output_optimizer(6)

    candidates, acq_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        structure_ids=list(_catalog(6)),
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    assert candidates.loc[0, "phase"] in set(_catalog(6))
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1150.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_mace_single_objective_ucb_optimizes_structure_and_process() -> None:
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


def test_mace_mixed_process_categories_optimize_with_structure_selection() -> None:
    torch.manual_seed(0)
    optimizer = _single_output_optimizer(6, mixed=True)

    candidates, acq_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        structure_ids=list(_catalog(6)),
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    assert candidates.loc[0, "phase"] in set(_catalog(6))
    assert candidates.loc[0, "furnace"] in {"A", "B"}
    assert candidates.loc[0, "atmosphere"] in {"air", "N2"}
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1150.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_mace_independent_multioutput_optimizes_nehvi() -> None:
    torch.manual_seed(0)
    frame = _multi_output_frame()
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="mace_gp",
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


def test_mace_correlated_multitask_mixed_optimizes_nehvi() -> None:
    torch.manual_seed(0)
    frame = _multi_output_frame(mixed=True)
    optimizer = TabularBayesianOptimizer(
        task_type="multi_objective",
        model_type="mace_multitask",
        input_cols=["temperature", "furnace", "phase", "pressure", "atmosphere"],
        categorical_cols=["furnace", "atmosphere"],
        target_cols=["strength", "conductivity"],
        structure_col="phase",
        structure_catalog=_catalog(6),
        bounds={"temperature": [850.0, 1150.0], "pressure": [0.5, 2.0]},
        model_kwargs={"encoder": _material_encoder(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    ).fit(frame)
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, MACEMixedMultiTaskGPModel)
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
    assert candidates.loc[0, "furnace"] in {"A", "B"}
    assert candidates.loc[0, "atmosphere"] in {"air", "N2"}
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1150.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_frozen_mace_cache_is_rebuilt_after_state_load() -> None:
    batch_builder = CountingBatchBuilder()
    optimizer = _single_output_optimizer(12, batch_builder=batch_builder)
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model
    assert isinstance(model, MACEGPModel)
    extractor = model.mace_feature_extractor
    extractor.clear_material_feature_cache()
    batch_builder.calls = 0
    test_X = optimizer.dataset.X[:3]

    first = extractor(test_X)
    calls_after_first = batch_builder.calls
    second = extractor(test_X)

    assert calls_after_first == 12
    assert batch_builder.calls == calls_after_first
    torch.testing.assert_close(first, second)

    state = model.state_dict()
    model.load_state_dict(state)
    assert extractor.material_feature_cache is None
    third = extractor(test_X)

    assert batch_builder.calls == calls_after_first + 12
    torch.testing.assert_close(first, third)
