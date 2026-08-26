from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from bochan.models.regression.gaussian.deep import ALIGNNDKLModel, ALIGNNGPModel
from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular.structure.candidates import _use_alternating_structure_search
from bochan.tabular.structure.scaling import optimize_alignn_structure_alternating


class CountingALIGNN(nn.Module):
    """Small ALIGNN-like encoder that records graph-encoding calls."""

    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.calls = 0
        self.input_projection = nn.Linear(3, output_dim)
        self.alignn_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.gcn_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.double()

    def encode(self, graph: Tensor) -> Tensor:
        self.calls += 1
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
                [1.0 + index, 0.2 + 0.01 * index, 0.1 + 0.02 * index],
                dtype=torch.double,
            )
            for index, _ in enumerate(structures)
        )


def _graphs(n: int = 4) -> list[Tensor]:
    return [
        torch.tensor(
            [1.0 + index, 0.2 + 0.1 * index, 0.1 + 0.2 * index],
            dtype=torch.double,
        )
        for index in range(n)
    ]


def _train_data(n_structures: int = 4) -> tuple[Tensor, Tensor]:
    rows = []
    targets = []
    for index in range(n_structures):
        rows.append([float(index), 900.0 + 25.0 * index, 0.8 + 0.1 * index])
        targets.append([0.2 * index + 0.01])
    return torch.tensor(rows, dtype=torch.double), torch.tensor(targets, dtype=torch.double)


def test_frozen_alignn_gp_reuses_structure_feature_bank() -> None:
    encoder = CountingALIGNN()
    train_X, train_Y = _train_data()
    model = ALIGNNGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=_graphs(),
        encoder=encoder,
        latent_dim=3,
        outcome_transform=None,
    )
    extractor = model.alignn_feature_extractor
    extractor.clear_material_feature_cache()
    encoder.calls = 0
    test_X = torch.tensor(
        [[0.0, 0.2, 0.3], [1.0, 0.4, 0.5], [0.0, 0.6, 0.7]],
        dtype=torch.double,
    )

    first = extractor(test_X)
    first_calls = encoder.calls
    second = extractor(test_X)

    assert model.structure_feature_cache_enabled
    assert first_calls > 0
    assert encoder.calls == first_calls
    assert extractor.material_feature_cache is not None
    assert extractor.material_feature_cache.shape == torch.Size([4, 4])
    torch.testing.assert_close(first, second)


def test_alignn_dkl_disables_frozen_structure_cache() -> None:
    encoder = CountingALIGNN()
    train_X, train_Y = _train_data()
    model = ALIGNNDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=_graphs(),
        encoder=encoder,
        latent_dim=3,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    extractor = model.alignn_feature_extractor
    extractor.clear_material_feature_cache()
    encoder.calls = 0
    test_X = torch.tensor(
        [[0.0, 0.2, 0.3], [1.0, 0.4, 0.5]],
        dtype=torch.double,
    )

    extractor(test_X)
    first_calls = encoder.calls
    extractor(test_X)

    assert not model.structure_feature_cache_enabled
    assert first_calls > 0
    assert encoder.calls > first_calls
    assert extractor.material_feature_cache is None


def test_structure_alternating_optimizer_preserves_joint_process_assignments(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_optimize_acqf_mixed_alternating(**kwargs):
        calls.append(kwargs)
        bounds = kwargs["bounds"]
        fixed = dict(kwargs.get("fixed_features") or {})
        cat_dims = kwargs["cat_dims"]
        candidate = bounds.mean(dim=0, keepdim=True)
        structure_dim = next(iter(cat_dims))
        candidate[:, structure_dim] = max(cat_dims[structure_dim])
        for dim, value in fixed.items():
            candidate[:, int(dim)] = float(value)
        score = float(fixed.get(4, 0.0)) + 0.1 * float(fixed.get(2, 0.0))
        return candidate, torch.tensor(score, dtype=bounds.dtype)

    import botorch.optim

    monkeypatch.setattr(
        botorch.optim,
        "optimize_acqf_mixed_alternating",
        fake_optimize_acqf_mixed_alternating,
    )
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0], [11.0, 1.0, 1.0, 1.0, 2.0]],
        dtype=torch.double,
    )
    process_assignments = [{2: 0.0, 4: 0.0}, {2: 1.0, 4: 2.0}]

    candidate, value = optimize_alignn_structure_alternating(
        acq_function=object(),
        bounds=bounds,
        q=1,
        num_restarts=4,
        raw_samples=32,
        structure_dim=0,
        structure_values=list(range(12)),
        process_fixed_features_list=process_assignments,
    )

    assert len(calls) == 2
    assert all(call["cat_dims"] == {0: [float(i) for i in range(12)]} for call in calls)
    assert [call["fixed_features"] for call in calls] == process_assignments
    assert candidate[0, 0].item() == pytest.approx(11.0)
    assert candidate[0, 2].item() == pytest.approx(1.0)
    assert candidate[0, 4].item() == pytest.approx(2.0)
    assert float(value) == pytest.approx(2.1)


def test_structure_scaling_auto_switch_is_conservative() -> None:
    base = SimpleNamespace(q=1, return_best_only=True, optimizer="optimize_acqf")

    assert not _use_alternating_structure_search(base, structure_count=10)
    assert _use_alternating_structure_search(base, structure_count=11)
    assert not _use_alternating_structure_search(
        SimpleNamespace(q=2, return_best_only=True, optimizer="optimize_acqf"),
        structure_count=20,
    )
    assert not _use_alternating_structure_search(
        SimpleNamespace(q=1, return_best_only=False, optimizer="optimize_acqf"),
        structure_count=20,
    )
    assert not _use_alternating_structure_search(
        SimpleNamespace(q=1, return_best_only=True, optimizer="torch"),
        structure_count=20,
    )


def _large_optimizer() -> TabularBayesianOptimizer:
    catalog = {f"s{index}": object() for index in range(12)}
    frame = pd.DataFrame(
        {
            "phase": list(catalog),
            "temperature": [900.0 + 10.0 * index for index in range(12)],
            "furnace": ["A" if index % 2 == 0 else "B" for index in range(12)],
            "pressure": [0.8 + 0.05 * index for index in range(12)],
            "atmosphere": ["air" if index % 2 == 0 else "N2" for index in range(12)],
            "property": [0.1 * index for index in range(12)],
        }
    )
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="alignn_gp",
        input_cols=["temperature", "furnace", "phase", "pressure", "atmosphere"],
        categorical_cols=["furnace", "atmosphere"],
        target_cols="property",
        structure_col="phase",
        structure_catalog=catalog,
        structure_graph_builder=FakeGraphBuilder(),
        bounds={"temperature": [850.0, 1100.0], "pressure": [0.5, 2.0]},
        model_kwargs={"encoder": CountingALIGNN(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    )
    return optimizer.fit(frame)


def test_large_structure_space_routes_q1_to_alternating_backend(monkeypatch) -> None:
    optimizer = _large_optimizer()
    captured: dict[str, object] = {}

    def fake_candidate(acq_config, opt_config, *, data_context=None, bounds=None, return_result=False):
        captured["opt_config"] = opt_config
        candidate = torch.tensor(
            [[11.0, 1000.0, 1.0, 1.2, 1.0]],
            dtype=torch.double,
        )
        return candidate, torch.tensor(0.5, dtype=torch.double)

    monkeypatch.setattr(optimizer.bo, "candidate", fake_candidate)
    candidates, _ = optimizer.candidate(acq_name="logei", q=1)
    config = captured["opt_config"]

    assert config.optimizer is optimize_alignn_structure_alternating
    assert config.fixed_features_list is None
    assert config.optimizer_kwargs["structure_dim"] == 0
    assert config.optimizer_kwargs["structure_values"] == [float(i) for i in range(12)]
    assert config.optimizer_kwargs["process_fixed_features_list"] == [
        {2: 0.0, 4: 0.0},
        {2: 1.0, 4: 1.0},
    ]
    assert candidates.loc[0, "phase"] == "s11"


def test_large_structure_space_keeps_exact_enumeration_for_batch_q(monkeypatch) -> None:
    optimizer = _large_optimizer()
    captured: dict[str, object] = {}

    def fake_candidate(acq_config, opt_config, *, data_context=None, bounds=None, return_result=False):
        captured["opt_config"] = opt_config
        candidate = torch.tensor(
            [
                [0.0, 950.0, 0.0, 1.0, 0.0],
                [1.0, 1000.0, 1.0, 1.2, 1.0],
            ],
            dtype=torch.double,
        )
        return candidate, torch.tensor(0.5, dtype=torch.double)

    monkeypatch.setattr(optimizer.bo, "candidate", fake_candidate)
    optimizer.candidate(acq_name="logei", q=2)
    config = captured["opt_config"]

    assert config.optimizer != optimize_alignn_structure_alternating
    assert config.fixed_features_list is not None
    assert len(config.fixed_features_list) == 24
