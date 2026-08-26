from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn

from bochan.models.regression.gaussian.deep import (
    ALIGNNMixedDKLModel,
    ALIGNNMixedGPModel,
)
from bochan.tabular import TabularBayesianOptimizer


class FakeALIGNN(nn.Module):
    """Small injected structure encoder for dependency-free tabular tests."""

    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_projection = nn.Linear(3, output_dim)
        self.alignn_layers = nn.ModuleList(
            [nn.Linear(output_dim, output_dim), nn.Linear(output_dim, output_dim)]
        )
        self.gcn_layers = nn.ModuleList(
            [nn.Linear(output_dim, output_dim), nn.Linear(output_dim, output_dim)]
        )
        self.double()

    def encode(self, graph: torch.Tensor) -> torch.Tensor:
        values = torch.tanh(self.input_projection(graph))
        for layer in self.alignn_layers:
            values = torch.tanh(layer(values))
        for layer in self.gcn_layers:
            values = torch.tanh(layer(values))
        return values


class FakeGraphBuilder:
    def build_many(self, structures: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.tensor(
                [1.0 + index, 0.2 + 0.1 * index, 0.1 + 0.2 * index],
                dtype=torch.double,
            )
            for index, _ in enumerate(structures)
        )


def _catalog() -> dict[str, object]:
    return {"alpha": object(), "beta": object(), "gamma": object()}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase": ["alpha", "beta", "gamma", "alpha", "beta", "gamma"],
            "temperature": [900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0],
            "furnace": ["A", "B", "A", "B", "A", "A"],
            "pressure": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
            "atmosphere": ["air", "N2", "air", "N2", "Ar", "Ar"],
            "property": [0.4, 0.8, 1.1, 0.9, 1.4, 1.8],
        }
    )


def _optimizer(
    model_type: str = "alignn_gp",
    *,
    encoder_training: str | None = None,
    input_type: str | None = None,
) -> TabularBayesianOptimizer:
    model_kwargs: dict[str, object] = {
        "encoder": FakeALIGNN(),
        "latent_dim": 3,
    }
    if encoder_training is not None:
        model_kwargs["encoder_training"] = encoder_training
    model_config = None if input_type is None else {"input_type": input_type}
    return TabularBayesianOptimizer(
        model_config=model_config,
        task_type="regression",
        model_type=model_type,
        input_cols=[
            "temperature",
            "furnace",
            "phase",
            "pressure",
            "atmosphere",
        ],
        categorical_cols=["furnace", "atmosphere"],
        target_cols="property",
        structure_col="phase",
        structure_catalog=_catalog(),
        structure_graph_builder=FakeGraphBuilder(),
        bounds={
            "temperature": [850.0, 1200.0],
            "pressure": [0.5, 2.0],
        },
        model_kwargs=model_kwargs,
        fit_config={"skip_fit": True},
    )


def test_tabular_alignn_gp_auto_resolves_mixed_process_contract() -> None:
    optimizer = _optimizer().fit(_frame())
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ALIGNNMixedGPModel)
    assert bundle.input_type == "mixed"
    assert optimizer.model_config.input_type == "mixed"
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
    assert 0 not in bundle.cat_dims
    assert bundle.model.process_dim == 2
    assert bundle.model.continuous_process_dims == (1, 3)
    assert bundle.model.categorical_process_dim == 2
    assert optimizer.dataset.category_maps["phase"] == {
        "alpha": 0,
        "beta": 1,
        "gamma": 2,
    }
    assert optimizer.dataset.category_maps["furnace"] == {"A": 0, "B": 1}
    assert optimizer.dataset.category_maps["atmosphere"] == {
        "air": 0,
        "N2": 1,
        "Ar": 2,
    }
    assert optimizer.data_config.category_maps["furnace"] == {"A": 0, "B": 1}
    assert optimizer.data_config.category_maps["atmosphere"] == {
        "air": 0,
        "N2": 1,
        "Ar": 2,
    }

    bounds = optimizer.dataset.bounds
    assert bounds is not None
    torch.testing.assert_close(bounds[:, 0], torch.tensor([0.0, 2.0], dtype=torch.double))
    torch.testing.assert_close(bounds[:, 1], torch.tensor([850.0, 1200.0], dtype=torch.double))
    torch.testing.assert_close(bounds[:, 2], torch.tensor([0.0, 1.0], dtype=torch.double))
    torch.testing.assert_close(bounds[:, 3], torch.tensor([0.5, 2.0], dtype=torch.double))
    torch.testing.assert_close(bounds[:, 4], torch.tensor([0.0, 2.0], dtype=torch.double))


def test_tabular_alignn_mixed_prediction_reuses_fitted_category_maps() -> None:
    optimizer = _optimizer().fit(_frame())

    X, index = optimizer._prediction_input(
        pd.DataFrame(
            {
                "phase": ["gamma", "alpha"],
                "temperature": [1025.0, 1075.0],
                "furnace": ["B", "A"],
                "pressure": [1.3, 1.5],
                "atmosphere": ["Ar", "air"],
            },
            index=[7, 8],
        )
    )

    assert index.tolist() == [7, 8]
    torch.testing.assert_close(X[:, 0], torch.tensor([2.0, 0.0], dtype=torch.double))
    torch.testing.assert_close(X[:, 2], torch.tensor([1.0, 0.0], dtype=torch.double))
    torch.testing.assert_close(X[:, 4], torch.tensor([2.0, 0.0], dtype=torch.double))


def test_tabular_alignn_mixed_dkl_maps_encoder_training_policy() -> None:
    optimizer = _optimizer("alignn_dkl", encoder_training="partial").fit(_frame())
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ALIGNNMixedDKLModel)
    assert bundle.input_type == "mixed"
    assert bundle.cat_dims == [2, 4]
    assert bundle.model.trainable_encoder_layers == 1
    assert any(
        parameter.requires_grad for parameter in bundle.model.material_encoder.parameters()
    )


def test_tabular_alignn_mixed_candidates_cross_structure_and_observed_categories(
    monkeypatch,
) -> None:
    optimizer = _optimizer().fit(_frame())
    captured: dict[str, object] = {}

    def fake_candidate(
        acq_config,
        opt_config,
        *,
        data_context=None,
        bounds=None,
        return_result=False,
    ):
        captured["opt_config"] = opt_config
        candidate = torch.tensor(
            [[1.0, 1030.0, 1.0, 1.35, 2.0]],
            dtype=torch.double,
        )
        return candidate, torch.tensor(0.7, dtype=torch.double)

    monkeypatch.setattr(optimizer.bo, "candidate", fake_candidate)
    candidates, acq_value = optimizer.candidate(
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
    assert candidates.loc[0, "temperature"] == pytest.approx(1030.0)
    assert candidates.loc[0, "pressure"] == pytest.approx(1.35)
    assert float(acq_value) == pytest.approx(0.7)


def test_tabular_alignn_mixed_rejects_normal_input_type_override() -> None:
    with pytest.raises(ValueError, match="input_type='mixed'"):
        _optimizer(input_type="normal").fit(_frame())
