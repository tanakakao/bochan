from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn

from bochan.models.regression.gaussian.deep import ALIGNNDKLModel, ALIGNNGPModel
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
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def build_many(self, structures: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
        self.calls.append(structures)
        graphs = []
        for index, _ in enumerate(structures):
            graphs.append(
                torch.tensor(
                    [1.0 + index, 0.2 + 0.1 * index, 0.1 + 0.2 * index],
                    dtype=torch.double,
                )
            )
        return tuple(graphs)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase": ["alpha", "beta", "gamma", "alpha", "beta", "gamma"],
            "temperature": [900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0],
            "pressure": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
            "property": [0.4, 0.8, 1.1, 0.9, 1.4, 1.8],
        }
    )


def _catalog() -> dict[str, object]:
    return {"alpha": object(), "beta": object(), "gamma": object()}


def _optimizer(
    model_type: str = "alignn_gp",
    *,
    encoder_training: str | None = None,
) -> tuple[TabularBayesianOptimizer, FakeGraphBuilder]:
    builder = FakeGraphBuilder()
    model_kwargs: dict[str, object] = {
        "encoder": FakeALIGNN(),
        "latent_dim": 3,
    }
    if encoder_training is not None:
        model_kwargs["encoder_training"] = encoder_training
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type=model_type,
        input_cols=["temperature", "phase", "pressure"],
        target_cols="property",
        structure_col="phase",
        structure_catalog=_catalog(),
        structure_graph_builder=builder,
        bounds={
            "temperature": [850.0, 1200.0],
            "pressure": [0.5, 2.0],
        },
        model_kwargs=model_kwargs,
        fit_config={"skip_fit": True},
    )
    return optimizer, builder


def test_tabular_alignn_gp_builds_canonical_structure_index_contract() -> None:
    optimizer, builder = _optimizer()
    optimizer.fit(_frame())
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ALIGNNGPModel)
    assert optimizer.dataset.feature_names == ["phase", "temperature", "pressure"]
    assert optimizer.dataset.cat_dims == [0]
    assert optimizer.model_config.cat_dims == []
    assert optimizer.model_config.pass_cat_dims is False
    assert optimizer.structure.structure_ids == ("alpha", "beta", "gamma")
    assert optimizer.dataset.category_maps["phase"] == {
        "alpha": 0,
        "beta": 1,
        "gamma": 2,
    }
    torch.testing.assert_close(
        optimizer.dataset.X[:, 0],
        torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.double),
    )
    assert len(bundle.model.structure_graphs) == 3
    assert len(builder.calls) == 1
    assert len(builder.calls[0]) == 3


def test_tabular_alignn_prediction_input_reuses_structure_catalog_mapping() -> None:
    optimizer, _ = _optimizer()
    optimizer.fit(_frame())

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


def test_tabular_alignn_candidates_enumerate_structures_and_decode_ids(
    monkeypatch,
) -> None:
    optimizer, _ = _optimizer()
    optimizer.fit(_frame())
    captured: dict[str, object] = {}

    def fake_candidate(
        acq_config,
        opt_config,
        *,
        data_context=None,
        bounds=None,
        return_result=False,
    ):
        captured["acq_config"] = acq_config
        captured["opt_config"] = opt_config
        captured["bounds"] = bounds
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


def test_tabular_alignn_dkl_maps_encoder_training_policy() -> None:
    optimizer, _ = _optimizer("alignn_dkl", encoder_training="partial")
    optimizer.fit(_frame())
    model = optimizer.bo.bundle.model

    assert isinstance(model, ALIGNNDKLModel)
    assert model.trainable_encoder_layers == 1
    assert any(parameter.requires_grad for parameter in model.material_encoder.parameters())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"structure_col": "phase"}, "structure_catalog"),
        (
            {
                "structure_col": "phase",
                "structure_catalog": _catalog(),
                "categorical_cols": ["furnace"],
            },
            "continuous process variables only",
        ),
    ],
)
def test_tabular_alignn_rejects_invalid_structure_configuration(
    kwargs: dict[str, object],
    match: str,
) -> None:
    frame = _frame().assign(furnace=["A", "B"] * 3)
    input_cols = ["phase", "temperature", "pressure"]
    if "categorical_cols" in kwargs:
        input_cols.append("furnace")

    with pytest.raises(ValueError, match=match):
        TabularBayesianOptimizer(
            model_type="alignn_gp",
            input_cols=input_cols,
            target_cols="property",
            bounds={
                "temperature": [850.0, 1200.0],
                "pressure": [0.5, 2.0],
                **(
                    {"furnace": [0.0, 1.0]}
                    if "categorical_cols" in kwargs
                    else {}
                ),
            },
            model_kwargs={"encoder": FakeALIGNN(), "latent_dim": 3},
            fit_config={"skip_fit": True},
            **kwargs,
        ).fit(frame)
