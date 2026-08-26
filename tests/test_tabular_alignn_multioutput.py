from __future__ import annotations

import pandas as pd
import pytest
import torch
from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor, nn

from bochan.models.regression.gaussian.deep import (
    ALIGNNDKLModel,
    ALIGNNGPModel,
    ALIGNNMixedDKLModel,
    ALIGNNMixedGPModel,
)
from bochan.tabular import TabularBayesianOptimizer


class FakeALIGNN(nn.Module):
    """Small injected ALIGNN-like encoder for dependency-free tests."""

    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_projection = nn.Linear(3, output_dim)
        self.alignn_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.gcn_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.double()

    def encode(self, graph: Tensor) -> Tensor:
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
                [1.0 + index, 0.2 + 0.1 * index, 0.1 + 0.2 * index],
                dtype=torch.double,
            )
            for index, _ in enumerate(structures)
        )


def _catalog() -> dict[str, object]:
    return {"alpha": object(), "beta": object(), "gamma": object()}


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


def _optimizer(model_type: str = "alignn_gp", *, mixed: bool = False) -> TabularBayesianOptimizer:
    input_cols = ["temperature", "phase", "pressure"]
    categorical_cols: list[str] = []
    if mixed:
        input_cols = ["temperature", "furnace", "phase", "pressure"]
        categorical_cols = ["furnace"]
    model_kwargs: dict[str, object] = {"encoder": FakeALIGNN(), "latent_dim": 3}
    if model_type == "alignn_dkl":
        model_kwargs["encoder_training"] = "partial"
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type=model_type,
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        target_cols=["strength", "conductivity"],
        structure_col="phase",
        structure_catalog=_catalog(),
        structure_graph_builder=FakeGraphBuilder(),
        bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
        model_kwargs=model_kwargs,
        fit_config={"skip_fit": True},
    )


def test_alignn_gp_multi_output_auto_builds_independent_model_list() -> None:
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
    assert all(isinstance(model, ALIGNNGPModel) for model in bundle.model.models)
    assert bundle.model.models[0].material_encoder is not bundle.model.models[1].material_encoder
    assert bundle.model.models[0].structure_graphs is bundle.model.models[1].structure_graphs
    assert optimizer.dataset.Y.shape == torch.Size([6, 2])

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


def test_alignn_mixed_gp_multi_output_preserves_process_categories() -> None:
    optimizer = _optimizer(mixed=True).fit(_frame(mixed=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert bundle.input_type == "mixed"
    assert bundle.cat_dims == [2]
    assert len(bundle.model.models) == 2
    assert all(isinstance(model, ALIGNNMixedGPModel) for model in bundle.model.models)
    assert all(model.cat_dims == [2] for model in bundle.model.models)
    assert optimizer.dataset.feature_names == [
        "phase",
        "temperature",
        "furnace",
        "pressure",
    ]
    assert optimizer.dataset.cat_dims == [0, 2]
    assert optimizer.dataset.category_maps["furnace"] == {"A": 0, "B": 1}


def test_alignn_dkl_multi_output_uses_independent_trainable_encoders() -> None:
    optimizer = _optimizer("alignn_dkl").fit(_frame(mixed=False))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    first, second = bundle.model.models
    assert isinstance(first, ALIGNNDKLModel)
    assert isinstance(second, ALIGNNDKLModel)
    assert first.material_encoder is not second.material_encoder
    assert first.trainable_encoder_layers == 1
    assert second.trainable_encoder_layers == 1
    first_params = {id(parameter) for parameter in first.material_encoder.parameters()}
    second_params = {id(parameter) for parameter in second.material_encoder.parameters()}
    assert first_params.isdisjoint(second_params)
    assert any(parameter.requires_grad for parameter in first.material_encoder.parameters())
    assert any(parameter.requires_grad for parameter in second.material_encoder.parameters())


def test_alignn_mixed_dkl_multi_output_uses_mixed_submodels() -> None:
    optimizer = _optimizer("alignn_dkl", mixed=True).fit(_frame(mixed=True))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert all(isinstance(model, ALIGNNMixedDKLModel) for model in bundle.model.models)
    assert all(model.cat_dims == [2] for model in bundle.model.models)


def test_alignn_multi_output_candidate_keeps_structure_and_category_contract(monkeypatch) -> None:
    optimizer = _optimizer(mixed=True).fit(_frame(mixed=True))
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


def test_alignn_rejects_explicit_user_multi_output_config() -> None:
    from bochan.api import MultiOutputConfig

    with pytest.raises(ValueError, match="derives independent multi-output structure automatically"):
        TabularBayesianOptimizer(
            task_type="multi_objective",
            model_type="alignn_gp",
            multi_output_config=MultiOutputConfig(output_names=["strength", "conductivity"]),
            input_cols=["temperature", "phase", "pressure"],
            target_cols=["strength", "conductivity"],
            structure_col="phase",
            structure_catalog=_catalog(),
            structure_graph_builder=FakeGraphBuilder(),
            bounds={"temperature": [850.0, 1200.0], "pressure": [0.5, 2.0]},
            model_kwargs={"encoder": FakeALIGNN(), "latent_dim": 3},
            fit_config={"skip_fit": True},
        ).fit(_frame(mixed=False))
