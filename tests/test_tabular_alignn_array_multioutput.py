from __future__ import annotations

import pytest
from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor, nn

from bochan.models.regression.gaussian.deep import ALIGNNGPModel
from bochan.tabular import TabularBayesianOptimizer


class FakeALIGNN(nn.Module):
    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_projection = nn.Linear(3, output_dim)
        self.alignn_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.gcn_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.double()

    def encode(self, graph: Tensor) -> Tensor:
        import torch

        x = torch.tanh(self.input_projection(graph))
        for layer in self.alignn_layers:
            x = torch.tanh(layer(x))
        for layer in self.gcn_layers:
            x = torch.tanh(layer(x))
        return x


class FakeGraphBuilder:
    def build_many(self, structures: tuple[object, ...]) -> tuple[Tensor, ...]:
        import torch

        return tuple(
            torch.tensor(
                [1.0 + index, 0.2 + 0.1 * index, 0.1 + 0.2 * index],
                dtype=torch.double,
            )
            for index, _ in enumerate(structures)
        )


def _array_optimizer(*, target_cols=None) -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type="alignn_gp",
        input_cols=[0, 1, 2],
        target_cols=target_cols,
        structure_col=0,
        structure_catalog={0: object(), 1: object(), 2: object()},
        structure_graph_builder=FakeGraphBuilder(),
        bounds={1: [850.0, 1200.0], 2: [0.5, 2.0]},
        model_kwargs={"encoder": FakeALIGNN(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    )


def _array_data():
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
    return X, y


def test_alignn_array_fit_derives_multi_output_from_y_width() -> None:
    X, y = _array_data()
    optimizer = _array_optimizer().fit(X, y)
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert bundle.task_type == "multi_objective"
    assert len(bundle.model.models) == 2
    assert all(isinstance(model, ALIGNNGPModel) for model in bundle.model.models)
    assert optimizer.dataset.Y.shape[-1] == 2
    assert optimizer.dataset.target_names == ["y0", "y1"]
    assert bundle.model_config.multi_output_config.output_names == ["y0", "y1"]


def test_alignn_array_fit_rejects_target_name_width_mismatch() -> None:
    X, y = _array_data()

    with pytest.raises(ValueError, match="target metadata must match"):
        _array_optimizer(target_cols="strength").fit(X, y)
