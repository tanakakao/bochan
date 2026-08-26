from __future__ import annotations

import io

import torch
from torch import Tensor, nn

from bochan.models.regression.gaussian.deep import ALIGNNGPModel


class TinyALIGNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 3
        self.input_projection = nn.Linear(3, 3)
        self.alignn_layers = nn.ModuleList([nn.Linear(3, 3)])
        self.gcn_layers = nn.ModuleList([nn.Linear(3, 3)])
        self.double()

    def encode(self, graph: Tensor) -> Tensor:
        x = torch.tanh(self.input_projection(graph))
        x = torch.tanh(self.alignn_layers[0](x))
        return torch.tanh(self.gcn_layers[0](x))


def test_frozen_structure_cache_is_not_pickled() -> None:
    graphs = [
        torch.tensor([1.0, 0.2, 0.1], dtype=torch.double),
        torch.tensor([0.3, 1.0, 0.4], dtype=torch.double),
    ]
    train_X = torch.tensor(
        [[0.0, 900.0], [1.0, 950.0], [0.0, 1000.0], [1.0, 1050.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.2], [0.5], [0.6], [0.9]], dtype=torch.double)
    model = ALIGNNGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=graphs,
        encoder=TinyALIGNN(),
        latent_dim=2,
        outcome_transform=None,
    )
    extractor = model.alignn_feature_extractor
    extractor.clear_material_feature_cache()
    extractor(torch.tensor([[0.0, 0.2], [1.0, 0.8]], dtype=torch.double))

    assert extractor.material_feature_cache is not None

    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)

    assert restored.alignn_feature_extractor.material_feature_cache is None
    assert restored.structure_feature_cache_enabled
