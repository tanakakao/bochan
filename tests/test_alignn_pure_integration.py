from __future__ import annotations

import pytest
import torch

from bochan.composition import ALIGNNEncoder
from bochan.models.regression.gaussian.deep import ALIGNNGPModel
from bochan.structure import ALIGNNGraphBuilder


def _si_structure(scale: float) -> dict[str, object]:
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


def _small_pure_encoder() -> ALIGNNEncoder:
    return ALIGNNEncoder(
        config={
            "name": "alignn_atomwise_pure",
            "alignn_layers": 1,
            "gcn_layers": 1,
            "atom_input_features": 92,
            "edge_input_features": 16,
            "triplet_input_features": 8,
            "embedding_features": 8,
            "hidden_features": 16,
            "output_features": 1,
            "calculate_gradient": False,
            "gradwise_weight": 0.0,
            "energy_mult_natoms": False,
            "use_penalty": False,
        }
    )


def test_real_pure_alignn_encoder_runs_on_torch_graphs() -> None:
    pytest.importorskip("alignn")
    graph = ALIGNNGraphBuilder(cutoff=5.0).build(_si_structure(5.43))
    encoder = _small_pure_encoder().double()

    features = encoder([graph])

    assert encoder.encoder.__class__.__module__ == "alignn.models.alignn_atomwise_pure"
    assert encoder.encoder.__class__.__name__ == "ALIGNNAtomWisePure"
    assert features.shape == torch.Size([1, 16])
    assert features.dtype == torch.double
    assert torch.isfinite(features).all()


def test_real_pure_alignn_gp_runs_from_crystal_graph_to_posterior() -> None:
    pytest.importorskip("alignn")
    builder = ALIGNNGraphBuilder(cutoff=5.0)
    graphs = builder.build_many([_si_structure(5.43), _si_structure(5.55)])
    encoder = _small_pure_encoder().double()
    train_X = torch.tensor(
        [
            [0.0, 900.0],
            [1.0, 930.0],
            [0.0, 980.0],
            [1.0, 1020.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.40], [0.72], [0.68], [1.05]], dtype=torch.double)

    model = ALIGNNGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=graphs,
        encoder=encoder,
        latent_dim=4,
        outcome_transform=None,
    )
    posterior = model.posterior(
        torch.tensor([[0.0, 950.0], [1.0, 950.0]], dtype=torch.double)
    )

    assert model.num_structures == 2
    assert model.process_dim == 1
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
