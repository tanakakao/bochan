from __future__ import annotations

import numpy as np
import pytest
import torch

from bochan.composition import CompositionTransformer
from bochan.tabular.composition.raw_bridge import CompositionRawDecisionBridge


def _transformer(representation: str) -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=["Al", "Ti", "V", "Nb"],
        representation=representation,
        reference_element="Nb" if representation == "alr" else None,
        pseudocount=1e-8,
        prefix="alloy",
    )
    transformer.fit(["AlTiVNb", "Al2TiV"])
    return transformer


def _bridge(representation: str) -> tuple[CompositionTransformer, CompositionRawDecisionBridge]:
    transformer = _transformer(representation)
    names = (
        "temperature",
        *transformer.representation_feature_names_,
        "pressure",
    )
    return transformer, CompositionRawDecisionBridge.from_transformer(transformer, names)


@pytest.mark.parametrize("representation", ["fractions", "clr", "alr", "ilr"])
def test_raw_bridge_matches_domain_transform(representation: str) -> None:
    transformer, bridge = _bridge(representation)
    raw = torch.tensor(
        [[900.0, 0.4, 0.3, 0.2, 0.1, 2.0]],
        dtype=torch.double,
    )

    model = bridge.decision_to_model(raw)
    expected_coordinates = transformer.simplex_transform_.transform(
        np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=float)
    )

    assert model.shape[-1] == bridge.model_dim
    assert model[0, 0].item() == pytest.approx(900.0)
    assert model[0, -1].item() == pytest.approx(2.0)
    np.testing.assert_allclose(
        model[0, bridge.coordinate_start : bridge.coordinate_stop].detach().numpy(),
        expected_coordinates[0],
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_raw_bridge_keeps_structural_zero_in_decision_space(representation: str) -> None:
    _transformer_value, bridge = _bridge(representation)
    raw = torch.tensor(
        [[900.0, 0.7, 0.3, 0.0, 0.0, 2.0]],
        dtype=torch.double,
    )

    model = bridge.decision_to_model(raw)
    restored = bridge.model_to_decision(model)

    assert torch.isfinite(model).all()
    assert raw[0, bridge.fraction_indices[2]].item() == 0.0
    assert raw[0, bridge.fraction_indices[3]].item() == 0.0
    # Log-ratio inversion sees the pseudocount-smoothed composition.  Exact
    # support therefore deliberately remains a decision-space property.
    assert restored[0, bridge.fraction_indices[2]].item() > 0.0
    assert restored[0, bridge.fraction_indices[3]].item() > 0.0


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_raw_bridge_preserves_gradients(representation: str) -> None:
    _transformer_value, bridge = _bridge(representation)
    raw = torch.tensor(
        [[900.0, 0.4, 0.3, 0.2, 0.1, 2.0]],
        dtype=torch.double,
        requires_grad=True,
    )

    model = bridge.decision_to_model(raw)
    objective = model[..., bridge.coordinate_start : bridge.coordinate_stop].square().sum()
    objective.backward()

    assert raw.grad is not None
    fraction_grad = raw.grad[..., bridge.fraction_slice]
    assert torch.isfinite(fraction_grad).all()
    assert float(fraction_grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    ("representation", "coordinate_width", "decision_dim"),
    [
        ("fractions", 4, 6),
        ("clr", 4, 6),
        ("alr", 3, 6),
        ("ilr", 3, 6),
    ],
)
def test_raw_bridge_layout_and_process_index_map(
    representation: str,
    coordinate_width: int,
    decision_dim: int,
) -> None:
    _transformer_value, bridge = _bridge(representation)

    assert bridge.coordinate_width == coordinate_width
    assert bridge.decision_dim == decision_dim
    assert bridge.decision_feature_names == (
        "temperature",
        "alloy__fraction__Al",
        "alloy__fraction__Ti",
        "alloy__fraction__V",
        "alloy__fraction__Nb",
        "pressure",
    )
    assert bridge.fraction_indices == (1, 2, 3, 4)
    assert bridge.process_index_map[0] == 0
    assert bridge.process_index_map[bridge.model_dim - 1] == bridge.decision_dim - 1


def test_raw_bridge_maps_component_bounds_to_decision_space() -> None:
    _transformer_value, bridge = _bridge("ilr")
    model_bounds = torch.tensor(
        [
            [800.0, -8.0, -8.0, -8.0, 1.0],
            [1200.0, 8.0, 8.0, 8.0, 5.0],
        ],
        dtype=torch.double,
    )

    bounds = bridge.decision_bounds(
        model_bounds,
        component_bounds={
            "Al": (0.1, 0.7),
            "Ti": (0.0, 0.6),
            "V": (0.0, 0.5),
            "Nb": (0.0, 0.8),
        },
        total=1.0,
    )

    assert bounds.shape == (2, 6)
    assert bounds[:, 0].tolist() == [800.0, 1200.0]
    assert bounds[:, -1].tolist() == [1.0, 5.0]
    torch.testing.assert_close(
        bounds[:, 1:5],
        torch.tensor(
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.7, 0.6, 0.5, 0.8],
            ],
            dtype=torch.double,
        ),
    )


def test_raw_bridge_rejects_noncontiguous_model_coordinates() -> None:
    transformer = _transformer("ilr")
    coordinates = transformer.representation_feature_names_
    names = (
        coordinates[0],
        "temperature",
        coordinates[1],
        coordinates[2],
    )

    with pytest.raises(ValueError, match="contiguous"):
        CompositionRawDecisionBridge.from_transformer(transformer, names)
