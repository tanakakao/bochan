from __future__ import annotations

from typing import Any

import pytest
import torch

from bochan.api import OptimizeConfig
from bochan.composition import CompositionTransformer
from bochan.tabular.composition.grid import CompositionGridFinalPostprocess
from bochan.tabular.composition.logratio_support import (
    prepare_logratio_best_subset_config,
)


def _transformer(representation: str = "ilr") -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=["Al", "Ti", "V", "Cr", "Nb"],
        representation=representation,
        reference_element="Nb" if representation == "alr" else None,
        pseudocount=1e-8,
        prefix="alloy",
    )
    transformer.fit(["AlTiVNb", "Al2TiV", "AlTiCrNb"])
    return transformer


def _site(representation: str = "ilr", **overrides: Any) -> dict[str, Any]:
    site: dict[str, Any] = {
        "column": "formula",
        "elements": ("Al", "Ti", "V", "Cr", "Nb"),
        "representation": representation,
        "normalization": "atomic_fraction",
        "reference_element": "Nb" if representation == "alr" else None,
        "pseudocount": 1e-8,
        "total": 1.0,
        "bounds": {
            "Al": (0.0, 1.0),
            "Ti": (0.0, 1.0),
            "V": (0.0, 1.0),
            "Cr": (0.0, 0.0),
            "Nb": (0.0, 1.0),
        },
        "steps": {
            "Al": 0.1,
            "Ti": 0.1,
            "V": 0.1,
            "Nb": 0.1,
        },
        "min_components": 3,
        "max_components": 3,
        "required_components": ("Al",),
        "forbidden_components": ("Cr",),
        "support_selection": "best_subset",
        "variable_total": False,
        "best_subset_strategy": "exact",
        "best_subset_max_combinations": 20,
        "best_subset_beam_width": 4,
        "best_subset_beam_steps": 3,
        "best_subset_max_evaluations": 20,
    }
    site.update(overrides)
    return site


def _model_layout(transformer: CompositionTransformer) -> list[str]:
    return [
        "temperature",
        *transformer.representation_feature_names_,
        "pressure",
    ]


def _model_bounds(transformer: CompositionTransformer) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    return torch.tensor(
        [
            [800.0, *([-8.0] * width), 1.0],
            [1200.0, *([8.0] * width), 5.0],
        ],
        dtype=torch.double,
    )


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_logratio_best_subset_step_grid_projects_raw_fractions(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    bridge, config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site(representation),
        transformer=transformer,
        model_feature_names=_model_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    callback = config.final_candidate_postprocess
    assert isinstance(callback, CompositionGridFinalPostprocess)
    raw = torch.tensor(
        [[913.0, 0.44, 0.31, 0.25, 0.0, 0.0, 2.3]],
        dtype=torch.double,
    )
    projected = callback(raw)
    fractions = projected[..., bridge.fraction_slice]

    assert fractions.sum().item() == pytest.approx(1.0, abs=1e-10)
    assert int((fractions > 1e-10).sum().item()) == 3
    assert fractions[0, 3].item() == pytest.approx(0.0)
    assert fractions[0, 4].item() == pytest.approx(0.0)
    for value in fractions[0, :3].tolist():
        assert value / 0.1 == pytest.approx(round(value / 0.1), abs=1e-8)
    assert torch.isfinite(bridge.decision_to_model(projected)).all()


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_logratio_step_grid_enforces_raw_fraction_linear_constraint(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    bridge, config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(
            equality_constraints=[
                (["alloy__fraction__Al"], [1.0], 0.4),
            ]
        ),
        site_name="alloy",
        site_config=_site(representation),
        transformer=transformer,
        model_feature_names=_model_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    callback = config.final_candidate_postprocess
    assert isinstance(callback, CompositionGridFinalPostprocess)
    raw = torch.tensor(
        [[913.0, 0.44, 0.31, 0.25, 0.0, 0.0, 2.3]],
        dtype=torch.double,
    )
    projected = callback(raw)
    fractions = projected[..., bridge.fraction_slice]

    assert fractions[0, 0].item() == pytest.approx(0.4, abs=1e-10)
    assert fractions.sum().item() == pytest.approx(1.0, abs=1e-10)
    assert int((fractions > 1e-10).sum().item()) == 3
    assert torch.isfinite(bridge.decision_to_model(projected)).all()


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_logratio_step_grid_rejects_process_coupled_raw_constraint(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    config = OptimizeConfig(
        equality_constraints=[
            (
                ["alloy__fraction__Al", "temperature"],
                [1.0, -0.001],
                -0.5,
            )
        ]
    )

    with pytest.raises(ValueError, match="mixes composition and non-composition"):
        prepare_logratio_best_subset_config(
            config,
            site_name="alloy",
            site_config=_site(representation),
            transformer=transformer,
            model_feature_names=_model_layout(transformer),
            model_bounds=_model_bounds(transformer),
            dtype=torch.double,
            device=None,
        )


def test_logratio_step_grid_auto_can_switch_to_beam() -> None:
    transformer = _transformer("ilr")
    _bridge, config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site(
            "ilr",
            best_subset_strategy="auto",
            best_subset_max_combinations=2,
        ),
        transformer=transformer,
        model_feature_names=_model_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )
    assert config.optimizer_kwargs["best_subset_strategy"] == "auto"
    assert config.optimizer_kwargs["best_subset_max_combinations"] == 2


def test_step_grid_strategy_honors_explicit_optimizer_kwargs_over_site_defaults() -> None:
    transformer = _transformer("ilr")
    _bridge, config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(
            optimizer_kwargs={
                "best_subset_strategy": "exact",
                "best_subset_max_combinations": 20,
            }
        ),
        site_name="alloy",
        site_config=_site(
            "ilr",
            best_subset_strategy="beam",
            best_subset_max_combinations=1,
        ),
        transformer=transformer,
        model_feature_names=_model_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )
    assert config.optimizer_kwargs["best_subset_strategy"] == "exact"
    assert config.optimizer_kwargs["best_subset_max_combinations"] == 20

    _bridge, beam_config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(optimizer_kwargs={"best_subset_strategy": "beam"}),
        site_name="alloy",
        site_config=_site("ilr", best_subset_strategy="exact"),
        transformer=transformer,
        model_feature_names=_model_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )
    assert beam_config.optimizer_kwargs["best_subset_strategy"] == "beam"
