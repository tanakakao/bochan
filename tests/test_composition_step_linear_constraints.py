from __future__ import annotations

from typing import Any

import pytest
import torch

from bochan.api import CandidateRepairConfig, OptimizeConfig
from bochan.api.support.best_subset import InfeasibleBestSubsetSupportError
from bochan.composition import CompositionTransformer
from bochan.tabular.composition.grid import (
    CompositionVariableTotalGridFinalPostprocess,
)
from bochan.tabular.composition.variable_total_support import (
    prepare_variable_total_best_subset_config,
)


def _transformer(representation: str = "ilr") -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=["Al", "Ti", "V", "Nb"],
        representation=representation,
        reference_element="Nb" if representation == "alr" else None,
        pseudocount=1e-8,
        prefix="alloy",
    )
    transformer.fit(["Al4Ti3V2Nb", "Al3Ti2V3Nb2"])
    return transformer


def _layout(transformer: CompositionTransformer) -> tuple[str, ...]:
    return (
        "temperature",
        *transformer.representation_feature_names_,
        "alloy__total",
        "pressure",
    )


def _bounds(transformer: CompositionTransformer) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    coordinate_lower = (
        [0.0] * width
        if transformer.representation == "fractions"
        else [-8.0] * width
    )
    coordinate_upper = (
        [1.0] * width
        if transformer.representation == "fractions"
        else [8.0] * width
    )
    return torch.tensor(
        [
            [800.0, *coordinate_lower, 40.0, 1.0],
            [1200.0, *coordinate_upper, 90.0, 5.0],
        ],
        dtype=torch.double,
    )


def _site(representation: str = "ilr") -> dict[str, Any]:
    return {
        "column": "formula",
        "elements": ("Al", "Ti", "V", "Nb"),
        "representation": representation,
        "normalization": "atomic_fraction",
        "reference_element": "Nb" if representation == "alr" else None,
        "pseudocount": 1e-8,
        "prefix": "alloy",
        "total": 60.0,
        "variable_total": True,
        "total_bounds": (40.0, 90.0),
        "total_feature": "alloy__total",
        "bounds": {
            "Al": (5.0, 70.0),
            "Ti": (5.0, 70.0),
            "V": (0.0, 70.0),
            "Nb": (0.0, 70.0),
        },
        "steps": {"Al": 5.0, "Ti": 5.0, "V": 5.0, "Nb": 5.0},
        "min_components": 3,
        "max_components": 3,
        "required_components": ("Al", "Ti"),
        "forbidden_components": (),
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "best_subset_max_combinations": 20,
    }


@pytest.mark.parametrize("representation", ["fractions", "clr", "alr", "ilr"])
def test_variable_total_step_grid_enforces_raw_amount_constraints(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    opt_config = OptimizeConfig(
        equality_constraints=[
            (["alloy__amount__Al", "alloy__amount__Ti"], [1.0, -1.0], 0.0),
            (["alloy__total"], [1.0], 60.0),
        ],
        inequality_constraints=[
            (["alloy__amount__Al", "alloy__amount__Ti"], [-1.0, -1.0], -45.0),
        ],
    )
    bridge, resolved, _raw_bounds = prepare_variable_total_best_subset_config(
        opt_config,
        site_name="alloy",
        site_config=_site(representation),
        transformer=transformer,
        model_feature_names=_layout(transformer),
        model_bounds=_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    projector = resolved.final_candidate_postprocess
    assert isinstance(projector, CompositionVariableTotalGridFinalPostprocess)
    raw = torch.tensor(
        [[913.0, 22.0, 18.0, 20.0, 0.0, 2.3]],
        dtype=torch.double,
    )
    projected = projector(raw)
    amounts = projected[..., list(bridge.amount_indices)]

    assert float(amounts.sum()) == pytest.approx(60.0, abs=1e-8)
    assert amounts[0, 0].item() == pytest.approx(amounts[0, 1].item(), abs=1e-8)
    assert amounts[0, 0].item() + amounts[0, 1].item() <= 45.0 + 1e-8
    assert int((amounts > 1e-10).sum()) == 3
    for value in amounts[0].tolist():
        if value > 1e-10:
            assert value / 5.0 == pytest.approx(round(value / 5.0), abs=1e-8)


def test_variable_total_step_grid_rejects_mixed_amount_process_constraint() -> None:
    transformer = _transformer("ilr")
    with pytest.raises(ValueError, match="mixes composition and non-composition"):
        prepare_variable_total_best_subset_config(
            OptimizeConfig(
                inequality_constraints=[
                    (
                        ["alloy__amount__Al", "temperature"],
                        [1.0, 0.01],
                        10.0,
                    )
                ]
            ),
            site_name="alloy",
            site_config=_site("ilr"),
            transformer=transformer,
            model_feature_names=_layout(transformer),
            model_bounds=_bounds(transformer),
            dtype=torch.double,
            device=None,
        )


def test_repair_amount_constraint_is_copied_to_variable_total_grid_milp() -> None:
    transformer = _transformer("ilr")
    bridge, resolved, _raw_bounds = prepare_variable_total_best_subset_config(
        OptimizeConfig(
            repair_config=CandidateRepairConfig(
                equality_constraints=[
                    (
                        ["alloy__amount__Al", "alloy__amount__Ti"],
                        [1.0, -1.0],
                        0.0,
                    )
                ]
            )
        ),
        site_name="alloy",
        site_config=_site("ilr"),
        transformer=transformer,
        model_feature_names=_layout(transformer),
        model_bounds=_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    projector = resolved.final_candidate_postprocess
    assert isinstance(projector, CompositionVariableTotalGridFinalPostprocess)
    projected = projector(
        torch.tensor(
            [[913.0, 24.0, 16.0, 20.0, 0.0, 2.3]],
            dtype=torch.double,
        )
    )
    amounts = projected[..., list(bridge.amount_indices)]
    assert amounts[0, 0].item() == pytest.approx(amounts[0, 1].item(), abs=1e-8)


def test_variable_total_prevalidation_allows_partially_infeasible_supports() -> None:
    transformer = _transformer("ilr")
    site = _site("ilr")
    site["bounds"] = {
        "Al": (5.0, 70.0),
        "Ti": (0.0, 70.0),
        "V": (0.0, 70.0),
        "Nb": (0.0, 70.0),
    }
    site["required_components"] = ("Al",)

    _bridge, resolved, _raw_bounds = prepare_variable_total_best_subset_config(
        OptimizeConfig(
            equality_constraints=[
                (["alloy__amount__Ti", "alloy__amount__V"], [1.0, -1.0], 0.0)
            ]
        ),
        site_name="alloy",
        site_config=site,
        transformer=transformer,
        model_feature_names=_layout(transformer),
        model_bounds=_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    assert isinstance(
        resolved.final_candidate_postprocess,
        CompositionVariableTotalGridFinalPostprocess,
    )


def test_variable_total_prevalidation_rejects_when_all_supports_are_infeasible() -> None:
    transformer = _transformer("ilr")

    with pytest.raises(
        InfeasibleBestSubsetSupportError,
        match="no feasible point|cannot satisfy",
    ):
        prepare_variable_total_best_subset_config(
            OptimizeConfig(
                equality_constraints=[(["alloy__amount__Al"], [1.0], 0.0)]
            ),
            site_name="alloy",
            site_config=_site("ilr"),
            transformer=transformer,
            model_feature_names=_layout(transformer),
            model_bounds=_bounds(transformer),
            dtype=torch.double,
            device=None,
        )
