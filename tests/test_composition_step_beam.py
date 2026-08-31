from __future__ import annotations

from typing import Any

import torch

from bochan.api import OptimizeConfig
from bochan.composition import CompositionTransformer
from bochan.tabular.composition.grid import (
    CompositionGridFinalPostprocess,
    CompositionVariableTotalGridFinalPostprocess,
)
from bochan.tabular.composition.logratio_support import prepare_logratio_best_subset_config
from bochan.tabular.composition.variable_total_support import (
    prepare_variable_total_best_subset_config,
)


def _transformer() -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=["Al", "Ti", "V", "Nb"],
        representation="ilr",
        pseudocount=1e-8,
        prefix="alloy",
    )
    transformer.fit(["Al4Ti3V2Nb", "Al3Ti2V3Nb2"])
    return transformer


def _fixed_site(**overrides: Any) -> dict[str, Any]:
    site: dict[str, Any] = {
        "column": "formula",
        "elements": ("Al", "Ti", "V", "Nb"),
        "representation": "ilr",
        "normalization": "atomic_fraction",
        "reference_element": None,
        "pseudocount": 1e-8,
        "prefix": "alloy",
        "total": 1.0,
        "variable_total": False,
        "bounds": {e: (0.0, 1.0) for e in ("Al", "Ti", "V", "Nb")},
        "steps": {e: 0.1 for e in ("Al", "Ti", "V", "Nb")},
        "min_components": 2,
        "max_components": 2,
        "required_components": (),
        "forbidden_components": (),
        "support_selection": "best_subset",
        "best_subset_strategy": "beam",
        "best_subset_max_combinations": 2,
        "best_subset_beam_width": 2,
        "best_subset_beam_steps": 2,
        "best_subset_max_evaluations": 5,
    }
    site.update(overrides)
    return site


def _fixed_layout(transformer: CompositionTransformer) -> list[str]:
    return ["temperature", *transformer.representation_feature_names_, "pressure"]


def _fixed_bounds(transformer: CompositionTransformer) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    return torch.tensor(
        [[800.0, *([-8.0] * width), 1.0], [1200.0, *([8.0] * width), 5.0]],
        dtype=torch.double,
    )


def _variable_site(**overrides: Any) -> dict[str, Any]:
    site = _fixed_site(**overrides)
    site.update({
        "variable_total": True,
        "total": 60.0,
        "total_bounds": (40.0, 90.0),
        "total_feature": "alloy__total",
        "bounds": {e: (0.0, 70.0) for e in ("Al", "Ti", "V", "Nb")},
        "steps": {e: 5.0 for e in ("Al", "Ti", "V", "Nb")},
    })
    return site


def _variable_layout(transformer: CompositionTransformer) -> tuple[str, ...]:
    return (
        "temperature",
        *transformer.representation_feature_names_,
        "alloy__total",
        "pressure",
    )


def _variable_bounds(transformer: CompositionTransformer) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    return torch.tensor(
        [[800.0, *([-8.0] * width), 40.0, 1.0], [1200.0, *([8.0] * width), 90.0, 5.0]],
        dtype=torch.double,
    )


def test_fixed_total_logratio_step_grid_accepts_beam() -> None:
    transformer = _transformer()
    _bridge, config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_fixed_site(),
        transformer=transformer,
        model_feature_names=_fixed_layout(transformer),
        model_bounds=_fixed_bounds(transformer),
        dtype=torch.double,
        device=None,
    )
    assert config.optimizer_kwargs["best_subset_strategy"] == "beam"
    assert isinstance(config.final_candidate_postprocess, CompositionGridFinalPostprocess)


def test_fixed_total_step_grid_auto_above_limit_is_accepted() -> None:
    transformer = _transformer()
    _bridge, config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_fixed_site(
            best_subset_strategy="auto",
            best_subset_max_combinations=2,
        ),
        transformer=transformer,
        model_feature_names=_fixed_layout(transformer),
        model_bounds=_fixed_bounds(transformer),
        dtype=torch.double,
        device=None,
    )
    assert config.optimizer_kwargs["best_subset_strategy"] == "auto"
    assert config.optimizer_kwargs["best_subset_max_combinations"] == 2


def test_variable_total_step_grid_accepts_beam() -> None:
    transformer = _transformer()
    _bridge, config, _bounds = prepare_variable_total_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_variable_site(),
        transformer=transformer,
        model_feature_names=_variable_layout(transformer),
        model_bounds=_variable_bounds(transformer),
        dtype=torch.double,
        device=None,
    )
    assert config.optimizer_kwargs["best_subset_strategy"] == "beam"
    assert isinstance(
        config.final_candidate_postprocess,
        CompositionVariableTotalGridFinalPostprocess,
    )
