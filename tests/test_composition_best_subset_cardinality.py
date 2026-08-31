from __future__ import annotations

from typing import Any

import pytest
import torch

from bochan.api import OptimizeConfig
from bochan.api.factory import optimize_candidates
from bochan.composition import CompositionTransformer
from bochan.tabular.composition.cardinality import (
    require_exact_cardinality_for_steps,
    resolve_composition_cardinality_range,
)
from bochan.tabular.composition.logratio_support import (
    prepare_logratio_best_subset_config,
)
from bochan.tabular.composition.support import resolve_composition_best_subset
from bochan.tabular.composition.variable_total_support import (
    prepare_variable_total_best_subset_config,
)
from bochan.tabular.data import resolve_optimize_config_columns


def _transformer(representation: str) -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=["Al", "Ti", "V", "Cr", "Nb"],
        representation=representation,
        reference_element="Nb" if representation == "alr" else None,
        pseudocount=1e-8,
        prefix="alloy",
    )
    transformer.fit(["AlTiVNb", "Al2TiV", "AlTiCrNb"])
    return transformer


def _site(
    representation: str,
    *,
    variable_total: bool = False,
    steps: dict[str, float] | None = None,
) -> dict[str, Any]:
    site: dict[str, Any] = {
        "column": "formula",
        "elements": ("Al", "Ti", "V", "Cr", "Nb"),
        "representation": representation,
        "normalization": "atomic_fraction",
        "reference_element": "Nb" if representation == "alr" else None,
        "pseudocount": 1e-8,
        "bounds": {
            "Al": (0.05, 0.8) if not variable_total else (5.0, 80.0),
            "Ti": (0.0, 0.8) if not variable_total else (0.0, 80.0),
            "V": (0.0, 0.8) if not variable_total else (0.0, 80.0),
            "Cr": (0.0, 0.0),
            "Nb": (0.0, 0.8) if not variable_total else (0.0, 80.0),
        },
        "steps": steps or {},
        "min_components": 2,
        "max_components": 4,
        "required_components": ("Al",),
        "forbidden_components": ("Cr",),
        "support_selection": "best_subset",
        "variable_total": variable_total,
        "best_subset_strategy": "auto",
        "best_subset_max_combinations": 20,
        "best_subset_beam_width": 4,
        "best_subset_beam_steps": 3,
        "best_subset_max_evaluations": 20,
    }
    if variable_total:
        site.update(
            total=None,
            total_bounds=(40.0, 100.0),
            total_feature="alloy__total",
        )
    else:
        site["total"] = 1.0
    return site


def _model_layout(transformer: CompositionTransformer, *, variable_total: bool) -> list[str]:
    names = [
        "temperature",
        *transformer.representation_feature_names_,
    ]
    if variable_total:
        names.append("alloy__total")
    names.append("pressure")
    return names


def _model_bounds(transformer: CompositionTransformer, *, variable_total: bool) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    lower = [800.0, *([-8.0] * width)]
    upper = [1200.0, *([8.0] * width)]
    if variable_total:
        lower.append(40.0)
        upper.append(100.0)
    lower.append(1.0)
    upper.append(5.0)
    return torch.tensor([lower, upper], dtype=torch.double)


def test_cardinality_helper_maps_total_range_to_optional_range() -> None:
    cardinality = resolve_composition_cardinality_range(
        {"min_components": 2, "max_components": 4},
        required_count=1,
        optional_count=3,
    )

    assert cardinality.minimum == 2
    assert cardinality.maximum == 4
    assert cardinality.optional_cardinalities == (1, 2, 3)


def test_step_grid_keeps_exact_cardinality_contract() -> None:
    cardinality = resolve_composition_cardinality_range(
        {"min_components": 2, "max_components": 4},
        required_count=1,
        optional_count=3,
    )

    with pytest.raises(ValueError, match="min_components == max_components"):
        require_exact_cardinality_for_steps(
            {"steps": {"Ti": 0.1}},
            cardinality,
        )


def test_fraction_best_subset_maps_component_range_to_generic_optional_k() -> None:
    transformer = _transformer("fractions")
    feature_names = list(transformer.representation_feature_names_)
    resolved = resolve_composition_best_subset(
        OptimizeConfig(),
        composition_sites={"alloy": _site("fractions")},
        composition_transformers={"alloy": transformer},
        feature_names=feature_names,
    )

    repair = resolved.repair_config
    assert repair is not None
    assert repair.support_selection == "best_subset"
    assert repair.k == 3
    assert repair.comp_idx == [
        "alloy__fraction__Ti",
        "alloy__fraction__V",
        "alloy__fraction__Nb",
    ]
    assert resolved.optimizer_kwargs["best_subset_min_k"] == 1
    assert resolved.optimizer_kwargs["best_subset_max_k"] == 3


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_logratio_best_subset_preserves_variable_cardinality_in_raw_fraction_space(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    bridge, resolved, _ = prepare_logratio_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site(representation),
        transformer=transformer,
        model_feature_names=_model_layout(transformer, variable_total=False),
        model_bounds=_model_bounds(transformer, variable_total=False),
        dtype=torch.double,
    )

    repair = resolved.repair_config
    assert repair is not None
    assert repair.k == 3
    assert tuple(repair.comp_idx or ()) == tuple(
        bridge.fraction_indices[index] for index in (1, 2, 4)
    )
    assert resolved.optimizer_kwargs["best_subset_min_k"] == 1
    assert resolved.optimizer_kwargs["best_subset_max_k"] == 3


@pytest.mark.parametrize("representation", ["fractions", "clr", "alr", "ilr"])
def test_variable_total_best_subset_preserves_variable_cardinality_in_raw_amount_space(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    bridge, resolved, _ = prepare_variable_total_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site(representation, variable_total=True),
        transformer=transformer,
        model_feature_names=_model_layout(transformer, variable_total=True),
        model_bounds=_model_bounds(transformer, variable_total=True),
        dtype=torch.double,
    )

    repair = resolved.repair_config
    assert repair is not None
    assert repair.k == 3
    assert tuple(repair.comp_idx or ()) == tuple(
        bridge.amount_indices[index] for index in (1, 2, 4)
    )
    assert resolved.optimizer_kwargs["best_subset_min_k"] == 1
    assert resolved.optimizer_kwargs["best_subset_max_k"] == 3


def test_fraction_variable_cardinality_compares_different_total_component_counts() -> None:
    transformer = _transformer("fractions")
    feature_names = list(transformer.representation_feature_names_)
    named = resolve_composition_best_subset(
        OptimizeConfig(),
        composition_sites={"alloy": _site("fractions")},
        composition_transformers={"alloy": transformer},
        feature_names=feature_names,
    )
    resolved = resolve_optimize_config_columns(
        named,
        feature_names,
        dtype=torch.double,
    )
    bounds = torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.double)

    # Al is required, Cr is forbidden. The generic sparse group is Ti/V/Nb and
    # may choose one, two, or three optional elements (2..4 total elements).
    best_support = (0, 1, 2, 4)

    def acqf(values: torch.Tensor) -> torch.Tensor:
        active = tuple(
            index
            for index in range(5)
            if bool((values[..., index].abs() > 1e-10).any().item())
        )
        return values.new_tensor(50.0 if active == best_support else float(len(active)))

    def optimizer(
        *,
        acq_function: Any,
        bounds: torch.Tensor,
        q: int,
        fixed_features: dict[int, float] | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = torch.ones(q, bounds.shape[-1], dtype=bounds.dtype)
        for index, value in (fixed_features or {}).items():
            candidate[:, int(index)] = float(value)
        return candidate, acq_function(candidate)

    resolved.optimizer = optimizer
    candidates, value = optimize_candidates(acqf, bounds, resolved)

    active = tuple(
        index
        for index in range(5)
        if bool((candidates[..., index].abs() > 1e-10).any().item())
    )
    assert active == best_support
    assert float(value.item()) == pytest.approx(50.0)


def test_variable_cardinality_with_steps_is_rejected_before_optimization() -> None:
    transformer = _transformer("ilr")
    site = _site("ilr", steps={"Al": 0.1, "Ti": 0.1, "V": 0.1, "Nb": 0.1})

    with pytest.raises(ValueError, match="min_components == max_components"):
        prepare_logratio_best_subset_config(
            OptimizeConfig(),
            site_name="alloy",
            site_config=site,
            transformer=transformer,
            model_feature_names=_model_layout(transformer, variable_total=False),
            model_bounds=_model_bounds(transformer, variable_total=False),
            dtype=torch.double,
        )
