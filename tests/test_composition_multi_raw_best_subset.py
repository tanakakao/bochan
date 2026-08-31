from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from bochan.api import OptimizeConfig
from bochan.api.support.multi_group_best_subset import BEST_SUBSET_GROUPS_KWARG
from bochan.composition import CompositionTransformer
from bochan.tabular.composition.multi_raw_support import (
    optimize_multi_raw_best_subset,
    prepare_multi_raw_best_subset_config,
    raw_best_subset_site_names,
    uses_multi_raw_best_subset,
)
from bochan.tabular.optimizer.candidates import CandidateService


def _transformer(
    prefix: str,
    elements: tuple[str, ...],
    representation: str,
) -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=list(elements),
        representation=representation,
        reference_element=elements[-1] if representation == "alr" else None,
        pseudocount=1e-8,
        prefix=prefix,
    )
    formulas = [
        "".join(elements),
        f"{elements[0]}2{elements[1]}{elements[2]}",
        f"{elements[0]}{elements[1]}2{elements[2]}",
    ]
    transformer.fit(formulas)
    return transformer


def _fixed_site(
    prefix: str,
    elements: tuple[str, ...],
    representation: str,
    *,
    steps: dict[str, float] | None = None,
    min_components: int = 2,
    max_components: int = 2,
) -> dict[str, Any]:
    return {
        "column": f"{prefix}_formula",
        "elements": elements,
        "representation": representation,
        "normalization": "atomic_fraction",
        "reference_element": elements[-1] if representation == "alr" else None,
        "pseudocount": 1e-8,
        "prefix": prefix,
        "total": 1.0,
        "variable_total": False,
        "bounds": {element: (0.0, 1.0) for element in elements},
        "steps": steps or {},
        "min_components": min_components,
        "max_components": max_components,
        "required_components": (elements[0],),
        "forbidden_components": (),
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "best_subset_max_combinations": 100,
        "best_subset_beam_width": 4,
        "best_subset_beam_steps": 3,
        "best_subset_max_evaluations": 50,
    }


def _variable_site(
    prefix: str,
    elements: tuple[str, ...],
    representation: str,
    *,
    total_feature: str | None = None,
    steps: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "column": f"{prefix}_formula",
        "elements": elements,
        "representation": representation,
        "normalization": "atomic_fraction",
        "reference_element": elements[-1] if representation == "alr" else None,
        "pseudocount": 1e-8,
        "prefix": prefix,
        "total": 65.0,
        "variable_total": True,
        "total_bounds": (40.0, 90.0),
        "total_feature": total_feature or f"{prefix}__total",
        "bounds": {element: (0.0, 70.0) for element in elements},
        "steps": steps or {},
        "min_components": 2,
        "max_components": 2,
        "required_components": (elements[0],),
        "forbidden_components": (),
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "best_subset_max_combinations": 100,
        "best_subset_beam_width": 4,
        "best_subset_beam_steps": 3,
        "best_subset_max_evaluations": 50,
    }


def _layout(
    transformers: tuple[CompositionTransformer, ...],
    *,
    total_features: tuple[str, ...] = (),
) -> tuple[str, ...]:
    names = ["temperature"]
    for transformer in transformers:
        names.extend(transformer.representation_feature_names_)
    names.extend(total_features)
    names.extend(["pressure", "phase"])
    return tuple(names)


def _bounds(
    transformers: tuple[CompositionTransformer, ...],
    *,
    total_features: tuple[str, ...] = (),
) -> torch.Tensor:
    lower: list[float] = [800.0]
    upper: list[float] = [1200.0]
    for transformer in transformers:
        width = len(transformer.representation_feature_names_)
        if transformer.representation == "fractions":
            lower.extend([0.0] * width)
            upper.extend([1.0] * width)
        else:
            lower.extend([-8.0] * width)
            upper.extend([8.0] * width)
    lower.extend([40.0] * len(total_features))
    upper.extend([90.0] * len(total_features))
    lower.extend([1.0, 0.0])
    upper.extend([5.0, 2.0])
    return torch.tensor([lower, upper], dtype=torch.double)


def _positive_raw_candidate(
    names: tuple[str, ...],
    *,
    fractions: dict[str, tuple[float, ...]],
    amounts: dict[str, tuple[float, ...]] | None = None,
) -> torch.Tensor:
    amounts = amounts or {}
    values = {name: 0.0 for name in names}
    values["temperature"] = 900.0
    values["pressure"] = 2.0
    values["phase"] = 1.0
    for prefix, row in fractions.items():
        for name, value in zip(
            [name for name in names if name.startswith(f"{prefix}__fraction__")],
            row,
            strict=True,
        ):
            values[name] = value
    for prefix, row in amounts.items():
        for name, value in zip(
            [name for name in names if name.startswith(f"{prefix}__amount__")],
            row,
            strict=True,
        ):
            values[name] = value
    return torch.tensor([[values[name] for name in names]], dtype=torch.double)


@pytest.mark.parametrize(
    ("left_repr", "right_repr"),
    [("clr", "ilr"), ("alr", "clr"), ("ilr", "alr")],
)
def test_two_logratio_sites_share_one_composite_raw_problem(
    left_repr: str,
    right_repr: str,
) -> None:
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    a = _transformer("a", a_elements, left_repr)
    b = _transformer("b", b_elements, right_repr)
    sites = {
        "a": _fixed_site("a", a_elements, left_repr),
        "b": _fixed_site(
            "b",
            b_elements,
            right_repr,
            min_components=2,
            max_components=3,
        ),
    }
    names = _layout((a, b))

    bridge, config, bounds = prepare_multi_raw_best_subset_config(
        OptimizeConfig(),
        composition_sites=sites,
        composition_transformers={"a": a, "b": b},
        model_feature_names=names,
        model_bounds=_bounds((a, b)),
        dtype=torch.double,
    )

    assert raw_best_subset_site_names(sites) == ("a", "b")
    assert uses_multi_raw_best_subset(sites)
    assert bounds.shape == (2, bridge.decision_dim)
    assert bridge.decision_feature_names == (
        "temperature",
        "a__fraction__Al",
        "a__fraction__Ti",
        "a__fraction__V",
        "b__fraction__Fe",
        "b__fraction__Ni",
        "b__fraction__Co",
        "pressure",
        "phase",
    )

    groups = config.optimizer_kwargs[BEST_SUBSET_GROUPS_KWARG]
    assert tuple(group["name"] for group in groups) == ("a", "b")
    assert tuple((group["min_k"], group["max_k"]) for group in groups) == (
        (1, 1),
        (1, 2),
    )

    raw = _positive_raw_candidate(
        bridge.decision_feature_names,
        fractions={
            "a": (0.5, 0.3, 0.2),
            "b": (0.4, 0.35, 0.25),
        },
    )
    model = bridge.decision_to_model(raw)
    restored = bridge.model_to_decision(model)
    assert torch.isfinite(model).all()
    torch.testing.assert_close(restored, raw, rtol=1e-6, atol=1e-6)


def test_logratio_and_variable_total_sites_share_fraction_and_amount_groups() -> None:
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    a = _transformer("a", a_elements, "ilr")
    b = _transformer("b", b_elements, "clr")
    sites = {
        "a": _fixed_site("a", a_elements, "ilr"),
        "b": _variable_site("b", b_elements, "clr"),
    }
    names = _layout((a, b), total_features=("b__total",))

    bridge, config, bounds = prepare_multi_raw_best_subset_config(
        OptimizeConfig(),
        composition_sites=sites,
        composition_transformers={"a": a, "b": b},
        model_feature_names=names,
        model_bounds=_bounds((a, b), total_features=("b__total",)),
        dtype=torch.double,
    )

    assert "b__total" not in bridge.decision_feature_names
    assert bridge.decision_feature_names == (
        "temperature",
        "a__fraction__Al",
        "a__fraction__Ti",
        "a__fraction__V",
        "b__amount__Fe",
        "b__amount__Ni",
        "b__amount__Co",
        "pressure",
        "phase",
    )
    groups = config.optimizer_kwargs[BEST_SUBSET_GROUPS_KWARG]
    assert tuple(group["name"] for group in groups) == ("a", "b")
    b_amount_indices = {
        bridge.decision_feature_names.index("b__amount__Fe"),
        bridge.decision_feature_names.index("b__amount__Ni"),
        bridge.decision_feature_names.index("b__amount__Co"),
    }
    total_constraints = [
        item
        for item in config.inequality_constraints or ()
        if set(torch.as_tensor(item[0]).reshape(-1).tolist()) == b_amount_indices
    ]
    assert len(total_constraints) >= 2
    assert bounds.shape[-1] == len(bridge.decision_feature_names)

    raw = _positive_raw_candidate(
        bridge.decision_feature_names,
        fractions={"a": (0.5, 0.3, 0.2)},
        amounts={"b": (25.0, 15.0, 10.0)},
    )
    model = bridge.decision_to_model(raw)
    assert model[0, names.index("b__total")].item() == pytest.approx(50.0)
    assert torch.isfinite(model).all()


def test_fraction_site_can_join_logratio_raw_group_without_extra_bridge() -> None:
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    a = _transformer("a", a_elements, "clr")
    b = _transformer("b", b_elements, "fractions")
    sites = {
        "a": _fixed_site("a", a_elements, "clr"),
        "b": _fixed_site("b", b_elements, "fractions"),
    }
    names = _layout((a, b))

    bridge, config, _bounds_raw = prepare_multi_raw_best_subset_config(
        OptimizeConfig(),
        composition_sites=sites,
        composition_transformers={"a": a, "b": b},
        model_feature_names=names,
        model_bounds=_bounds((a, b)),
        dtype=torch.double,
    )

    assert len(bridge.stages) == 1
    assert "b__fraction__Fe" in bridge.decision_feature_names
    groups = config.optimizer_kwargs[BEST_SUBSET_GROUPS_KWARG]
    assert tuple(group["name"] for group in groups) == ("a", "b")


def test_two_variable_total_sites_remove_independent_total_features() -> None:
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    a = _transformer("a", a_elements, "fractions")
    b = _transformer("b", b_elements, "ilr")
    sites = {
        "a": _variable_site("a", a_elements, "fractions"),
        "b": _variable_site("b", b_elements, "ilr"),
    }
    totals = ("a__total", "b__total")
    names = _layout((a, b), total_features=totals)

    bridge, config, _bounds_raw = prepare_multi_raw_best_subset_config(
        OptimizeConfig(),
        composition_sites=sites,
        composition_transformers={"a": a, "b": b},
        model_feature_names=names,
        model_bounds=_bounds((a, b), total_features=totals),
        dtype=torch.double,
    )

    assert "a__total" not in bridge.decision_feature_names
    assert "b__total" not in bridge.decision_feature_names
    assert all("__amount__" in name for name in bridge.decision_feature_names[1:7])
    assert tuple(
        group["name"]
        for group in config.optimizer_kwargs[BEST_SUBSET_GROUPS_KWARG]
    ) == ("a", "b")

    raw = _positive_raw_candidate(
        bridge.decision_feature_names,
        fractions={},
        amounts={
            "a": (20.0, 15.0, 10.0),
            "b": (25.0, 15.0, 10.0),
        },
    )
    model = bridge.decision_to_model(raw)
    assert model[0, names.index("a__total")].item() == pytest.approx(45.0)
    assert model[0, names.index("b__total")].item() == pytest.approx(50.0)


def test_variable_total_sites_require_distinct_total_features() -> None:
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    a = _transformer("a", a_elements, "fractions")
    b = _transformer("b", b_elements, "fractions")
    sites = {
        "a": _variable_site(
            "a",
            a_elements,
            "fractions",
            total_feature="shared_total",
        ),
        "b": _variable_site(
            "b",
            b_elements,
            "fractions",
            total_feature="shared_total",
        ),
    }
    names = _layout((a, b), total_features=("shared_total",))

    with pytest.raises(KeyError, match="shared_total"):
        prepare_multi_raw_best_subset_config(
            OptimizeConfig(),
            composition_sites=sites,
            composition_transformers={"a": a, "b": b},
            model_feature_names=names,
            model_bounds=_bounds((a, b), total_features=("shared_total",)),
            dtype=torch.double,
        )


def test_cross_site_step_constraint_is_rejected_for_logratio_and_variable_total() -> None:
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    a = _transformer("a", a_elements, "ilr")
    b = _transformer("b", b_elements, "clr")
    sites = {
        "a": _fixed_site(
            "a",
            a_elements,
            "ilr",
            steps={element: 0.5 for element in a_elements},
        ),
        "b": _variable_site(
            "b",
            b_elements,
            "clr",
            steps={element: 10.0 for element in b_elements},
        ),
    }
    names = _layout((a, b), total_features=("b__total",))

    with pytest.raises(
        ValueError,
        match="cannot couple two composition Best Subset sites",
    ):
        prepare_multi_raw_best_subset_config(
            OptimizeConfig(
                equality_constraints=[
                    (
                        ["a__fraction__Al", "b__amount__Fe"],
                        [1.0, -0.01],
                        0.0,
                    )
                ]
            ),
            composition_sites=sites,
            composition_transformers={"a": a, "b": b},
            model_feature_names=names,
            model_bounds=_bounds((a, b), total_features=("b__total",)),
            dtype=torch.double,
        )


class _RecordingAcquisition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace()
        self.last_x: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_x = x
        return -x.square().sum(dim=(-1, -2))


def test_multi_raw_optimizer_returns_model_candidates_and_exact_raw_support() -> None:
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    a = _transformer("a", a_elements, "ilr")
    b = _transformer("b", b_elements, "clr")
    sites = {
        "a": _fixed_site("a", a_elements, "ilr"),
        "b": _variable_site("b", b_elements, "clr"),
    }
    names = _layout((a, b), total_features=("b__total",))
    base = _RecordingAcquisition()
    captured: dict[str, Any] = {}

    def fake_optimize(*, acqf: Any, bounds: torch.Tensor, config: OptimizeConfig):
        captured["config"] = config
        captured["bounds"] = bounds
        raw_names = (
            "temperature",
            "a__fraction__Al",
            "a__fraction__Ti",
            "a__fraction__V",
            "b__amount__Fe",
            "b__amount__Ni",
            "b__amount__Co",
            "pressure",
            "phase",
        )
        raw = _positive_raw_candidate(
            raw_names,
            fractions={"a": (0.6, 0.4, 0.0)},
            amounts={"b": (30.0, 20.0, 0.0)},
        )
        return raw, acqf(raw.unsqueeze(-2))

    result = optimize_multi_raw_best_subset(
        base,
        OptimizeConfig(),
        composition_sites=sites,
        composition_transformers={"a": a, "b": b},
        model_feature_names=names,
        model_bounds=_bounds((a, b), total_features=("b__total",)),
        dtype=torch.double,
        optimize_fn=fake_optimize,
    )

    assert BEST_SUBSET_GROUPS_KWARG in captured["config"].optimizer_kwargs
    assert result.raw_candidates.shape[-1] == result.bridge.decision_dim
    assert result.candidates.shape[-1] == result.bridge.model_dim
    assert torch.isfinite(result.candidates).all()
    assert base.last_x is not None
    torch.testing.assert_close(base.last_x, result.candidates)


def test_candidate_service_routes_multiple_raw_sites_before_single_site_guard() -> None:
    sites = {
        "a": _fixed_site("a", ("Al", "Ti", "V"), "ilr"),
        "b": _fixed_site("b", ("Fe", "Ni", "Co"), "clr"),
    }
    service = object.__new__(CandidateService)
    service.composition = SimpleNamespace(sites=sites)
    marker = object()
    service._raw_multi_candidate = lambda *args, **kwargs: marker

    result = CandidateService._raw_candidate(
        service,
        SimpleNamespace(),
        SimpleNamespace(),
        OptimizeConfig(),
        data_context=None,
        bounds=None,
        return_result=False,
    )
    assert result is marker
