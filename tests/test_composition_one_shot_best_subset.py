from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from torch import Tensor, nn

from bochan.api import OptimizeConfig
from bochan.composition import CompositionTransformer
from bochan.tabular.composition.logratio_support import optimize_logratio_best_subset
from bochan.tabular.composition.multi_raw_support import optimize_multi_raw_best_subset
from bochan.tabular.composition.variable_total_support import (
    optimize_variable_total_best_subset,
)


class _QuadraticOneShot(OneShotAcquisitionFunction):
    def __init__(self, num_auxiliary: int = 2) -> None:
        super().__init__(model=nn.Identity())
        self.num_auxiliary = int(num_auxiliary)
        self.seen_shapes: list[tuple[int, ...]] = []

    def get_augmented_q_batch_size(self, q: int) -> int:
        return int(q) + self.num_auxiliary

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        return X_full[..., : -self.num_auxiliary, :]

    def forward(self, X: Tensor) -> Tensor:
        self.seen_shapes.append(tuple(X.shape))
        scale = X.detach().abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1.0)
        return -(X / scale).square().sum(dim=(-1, -2))


def _transformer(prefix: str, elements: tuple[str, ...], representation: str) -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=list(elements),
        representation=representation,
        reference_element=elements[-1] if representation == "alr" else None,
        pseudocount=1e-8,
        prefix=prefix,
    )
    transformer.fit(
        [
            "".join(elements),
            f"{elements[0]}2{elements[1]}{elements[2]}",
            f"{elements[0]}{elements[1]}2{elements[2]}",
        ]
    )
    return transformer


def _fixed_site(prefix: str, elements: tuple[str, ...], representation: str) -> dict[str, Any]:
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
        "steps": {},
        "min_components": 2,
        "max_components": 2,
        "required_components": (elements[0],),
        "forbidden_components": (),
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "best_subset_max_combinations": 20,
    }


def _variable_site(prefix: str, elements: tuple[str, ...], representation: str) -> dict[str, Any]:
    return {
        **_fixed_site(prefix, elements, representation),
        "total": 65.0,
        "variable_total": True,
        "total_bounds": (40.0, 90.0),
        "total_feature": f"{prefix}__total",
        "bounds": {element: (0.0, 70.0) for element in elements},
    }


def _opt_config() -> OptimizeConfig:
    return OptimizeConfig(
        q=1,
        num_restarts=2,
        raw_samples=16,
        sequential=False,
        optimizer_kwargs={
            "best_subset_strategy": "exact",
            "options": {"maxiter": 15, "batch_limit": 2},
        },
    )


def _model_bounds(names: tuple[str, ...], coordinate_names: set[str]) -> Tensor:
    lower: list[float] = []
    upper: list[float] = []
    for name in names:
        if name == "temperature":
            lower.append(800.0)
            upper.append(1200.0)
        elif name == "pressure":
            lower.append(1.0)
            upper.append(5.0)
        elif name.endswith("__total"):
            lower.append(40.0)
            upper.append(90.0)
        elif name in coordinate_names:
            lower.append(-8.0)
            upper.append(8.0)
        else:
            lower.append(0.0)
            upper.append(1.0)
    return torch.tensor([lower, upper], dtype=torch.double)


def test_ilr_one_shot_optimizes_support_in_raw_fraction_space() -> None:
    elements = ("Al", "Ti", "V")
    transformer = _transformer("alloy", elements, "ilr")
    names = (
        "temperature",
        *transformer.representation_feature_names_,
        "pressure",
    )
    base = _QuadraticOneShot()

    result = optimize_logratio_best_subset(
        base,
        _opt_config(),
        site_name="alloy",
        site_config=_fixed_site("alloy", elements, "ilr"),
        transformer=transformer,
        model_feature_names=names,
        model_bounds=_model_bounds(names, set(transformer.representation_feature_names_)),
        dtype=torch.double,
    )

    fractions = result.raw_candidates[..., result.bridge.fraction_slice]
    assert fractions.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert int((fractions > 1e-6).sum().item()) == 2
    assert fractions[..., 0].item() > 0.0
    assert torch.isfinite(result.candidates).all()
    assert torch.isfinite(torch.as_tensor(result.acq_value)).all()
    assert any(shape[-2] == 3 for shape in base.seen_shapes)  # q=1 + two auxiliary rows.


def test_variable_total_one_shot_optimizes_raw_amount_support_and_total() -> None:
    elements = ("Fe", "Ni", "Co")
    transformer = _transformer("powder", elements, "ilr")
    names = (
        "temperature",
        *transformer.representation_feature_names_,
        "powder__total",
        "pressure",
    )
    base = _QuadraticOneShot()

    result = optimize_variable_total_best_subset(
        base,
        _opt_config(),
        site_name="powder",
        site_config=_variable_site("powder", elements, "ilr"),
        transformer=transformer,
        model_feature_names=names,
        model_bounds=_model_bounds(names, set(transformer.representation_feature_names_)),
        dtype=torch.double,
    )

    amounts = result.raw_candidates[..., list(result.bridge.amount_indices)]
    total = float(amounts.sum())
    assert 40.0 - 1e-6 <= total <= 90.0 + 1e-6
    assert int((amounts > 1e-6).sum().item()) == 2
    assert amounts[..., 0].item() > 0.0
    total_index = names.index("powder__total")
    assert result.candidates[..., total_index].item() == pytest.approx(total, abs=1e-5)
    assert torch.isfinite(torch.as_tensor(result.acq_value)).all()


def test_multi_raw_one_shot_uses_independent_group_supports() -> None:
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    a = _transformer("a", a_elements, "ilr")
    b = _transformer("b", b_elements, "clr")
    sites = {
        "a": _fixed_site("a", a_elements, "ilr"),
        "b": _variable_site("b", b_elements, "clr"),
    }
    names = (
        "temperature",
        *a.representation_feature_names_,
        *b.representation_feature_names_,
        "b__total",
        "pressure",
    )
    coordinate_names = {
        *a.representation_feature_names_,
        *b.representation_feature_names_,
    }
    base = _QuadraticOneShot()

    result = optimize_multi_raw_best_subset(
        base,
        _opt_config(),
        composition_sites=sites,
        composition_transformers={"a": a, "b": b},
        model_feature_names=names,
        model_bounds=_model_bounds(names, coordinate_names),
        dtype=torch.double,
    )

    decision_names = result.bridge.decision_feature_names
    a_indices = [
        index for index, name in enumerate(decision_names) if name.startswith("a__fraction__")
    ]
    b_indices = [
        index for index, name in enumerate(decision_names) if name.startswith("b__amount__")
    ]
    a_values = result.raw_candidates[..., a_indices]
    b_values = result.raw_candidates[..., b_indices]
    assert a_values.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert int((a_values > 1e-6).sum().item()) == 2
    assert int((b_values > 1e-6).sum().item()) == 2
    assert 40.0 - 1e-6 <= float(b_values.sum()) <= 90.0 + 1e-6
    assert torch.isfinite(result.candidates).all()
    assert torch.isfinite(torch.as_tensor(result.acq_value)).all()
