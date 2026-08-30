from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from bochan.api import CandidateRepairConfig, OptimizeConfig
from bochan.composition import CompositionTransformer
from bochan.tabular.composition.logratio_support import (
    RawDecisionAcquisition,
    optimize_logratio_best_subset,
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
            "Al": (0.05, 0.8),
            "Ti": (0.0, 0.8),
            "V": (0.0, 0.8),
            "Cr": (0.0, 0.0),
            "Nb": (0.0, 0.8),
        },
        "steps": {},
        "min_components": 3,
        "max_components": 3,
        "required_components": ("Al",),
        "forbidden_components": ("Cr",),
        "support_selection": "best_subset",
        "variable_total": False,
        "best_subset_strategy": "auto",
        "best_subset_max_combinations": 2,
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
        "phase",
    ]


def _model_bounds(transformer: CompositionTransformer) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    return torch.tensor(
        [
            [800.0, *([-8.0] * width), 1.0, 0.0],
            [1200.0, *([8.0] * width), 5.0, 2.0],
        ],
        dtype=torch.double,
    )


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_prepare_logratio_best_subset_reuses_fraction_support_resolver(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    feature_names = _model_layout(transformer)
    bridge, config, raw_bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site(representation),
        transformer=transformer,
        model_feature_names=feature_names,
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    repair = config.repair_config
    assert repair is not None
    assert repair.support_selection == "best_subset"
    assert repair.k == 2
    assert repair.comp_idx == [2, 3, 5]  # Ti, V, Nb in raw decision space
    assert config.fixed_features == {4: 0.0}  # forbidden Cr
    assert repair.final_sum_constraint is not None
    indices, total = repair.final_sum_constraint
    assert tuple(int(index) for index in indices) == (1, 2, 3, 4, 5)
    assert float(total) == pytest.approx(1.0)
    assert raw_bounds.shape == (2, bridge.decision_dim)
    assert bridge.decision_feature_names[1:6] == (
        "alloy__fraction__Al",
        "alloy__fraction__Ti",
        "alloy__fraction__V",
        "alloy__fraction__Cr",
        "alloy__fraction__Nb",
    )
    assert config.optimizer_kwargs["best_subset_strategy"] == "auto"
    assert config.optimizer_kwargs["best_subset_beam_width"] == 4


def test_process_fixed_features_and_constraints_shift_after_ilr_expansion() -> None:
    transformer = _transformer("ilr")
    model_names = _model_layout(transformer)
    pressure_index = model_names.index("pressure")
    phase_index = model_names.index("phase")
    process_constraint = (
        torch.tensor([pressure_index], dtype=torch.long),
        torch.tensor([1.0], dtype=torch.double),
        2.0,
    )
    config = OptimizeConfig(
        fixed_features={phase_index: 1.0},
        inequality_constraints=[process_constraint],
    )

    bridge, resolved, _ = prepare_logratio_best_subset_config(
        config,
        site_name="alloy",
        site_config=_site("ilr"),
        transformer=transformer,
        model_feature_names=model_names,
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    raw_pressure = bridge.process_index_map[pressure_index]
    raw_phase = bridge.process_index_map[phase_index]
    assert raw_pressure == pressure_index + 1
    assert raw_phase == phase_index + 1
    assert resolved.fixed_features is not None
    assert resolved.fixed_features[raw_phase] == pytest.approx(1.0)
    assert resolved.fixed_features[bridge.fraction_indices[3]] == pytest.approx(0.0)
    assert resolved.inequality_constraints is not None
    found = [
        item
        for item in resolved.inequality_constraints
        if tuple(int(index) for index in item[0]) == (raw_pressure,)
    ]
    assert len(found) == 1


def test_mixed_categorical_process_feature_is_remapped_and_inferred() -> None:
    transformer = _transformer("ilr")
    names = _model_layout(transformer)
    phase_index = names.index("phase")
    train_x = torch.tensor(
        [
            [900.0, 0.1, -0.2, 0.3, 0.0, 2.0, 0.0],
            [950.0, 0.2, -0.1, 0.1, -0.3, 3.0, 1.0],
            [980.0, 0.0, 0.1, -0.2, 0.2, 2.5, 1.0],
        ],
        dtype=torch.double,
    )

    bridge, config, _ = prepare_logratio_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site("ilr"),
        transformer=transformer,
        model_feature_names=names,
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
        model_cat_dims=[phase_index],
        train_x=train_x,
    )

    raw_phase = bridge.process_index_map[phase_index]
    assert str(config.optimizer) == "optimize_acqf_mixed"
    assert config.fixed_features_list == [
        {raw_phase: 0.0},
        {raw_phase: 1.0},
    ]


class _RecordingAcquisition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_x: torch.Tensor | None = None
        self.model = SimpleNamespace()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_x = x
        return -x.square().sum(dim=(-1, -2))


def test_raw_decision_acquisition_evaluates_finite_model_coordinates() -> None:
    transformer = _transformer("ilr")
    bridge, _config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site("ilr"),
        transformer=transformer,
        model_feature_names=_model_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )
    base = _RecordingAcquisition()
    wrapped = RawDecisionAcquisition(base, bridge)
    raw = torch.tensor(
        [[[900.0, 0.6, 0.4, 0.0, 0.0, 0.0, 2.0, 1.0]]],
        dtype=torch.double,
        requires_grad=True,
    )

    value = wrapped(raw)
    value.sum().backward()

    assert base.last_x is not None
    assert base.last_x.shape[-1] == bridge.model_dim
    assert torch.isfinite(base.last_x).all()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_optimize_logratio_best_subset_returns_model_space_and_exact_raw_support() -> None:
    transformer = _transformer("ilr")
    names = _model_layout(transformer)
    base = _RecordingAcquisition()
    captured: dict[str, Any] = {}

    def fake_optimize(*, acqf: Any, bounds: torch.Tensor, config: OptimizeConfig):
        captured["config"] = config
        captured["bounds"] = bounds
        # temperature + five fractions + pressure + phase
        raw = torch.tensor(
            [[900.0, 0.5, 0.3, 0.0, 0.0, 0.2, 2.0, 1.0]],
            dtype=torch.double,
        )
        score = acqf(raw.unsqueeze(-2))
        return raw, score

    result = optimize_logratio_best_subset(
        base,
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site("ilr"),
        transformer=transformer,
        model_feature_names=names,
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
        optimize_fn=fake_optimize,
    )

    assert captured["config"].repair_config.support_selection == "best_subset"
    assert result.raw_candidates.shape[-1] == result.bridge.decision_dim
    assert result.candidates.shape[-1] == result.bridge.model_dim
    fractions = result.raw_candidates[..., result.bridge.fraction_slice]
    assert int((fractions > 1e-12).sum().item()) == 3
    assert fractions[0, 3].item() == 0.0  # forbidden Cr
    assert torch.isfinite(result.candidates).all()
    assert base.last_x is not None
    torch.testing.assert_close(base.last_x.squeeze(-2), result.candidates)


def test_explicit_logratio_coordinate_fixed_feature_is_rejected() -> None:
    transformer = _transformer("ilr")
    names = _model_layout(transformer)
    coordinate_index = names.index(transformer.representation_feature_names_[0])

    with pytest.raises(ValueError, match="coordinate indices"):
        prepare_logratio_best_subset_config(
            OptimizeConfig(fixed_features={coordinate_index: 0.0}),
            site_name="alloy",
            site_config=_site("ilr"),
            transformer=transformer,
            model_feature_names=names,
            model_bounds=_model_bounds(transformer),
            dtype=torch.double,
            device=None,
        )


def test_existing_generic_sparse_group_remains_an_explicit_conflict() -> None:
    transformer = _transformer("ilr")
    names = _model_layout(transformer)
    pressure = names.index("pressure")
    config = OptimizeConfig(
        repair_config=CandidateRepairConfig(
            comp_idx=[pressure],
            k=1,
            support_selection="topk",
        )
    )

    with pytest.raises(ValueError, match="owns CandidateRepairConfig.comp_idx"):
        prepare_logratio_best_subset_config(
            config,
            site_name="alloy",
            site_config=_site("ilr"),
            transformer=transformer,
            model_feature_names=names,
            model_bounds=_model_bounds(transformer),
            dtype=torch.double,
            device=None,
        )


def test_explicit_optimizer_kwargs_override_site_defaults() -> None:
    transformer = _transformer("ilr")
    _bridge, config, _bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(
            optimizer_kwargs={
                "best_subset_strategy": "exact",
                "best_subset_beam_width": 99,
            }
        ),
        site_name="alloy",
        site_config=_site("ilr"),
        transformer=transformer,
        model_feature_names=_model_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    assert config.optimizer_kwargs["best_subset_strategy"] == "exact"
    assert config.optimizer_kwargs["best_subset_beam_width"] == 99


def test_model_space_final_postprocess_is_rejected() -> None:
    transformer = _transformer("ilr")
    config = OptimizeConfig(final_candidate_postprocess=lambda value: value)

    with pytest.raises(ValueError, match="final_candidate_postprocess"):
        prepare_logratio_best_subset_config(
            config,
            site_name="alloy",
            site_config=_site("ilr"),
            transformer=transformer,
            model_feature_names=_model_layout(transformer),
            model_bounds=_model_bounds(transformer),
            dtype=torch.double,
            device=None,
        )
