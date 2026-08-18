"""Tests for column-name objective selection in the tabular candidate API."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.tabular.optimizer.candidates import (
    CandidateService,
    _resolve_named_objective_outputs,
)


class _NoopConstraintResolver:
    def named_constraints(self, constraints, sites, *args):
        return []

    def merge_optimize_config(self, opt_config, constraints):
        return opt_config


def _candidate_service() -> CandidateService:
    service = object.__new__(CandidateService)
    service.composition = SimpleNamespace(sites=[], transformers=[])
    service.total_resolver = _NoopConstraintResolver()
    service.element_resolver = _NoopConstraintResolver()
    service.total_constraints = []
    service.element_constraints = []
    return service


def _owner(*target_names: str):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            target_names=list(target_names),
            target_category_maps={},
        )
    )


def test_resolve_named_objective_outputs_supports_names_and_indices() -> None:
    resolved = _resolve_named_objective_outputs(
        {
            "objective_output": "property",
            "objective_outputs": ["property", "cost"],
            "objective_specs": [
                {"output": "cost", "direction": "minimize"},
            ],
            "objective_config": {
                "output": "property",
                "outputs": ["property", "cost"],
            },
        },
        ["property", "cost"],
    )

    assert resolved["objective_output"] == 0
    assert resolved["objective_outputs"] == [0, 1]
    assert resolved["objective_specs"][0]["output"] == 1
    assert resolved["objective_config"]["output"] == 0
    assert resolved["objective_config"]["outputs"] == [0, 1]

    integer_resolved = _resolve_named_objective_outputs(
        {"objective_output": 1},
        ["property", "cost"],
    )
    assert integer_resolved["objective_output"] == 1


def test_prepare_configs_resolves_explicit_tabular_objective_name() -> None:
    service = _candidate_service()

    acq_config, _ = service._prepare_configs(
        _owner("property"),
        {"name": "qlogei"},
        None,
        {
            "objective_mode": "scalar",
            "objective_output": "property",
            "objective_direction": "maximize",
        },
    )

    assert acq_config.objective_config is not None
    assert acq_config.objective_config.output == 0
    assert acq_config.objective_config.direction == "maximize"


def test_prepare_configs_keeps_single_output_objective_implicit_when_omitted() -> None:
    service = _candidate_service()

    acq_config, _ = service._prepare_configs(
        _owner("property"),
        {"name": "qlogei"},
        None,
        {},
    )

    assert acq_config.objective_config is None


def test_named_objective_output_rejects_unknown_target_name() -> None:
    with pytest.raises(KeyError, match="Unknown column"):
        _resolve_named_objective_outputs(
            {"objective_output": "missing"},
            ["property"],
        )
