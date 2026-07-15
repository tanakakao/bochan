"""FastAPI coverage for tabular category metadata and string constraints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from bochan.serving.fastapi.schemas import (
    AcquisitionConfigSchema,
    ModelConfigSchema,
    OutputConfigSchema,
)
from bochan.serving.fastapi.tabular_compat import (
    bind_category_metadata,
    to_acquisition_config,
    to_model_config,
)


def _model_schema() -> ModelConfigSchema:
    return ModelConfigSchema(
        task_type="hybrid",
        model_type="base",
        multi_output_config={
            "use_hybrid": True,
            "output_configs": [
                {
                    "name": "property",
                    "task_type": "regression",
                    "model_type": "base",
                },
                {
                    "name": "quality",
                    "task_type": "ordinal",
                    "model_type": "base",
                    "ordered_categories": ["a", "b", "c"],
                },
            ],
        },
    )


def _optimizer_with_category_metadata():
    model_config = to_model_config(_model_schema())
    optimizer = SimpleNamespace(
        model=SimpleNamespace(output_names=["property", "quality"]),
        model_config=model_config,
    )
    bind_category_metadata(optimizer, model_config)
    return optimizer, model_config


def test_output_config_schema_advertises_tabular_category_fields() -> None:
    properties = OutputConfigSchema.model_json_schema()["properties"]

    assert "ordered_categories" in properties
    assert "categories" in properties
    assert "category_map" in properties


def test_fastapi_model_converter_strips_and_retains_ordered_categories() -> None:
    optimizer, model_config = _optimizer_with_category_metadata()

    output_config = model_config.multi_output_config.output_configs[1]
    assert output_config.name == "quality"
    assert not hasattr(output_config, "ordered_categories")

    maps = optimizer._bochan_fastapi_target_category_maps
    assert maps == {"quality": {"a": 0, "b": 1, "c": 2}}


def test_fastapi_resolves_string_ordinal_rank() -> None:
    optimizer, _ = _optimizer_with_category_metadata()
    schema = AcquisitionConfigSchema(
        name="ei",
        outcome_constraint_config={
            "constraints": [
                {
                    "kind": "ordinal_rank",
                    "output": "quality",
                    "rank": "b",
                    "sense": "ge",
                    "probability_threshold": 0.8,
                }
            ]
        },
    )

    config = to_acquisition_config(schema, optimizer=optimizer)
    constraint = config.outcome_constraint_config.constraints[0]

    assert constraint.output == "quality"
    assert constraint.rank == 1
    assert constraint.probability_threshold == pytest.approx(0.8)


def test_fastapi_resolves_string_target_class() -> None:
    optimizer, _ = _optimizer_with_category_metadata()
    schema = AcquisitionConfigSchema(
        name="ei",
        outcome_constraint_config={
            "constraints": [
                {
                    "output": "quality",
                    "target_class": "c",
                    "threshold": 0.7,
                    "sense": "ge",
                }
            ]
        },
    )

    config = to_acquisition_config(schema, optimizer=optimizer)
    constraint = config.outcome_constraint_config.constraints[0]

    assert constraint.target_class == 2


def test_fastapi_rejects_unknown_string_ordinal_rank() -> None:
    optimizer, _ = _optimizer_with_category_metadata()
    schema = AcquisitionConfigSchema(
        name="ei",
        outcome_constraint_config={
            "constraints": [
                {
                    "kind": "ordinal_rank",
                    "output": "quality",
                    "rank": "unknown",
                    "sense": "ge",
                    "probability_threshold": 0.8,
                }
            ]
        },
    )

    with pytest.raises(KeyError, match="Available labels"):
        to_acquisition_config(schema, optimizer=optimizer)


def test_output_config_schema_rejects_conflicting_category_declarations() -> None:
    with pytest.raises(ValueError, match="Specify only one"):
        OutputConfigSchema(
            name="quality",
            task_type="ordinal",
            ordered_categories=["a", "b", "c"],
            category_map={"a": 0, "b": 1, "c": 2},
        )
