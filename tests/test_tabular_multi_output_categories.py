from __future__ import annotations

import pytest

from bochan.tabular import TabularBayesianOptimizer


def _hybrid_model_config(category_fields: dict[str, object]) -> dict[str, object]:
    return {
        "task_type": "hybrid",
        "model_type": "base",
        "multi_output_config": {
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
                    **category_fields,
                },
            ],
        },
    }


def test_init_extracts_ordered_categories_from_model_config() -> None:
    optimizer = TabularBayesianOptimizer(
        model_config=_hybrid_model_config(
            {"ordered_categories": ["low", "medium", "high"]}
        ),
        input_cols=["x1"],
        target_cols=["property", "quality"],
    )

    assert optimizer.data_config.target_categorical_cols == ["quality"]
    assert optimizer.data_config.target_category_maps == {
        "quality": {"low": 0, "medium": 1, "high": 2}
    }

    output_configs = optimizer.model_config.multi_output_config.output_configs
    assert output_configs is not None
    assert "ordered_categories" not in output_configs[1]


def test_init_extracts_categories_from_direct_multi_output_config() -> None:
    optimizer = TabularBayesianOptimizer(
        task_type="hybrid",
        model_type="base",
        multi_output_config={
            "use_hybrid": True,
            "output_configs": [
                {
                    "name": "property",
                    "task_type": "regression",
                },
                {
                    "name": "defect",
                    "task_type": "binary",
                    "categories": ["normal", "defect"],
                },
            ],
        },
        input_cols=["x1"],
        target_cols=["property", "defect"],
    )

    assert optimizer.data_config.target_categorical_cols == ["defect"]
    assert optimizer.data_config.target_category_maps == {
        "defect": {"normal": 0, "defect": 1}
    }


def test_init_accepts_explicit_category_map() -> None:
    optimizer = TabularBayesianOptimizer(
        model_config=_hybrid_model_config(
            {"category_map": {10: 0, 20: 1, 30: 2}}
        ),
        input_cols=["x1"],
        target_cols=["property", "quality"],
    )

    assert optimizer.data_config.target_category_maps == {
        "quality": {10: 0, 20: 1, 30: 2}
    }


def test_init_merges_matching_explicit_target_category_map() -> None:
    optimizer = TabularBayesianOptimizer(
        model_config=_hybrid_model_config(
            {"ordered_categories": ["low", "medium", "high"]}
        ),
        input_cols=["x1"],
        target_cols=["property", "quality"],
        target_categorical_cols=["quality"],
        target_category_maps={
            "quality": {"low": 0, "medium": 1, "high": 2}
        },
    )

    assert optimizer.data_config.target_categorical_cols == ["quality"]
    assert optimizer.data_config.target_category_maps == {
        "quality": {"low": 0, "medium": 1, "high": 2}
    }


def test_init_rejects_conflicting_explicit_target_category_map() -> None:
    with pytest.raises(ValueError, match="conflicts with target_category_maps"):
        TabularBayesianOptimizer(
            model_config=_hybrid_model_config(
                {"ordered_categories": ["low", "medium", "high"]}
            ),
            input_cols=["x1"],
            target_cols=["property", "quality"],
            target_category_maps={
                "quality": {"high": 0, "medium": 1, "low": 2}
            },
        )


def test_init_rejects_multiple_category_declarations() -> None:
    with pytest.raises(ValueError, match="Specify only one"):
        TabularBayesianOptimizer(
            model_config=_hybrid_model_config(
                {
                    "ordered_categories": ["low", "medium", "high"],
                    "category_map": {"low": 0, "medium": 1, "high": 2},
                }
            ),
            input_cols=["x1"],
            target_cols=["property", "quality"],
        )


def test_init_rejects_nonconsecutive_category_map() -> None:
    with pytest.raises(ValueError, match="consecutive integers starting at 0"):
        TabularBayesianOptimizer(
            model_config=_hybrid_model_config(
                {"category_map": {"low": 0, "medium": 2, "high": 3}}
            ),
            input_cols=["x1"],
            target_cols=["property", "quality"],
        )
