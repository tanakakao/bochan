from __future__ import annotations

import numpy as np

from bochan.serving.webapp.composition_element_importance_figures import (
    append_element_importance_figures,
)
from bochan.serving.webapp.composition_feature_importance import (
    _resolve_perturbed_fractions,
)
from bochan.serving.webapp.composition_feature_importance_views import (
    _replace_predictive_summary,
)


def test_element_perturbation_preserves_other_element_ratios() -> None:
    baseline = np.asarray(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
        ]
    )
    proposed = baseline.copy()
    proposed[:, 0] = proposed[::-1, 0]

    resolved, joint = _resolve_perturbed_fractions(
        baseline,
        proposed,
        mode="proportional",
    )

    assert joint is False
    assert np.allclose(resolved.sum(axis=1), 1.0)
    assert np.allclose(resolved[:, 0], [0.4, 0.2])
    assert np.allclose(resolved[0, 1] / resolved[0, 2], 0.3 / 0.5)
    assert np.allclose(resolved[1, 1] / resolved[1, 2], 0.4 / 0.2)


def test_joint_composition_permutation_keeps_observed_rows() -> None:
    baseline = np.asarray(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
        ]
    )

    resolved, joint = _resolve_perturbed_fractions(
        baseline,
        baseline[::-1],
    )

    assert joint is True
    assert np.allclose(resolved, baseline[::-1])


def test_predictive_summary_replaces_ilr_coordinates_by_composition_total() -> None:
    result = {
        "feature_importance_summary": [
            {
                "output_name": "property",
                "importance_kind": "predictive",
                "method": "permutation",
                "feature": "temperature",
                "mean": 0.2,
                "normalized_mean": 0.4,
            },
            {
                "output_name": "property",
                "importance_kind": "predictive",
                "method": "permutation",
                "feature": "formula__ilr__1",
                "mean": 0.1,
            },
            {
                "output_name": "property",
                "importance_kind": "predictive",
                "method": "permutation",
                "feature": "formula__ilr__2",
                "mean": 0.15,
            },
        ]
    }
    payload = {
        "coordinate_features": ["formula__ilr__1", "formula__ilr__2"],
        "overall": [
            {
                "output_name": "property",
                "importance_kind": "predictive",
                "method": "permutation",
                "feature": "組成全体",
                "feature_type": "group",
                "role": "composition",
                "mean": 0.3,
                "std": 0.02,
            }
        ],
    }

    rows = _replace_predictive_summary(result, payload)
    by_feature = {row["feature"]: row for row in rows}

    assert set(by_feature) == {"temperature", "組成全体"}
    assert by_feature["組成全体"]["rank"] == 1
    assert by_feature["temperature"]["rank"] == 2
    assert np.isclose(
        by_feature["組成全体"]["normalized_mean"]
        + by_feature["temperature"]["normalized_mean"],
        1.0,
    )


def test_element_importance_adds_existing_panel_compatible_figure() -> None:
    result = {
        "composition_feature_importance": {
            "mode_label": "残りの元素比を維持",
            "elements": [
                {
                    "output_name": "property",
                    "feature": "Fe",
                    "label": "Fe 比率",
                    "mean": 0.4,
                    "std": 0.03,
                    "normalized_mean": 0.8,
                    "metric_name": "rmse",
                    "baseline_metric": 0.1,
                    "n_repeats": 10,
                },
                {
                    "output_name": "property",
                    "feature": "Co",
                    "label": "Co 比率",
                    "mean": 0.1,
                    "std": 0.01,
                    "normalized_mean": 0.2,
                    "metric_name": "rmse",
                    "baseline_metric": 0.1,
                    "n_repeats": 10,
                },
            ],
        },
        "feature_importance_visualizations": [],
    }

    append_element_importance_figures(result)

    figures = result["feature_importance_visualizations"]
    assert len(figures) == 1
    assert "predictive" in figures[0]["id"]
    assert figures[0]["title"] == "property: 組成内の元素別影響度"
    assert figures[0]["figure"]["data"][0]["y"] == ["Fe 比率", "Co 比率"]
