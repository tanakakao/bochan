from __future__ import annotations

from pathlib import Path

import pytest

from bochan.serving.webapp.composition.support import (
    composition_site,
    normalize_web_composition_settings,
)


def _settings(*, variable_total: bool = False, steps: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "enabled": True,
        "column": "formula",
        "elements": ["Al", "Ti", "V", "Nb"],
        "representation": "ilr",
        "normalization": "atomic_fraction",
        "bounds": {
            "Al": [0.05, 0.8] if not variable_total else [5.0, 80.0],
            "Ti": [0.0, 0.8] if not variable_total else [0.0, 80.0],
            "V": [0.0, 0.8] if not variable_total else [0.0, 80.0],
            "Nb": [0.0, 0.8] if not variable_total else [0.0, 80.0],
        },
        "steps": (
            {"Al": 0.05, "Ti": 0.05, "V": 0.05, "Nb": 0.05}
            if steps and not variable_total
            else {"Al": 5.0, "Ti": 5.0, "V": 5.0, "Nb": 5.0}
            if steps
            else {}
        ),
        "min_components": 2,
        "max_components": 4,
        "required_components": ["Al"],
        "forbidden_components": [],
        "support_selection": "best_subset",
        "best_subset_strategy": "auto",
        "best_subset_max_combinations": 20,
        "best_subset_beam_width": 4,
        "best_subset_beam_steps": 3,
        "best_subset_max_evaluations": 20,
    }
    if variable_total:
        result["total"] = None
        result["total_bounds"] = [40.0, 100.0]
    else:
        result["total"] = 1.0
    return result


def test_web_normalizer_accepts_fixed_total_variable_cardinality() -> None:
    config = normalize_web_composition_settings(_settings())

    assert config["support_selection"] == "best_subset"
    assert config["min_components"] == 2
    assert config["max_components"] == 4
    site = composition_site(config)
    assert site["min_components"] == 2
    assert site["max_components"] == 4


def test_web_normalizer_accepts_variable_total_variable_cardinality() -> None:
    config = normalize_web_composition_settings(_settings(variable_total=True))

    assert config["variable_total"] is True
    assert config["total_bounds"] == pytest.approx((40.0, 100.0))
    assert config["min_components"] == 2
    assert config["max_components"] == 4
    site = composition_site(config)
    assert site["total_bounds"] == pytest.approx((40.0, 100.0))
    assert site["min_components"] == 2
    assert site["max_components"] == 4


@pytest.mark.parametrize("variable_total", [False, True])
def test_web_normalizer_accepts_step_grid_with_variable_cardinality(
    variable_total: bool,
) -> None:
    config = normalize_web_composition_settings(
        _settings(variable_total=variable_total, steps=True)
    )

    assert config["min_components"] == 2
    assert config["max_components"] == 4
    assert set(config["steps"]) == {"Al", "Ti", "V", "Nb"}
    site = composition_site(config)
    assert site["min_components"] == 2
    assert site["max_components"] == 4
    assert site["steps"] == config["steps"]


def test_react_best_subset_ui_exposes_stepped_variable_cardinality() -> None:
    source = Path("web/src/components/CompositionBestSubsetSettings.tsx").read_text(
        encoding="utf-8"
    )

    assert "使用元素数・最小" in source
    assert "使用元素数・最大" in source
    assert "combinationRangeCount" in source
    assert "使用元素数を1つに固定してください" not in source
    assert "step付きBest SubsetのMILP投影は現在exact-cardinalityのみ対応" not in source
    assert "step指定時も、各supportのcardinalityを保持したまま実験格子へMILP投影" in source


def test_workbench_validation_accepts_step_cardinality_range() -> None:
    source = Path("web/src/context/workbenchValidation.ts").read_text(encoding="utf-8")

    assert "combinationRangeCount" in source
    assert "if (settings.maxComponents === null) return false;" in source
    assert "if (hasSteps && effectiveMin !== effectiveMax) return false;" not in source
    assert "settings.minComponents !== settings.maxComponents" not in source
