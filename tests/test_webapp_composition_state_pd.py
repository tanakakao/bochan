from __future__ import annotations

from pathlib import Path

import numpy as np

from bochan.serving.webapp.composition_pd_compat import (
    _aggregate_prediction,
    _vary_fraction_rows,
)
from bochan.serving.webapp.composition_visualization import _CompositionContext


def _context() -> _CompositionContext:
    return _CompositionContext(
        site_name="composition",
        column="formula",
        prefix="formula",
        elements=("Fe", "Co", "Ni"),
        fraction_features=(
            "formula__fraction__Fe",
            "formula__fraction__Co",
            "formula__fraction__Ni",
        ),
        config={
            "column": "formula",
            "normalization": "atomic_fraction",
            "total": 1.0,
            "precision": 6,
            "bounds": {
                "Fe": (0.0, 1.0),
                "Co": (0.0, 1.0),
                "Ni": (0.0, 1.0),
            },
            "min_components": 1,
            "max_components": 3,
            "required_components": (),
        },
    )


def test_pd_fraction_axis_preserves_each_rows_remaining_ratio() -> None:
    baselines = np.asarray(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.1, 0.5],
        ]
    )

    fractions, valid = _vary_fraction_rows(
        _context(),
        baselines,
        "formula__fraction__Fe",
        0.6,
        mode="proportional",
        balance_element=None,
    )

    assert valid.tolist() == [True, True]
    assert np.allclose(fractions[:, 0], 0.6)
    assert np.allclose(fractions.sum(axis=1), 1.0)
    assert np.isclose(fractions[0, 1] / fractions[0, 2], 0.3 / 0.5)
    assert np.isclose(fractions[1, 1] / fractions[1, 2], 0.1 / 0.5)


def test_pd_balance_element_absorbs_residual_per_row() -> None:
    baselines = np.asarray(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.1, 0.5],
        ]
    )

    fractions, valid = _vary_fraction_rows(
        _context(),
        baselines,
        "formula__fraction__Fe",
        0.4,
        mode="balance",
        balance_element="Ni",
    )

    assert valid.tolist() == [True, True]
    assert np.allclose(fractions[:, 0], 0.4)
    assert np.allclose(fractions[:, 1], [0.3, 0.1])
    assert np.allclose(fractions[:, 2], [0.3, 0.5])


def test_pd_aggregate_uses_total_predictive_variance() -> None:
    mean, std = _aggregate_prediction(
        np.asarray([1.0, 3.0]),
        np.asarray([0.5, 0.5]),
    )

    assert np.isclose(mean, 2.0)
    assert np.isclose(std, np.sqrt(1.25))


def test_dataset_state_resets_and_hides_stale_composition_settings() -> None:
    source = Path("web/src/compositionDatasetState.ts").read_text(encoding="utf-8")
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")

    assert 'ACTIVE_DATASET_KEY = "bochan-web-composition-dataset-id"' in source
    assert "resetCompositionSelection()" in source
    assert 'settings.enabled && String(settings.column ?? "").trim()' in source
    assert ".composition-model-settings-host, .composition-constraint-settings-host" in source
    assert "installCompositionDatasetState();" in main


def test_composition_pd_adapter_is_installed_after_visualization_compat() -> None:
    init_source = Path("src/bochan/serving/webapp/__init__.py").read_text(
        encoding="utf-8"
    )
    compat_source = Path(
        "src/bochan/serving/webapp/composition_pd_compat.py"
    ).read_text(encoding="utf-8")

    visualization_position = init_source.index(
        "install_composition_visualization_compat()"
    )
    pd_position = init_source.index("install_composition_pd_compat()")
    assert visualization_position < pd_position
    assert "_build_partial_dependence_1d" in compat_source
    assert "各学習行の組成と他の説明変数を保持" in compat_source
