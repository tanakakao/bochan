from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bochan.serving.webapp.composition_visualization import (
    _CompositionContext,
    _extend_visualization_options,
    _resolve_fraction_matrix,
)


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
                "Fe": (0.1, 0.8),
                "Co": (0.0, 0.8),
                "Ni": (0.0, 0.8),
            },
            "min_components": 1,
            "max_components": 3,
            "required_components": (),
        },
    )


def test_proportional_composition_axis_preserves_remaining_ratio() -> None:
    context = _context()
    fractions, valid = _resolve_fraction_matrix(
        context,
        baseline=np.asarray([0.2, 0.3, 0.5]),
        axis_values={"formula__fraction__Fe": np.asarray([0.4, 0.6])},
        mode="proportional",
        balance_element=None,
    )

    assert valid.tolist() == [True, True]
    assert np.allclose(fractions.sum(axis=1), 1.0)
    assert np.allclose(fractions[:, 1] / fractions[:, 2], 0.3 / 0.5)
    assert np.allclose(fractions[:, 0], [0.4, 0.6])


def test_balance_element_absorbs_remaining_fraction() -> None:
    context = _context()
    fractions, valid = _resolve_fraction_matrix(
        context,
        baseline=np.asarray([0.2, 0.3, 0.5]),
        axis_values={"formula__fraction__Fe": np.asarray([0.4])},
        mode="balance",
        balance_element="Ni",
    )

    assert valid.tolist() == [True]
    assert np.allclose(fractions[0], [0.4, 0.3, 0.3])


def test_visualization_options_add_element_fraction_axes() -> None:
    context = _context()

    class Transformer:
        prefix = context.prefix

        @staticmethod
        def _require_fitted() -> tuple[str, ...]:
            return context.elements

    restored = pd.DataFrame(
        {
            "formula": ["Fe2Co3Ni5", "Fe4Co4Ni2"],
            "formula__fraction__Fe": [0.2, 0.4],
            "formula__fraction__Co": [0.3, 0.4],
            "formula__fraction__Ni": [0.5, 0.2],
            "temperature": [900.0, 1000.0],
            "property": [1.0, 2.0],
        }
    )

    optimizer = SimpleNamespace(
        composition_sites={context.site_name: context.config},
        composition_transformers_={context.site_name: Transformer()},
        transform_compositions=lambda data: data,
        inverse_compositions=lambda data, **kwargs: restored,
    )
    session = SimpleNamespace(
        tabular_optimizer=optimizer,
        data=pd.DataFrame(
            {
                "formula": ["Fe2Co3Ni5", "Fe4Co4Ni2"],
                "temperature": [900.0, 1000.0],
                "property": [1.0, 2.0],
            }
        ),
    )
    options = _extend_visualization_options(
        {
            "feature_columns": ["formula", "temperature"],
            "numeric_features": ["temperature"],
            "feature_controls": {
                "formula": {
                    "kind": "categorical",
                    "values": ["Fe2Co3Ni5", "Fe4Co4Ni2"],
                    "default": "Fe2Co3Ni5",
                },
                "temperature": {
                    "kind": "numeric",
                    "min": 900.0,
                    "max": 1000.0,
                    "default": 950.0,
                },
            },
            "ternary_groups": [],
        },
        session,
    )

    assert options["numeric_features"] == [
        "temperature",
        "formula__fraction__Fe",
        "formula__fraction__Co",
        "formula__fraction__Ni",
    ]
    assert options["feature_labels"]["formula__fraction__Fe"] == "Fe 比率"
    assert options["composition"]["elements"] == ["Fe", "Co", "Ni"]
    assert options["ternary_groups"] == [
        {
            "features": [
                "formula__fraction__Fe",
                "formula__fraction__Co",
                "formula__fraction__Ni",
            ],
            "sum_value": 1.0,
        }
    ]


def test_web_source_exposes_composition_axis_controls() -> None:
    source = Path("web/src/InteractiveResultPlots.tsx").read_text(encoding="utf-8")
    type_source = Path("web/src/compositionVisualizationTypes.ts").read_text(
        encoding="utf-8"
    )

    assert "残りの元素比を維持" in source
    assert "バランス元素で調整" in source
    assert "compositionFeatures" in source
    assert "feature_labels" in type_source
    assert "fraction_features" in type_source
