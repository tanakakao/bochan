from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bochan.serving.webapp.composition_multielement_ternary import (
    _extend_multielement_ternary_options,
    _ternary_slice_grid,
    _ternary_sum_value,
)
from bochan.serving.webapp.composition_visualization import (
    _CompositionContext,
    _extend_visualization_options,
    _resolve_fraction_matrix,
)
from bochan.serving.webapp.composition_visualization_compat import (
    _constant_composition_grid,
    _object_backed_string_columns,
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


def _four_element_context() -> _CompositionContext:
    return _CompositionContext(
        site_name="composition",
        column="formula",
        prefix="formula",
        elements=("Fe", "Co", "Ni", "Cr"),
        fraction_features=(
            "formula__fraction__Fe",
            "formula__fraction__Co",
            "formula__fraction__Ni",
            "formula__fraction__Cr",
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
                "Cr": (0.0, 1.0),
            },
            "min_components": 1,
            "max_components": 4,
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


def test_constant_composition_grid_preserves_baseline_for_ordinary_axes() -> None:
    baseline = np.asarray([0.2, 0.3, 0.5])

    fractions, valid = _constant_composition_grid(_context(), baseline, 4)

    assert valid.tolist() == [True, True, True, True]
    assert np.allclose(fractions, np.tile(baseline, (4, 1)))


def test_generated_string_categories_are_object_backed_for_pandas3() -> None:
    source = pd.DataFrame(
        {
            "formula": pd.Series(["Fe2O3", "Al2O3"], dtype="string"),
            "category": pd.Series(["A", "B"], dtype="string"),
            "temperature": [900.0, 1000.0],
        }
    )

    converted = _object_backed_string_columns(source)

    assert converted["formula"].dtype == object
    assert converted["category"].dtype == object
    assert pd.api.types.is_float_dtype(converted["temperature"].dtype)
    assert converted.to_dict("list") == source.to_dict("list")


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


def test_multielement_ternary_options_use_first_three_default_fractions() -> None:
    context = _four_element_context()

    class Transformer:
        prefix = context.prefix

        @staticmethod
        def _require_fitted() -> tuple[str, ...]:
            return context.elements

    session = SimpleNamespace(
        tabular_optimizer=SimpleNamespace(
            composition_sites={context.site_name: context.config},
            composition_transformers_={context.site_name: Transformer()},
        )
    )
    options = _extend_multielement_ternary_options(
        {
            "ternary_groups": [],
            "composition": {
                "features": [
                    {"name": context.fraction_features[0], "default": 0.25},
                    {"name": context.fraction_features[1], "default": 0.20},
                    {"name": context.fraction_features[2], "default": 0.15},
                    {"name": context.fraction_features[3], "default": 0.40},
                ]
            },
        },
        session,
    )

    assert options["ternary_groups"] == [
        {
            "features": list(context.fraction_features[:3]),
            "sum_value": 0.6,
        }
    ]


def test_multielement_ternary_grid_preserves_unplotted_fraction() -> None:
    context = _four_element_context()
    baseline = np.asarray([0.25, 0.20, 0.15, 0.40])
    features = list(context.fraction_features[:3])
    sum_value = _ternary_sum_value(
        context,
        {"sum_value": 0.6},
        baseline,
        features,
    )
    grid = _ternary_slice_grid(sum_value, 5)
    fractions, valid = _resolve_fraction_matrix(
        context,
        baseline=baseline,
        axis_values={
            features[0]: grid[:, 0],
            features[1]: grid[:, 1],
            features[2]: grid[:, 2],
        },
        mode="proportional",
        balance_element=None,
    )

    assert valid.all()
    assert np.allclose(grid.sum(axis=1), 0.6)
    assert np.allclose(fractions.sum(axis=1), 1.0)
    assert np.allclose(fractions[:, 3], 0.4)


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


def test_visualization_compat_routes_ordinary_and_multielement_ternary_axes() -> None:
    compatibility = Path(
        "src/bochan/serving/webapp/composition_visualization_compat.py"
    ).read_text(encoding="utf-8")
    ternary_backend = Path(
        "src/bochan/serving/webapp/composition_multielement_ternary.py"
    ).read_text(encoding="utf-8")
    frontend = Path("web/src/compositionVisualizationGuard.ts").read_text(
        encoding="utf-8"
    )
    web_init = Path("src/bochan/serving/webapp/__init__.py").read_text(
        encoding="utf-8"
    )
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")

    assert 'kind in {"1d", "2d"} and not composition_axes' in compatibility
    assert "_build_ordinary_axis_composition_visualization" in compatibility
    assert "_object_backed_string_columns(source)" in compatibility
    assert "_build_multielement_ternary_visualization" in ternary_backend
    assert "_ternary_slice_grid(sum_value, divisions)" in ternary_backend
    assert "len(features) == 3 and len(composition_axes) == 3" in ternary_backend
    assert "fractionOptions.size >= 3" in frontend
    assert 'kindSelect.value = "1d"' in frontend
    assert "install_composition_multielement_ternary" in web_init
    assert "installCompositionVisualizationGuard" in main
