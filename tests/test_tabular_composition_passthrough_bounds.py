import pandas as pd

from bochan.tabular import TabularBayesianOptimizer


def _frame():
    return pd.DataFrame({
        "formula": ["Fe0.5Co0.3Ni0.2", "Fe0.4Co0.4Ni0.2"],
        "temperature": [900.0, 1000.0], "furnace": ["A", "B"],
        "property": [10.0, 12.0],
    })


def _site(**values):
    config = {"column": "formula", "elements": ["Fe", "Co", "Ni"],
              "representation": "ilr"}
    config.update(values)
    return {"formula": config}


def _bounds(optimizer, frame, bounds=None):
    transformed = optimizer.composition.prepare_frame(frame, fit_transformers=True)
    return transformed, optimizer.composition.expanded_bounds(bounds, transformed)


def test_single_site_infers_passthrough_and_descriptor_bounds():
    optimizer = TabularBayesianOptimizer(
        input_cols=["formula", "temperature"], target_cols="property",
        composition_sites=_site(
            include_descriptors=True,
            descriptor_properties=["atomic_number", "atomic_weight", "electronegativity"],
            element_properties={"electronegativity": {"Fe": 1.83, "Co": 1.88, "Ni": 1.91}},
        ),
    )
    transformed, bounds = _bounds(optimizer, _frame())
    transformer = optimizer.composition.transformers["formula"]
    representation = set(transformer._representation_names(transformer._require_fitted()))
    descriptors = set(transformer.feature_names_ or ()) - representation
    assert bounds["temperature"] == [900.0, 1000.0]
    assert representation and descriptors
    assert representation.issubset(bounds) and descriptors.issubset(bounds)
    assert descriptors.issubset(transformed.columns)


def test_explicit_passthrough_bound_is_preserved():
    optimizer = TabularBayesianOptimizer(
        input_cols=["formula", "temperature"], target_cols="property",
        composition_sites=_site(),
    )
    _, bounds = _bounds(optimizer, _frame(), {"temperature": [850.0, 1050.0]})
    assert bounds["temperature"] == [850.0, 1050.0]


def test_categorical_passthrough_bound_matches_label_encoding():
    optimizer = TabularBayesianOptimizer(
        input_cols=["formula", "temperature", "furnace"], target_cols="property",
        categorical_cols=["furnace"], composition_sites=_site(),
    )
    _, bounds = _bounds(optimizer, _frame())
    assert bounds["temperature"] == [900.0, 1000.0]
    assert bounds["furnace"] == [0.0, 1.0]


def test_multi_site_composition_infers_passthrough_bounds():
    frame = pd.DataFrame({
        "A_formula": ["La0.6Sr0.4", "La0.7Sr0.3"],
        "temperature": [900.0, 1000.0], "property": [10.0, 12.0],
    })
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_formula", "temperature"], target_cols="property",
        composition_sites={"A": {"column": "A_formula", "elements": ["La", "Sr"],
                                  "representation": "ilr"}},
    )
    _, bounds = _bounds(optimizer, frame)
    assert bounds["temperature"] == [900.0, 1000.0]
    assert "A__ilr__1" in bounds
