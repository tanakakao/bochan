import numpy as np
import pandas as pd
import pytest

from bochan.tabular.composition import (
    CompositionColumnConfig,
    CompositionDescriptorCalculator,
    CompositionSearchSpace,
    CompositionTabularPreprocessor,
    CompositionTransformer,
    SimplexTransform,
    format_formula,
    parse_formula,
)


def test_parse_formula_supports_groups_and_hydrates():
    assert parse_formula("(La0.6Sr0.4)(Co0.2Fe0.8)O3") == {
        "La": 0.6,
        "Sr": 0.4,
        "Co": 0.2,
        "Fe": 0.8,
        "O": 3.0,
    }
    assert parse_formula("CuSO4·5H2O") == {"Cu": 1.0, "S": 1.0, "O": 9.0, "H": 10.0}


def test_formula_formatter_is_deterministic():
    formula = format_formula({"Co": 0.2, "Fe": 0.5, "Ni": 0.3}, order=["Fe", "Ni", "Co"])
    assert formula == "Fe0.5Ni0.3Co0.2"


def test_formula_formatter_preserves_small_nonzero_components_at_precision():
    formula = format_formula({"Fe": 0.999999, "Co": 0.000001}, order=["Fe", "Co"], precision=6)
    assert formula == "Fe0.999999Co0.000001"


def test_parse_formula_rejects_unmatched_closing_bracket():
    with pytest.raises(ValueError, match="Unexpected closing bracket"):
        parse_formula("Fe)O")


@pytest.mark.parametrize("method", ["clr", "alr", "ilr"])
def test_log_ratio_round_trip(method):
    values = np.asarray([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    transform = SimplexTransform(method=method, pseudocount=1e-12)
    recovered = transform.inverse_transform(transform.transform(values), n_components=3)
    np.testing.assert_allclose(recovered, values, atol=1e-10)


def test_descriptor_calculator_supports_custom_properties():
    calculator = CompositionDescriptorCalculator(
        properties=["electronegativity"],
        statistics=["mean", "range"],
        element_properties={"electronegativity": {"Fe": 1.83, "Ni": 1.91}},
        include_num_elements=False,
        include_mixing_entropy=False,
    )
    result = calculator.transform([[0.5, 0.5]], ["Fe", "Ni"])
    np.testing.assert_allclose(result, [[1.87, 0.08]])


def test_transformer_outputs_model_ready_frame_and_inverse_formula():
    formulas = pd.Series(["Fe0.5Ni0.3Co0.2", "Fe0.4Ni0.4Co0.2"])
    transformer = CompositionTransformer(
        representation="ilr",
        include_descriptors=True,
        prefix="alloy",
    )
    transformed = transformer.fit_transform(formulas)
    assert transformed.shape[0] == 2
    assert "alloy__ilr__1" in transformed.columns
    assert "alloy__descriptor__atomic_number__mean" in transformed.columns
    recovered = transformer.inverse_transform(transformed)
    assert recovered.iloc[0] == "Fe0.5Co0.2Ni0.3"


def test_search_space_repairs_bounds_total_steps_and_sparsity():
    space = CompositionSearchSpace(
        components=["Fe", "Ni", "Co", "Cr"],
        total=1.0,
        bounds={"Fe": (0.2, 0.8), "Ni": (0.0, 0.6), "Co": (0.0, 0.4), "Cr": (0.0, 0.2)},
        steps={"Fe": 0.01, "Ni": 0.01, "Co": 0.01, "Cr": 0.01},
        min_active_components=2,
        max_active_components=3,
        required_components=["Fe"],
    )
    repaired = space.repair({"Fe": 0.51, "Ni": 0.28, "Co": 0.15, "Cr": 0.06})
    assert not space.validate(repaired)
    assert sum(value > 0 for value in repaired.values()) <= 3
    assert sum(repaired.values()) == pytest.approx(1.0)


def test_preprocessor_bridges_existing_tabular_dataframe_api():
    frame = pd.DataFrame(
        {
            "formula": ["Fe0.5Ni0.3Co0.2", "Fe0.4Ni0.4Co0.2"],
            "temperature": [900.0, 950.0],
            "property": [10.0, 12.0],
        }
    )
    config = CompositionColumnConfig(column="formula", representation="alr", reference_element="Co")
    preprocessor = CompositionTabularPreprocessor(config)
    transformed = preprocessor.fit_transform(frame)
    assert "formula" not in transformed.columns
    assert "temperature" in transformed.columns
    assert "formula__alr__Fe_over_Co" in transformed.columns
    recovered = preprocessor.inverse_candidates(transformed.drop(columns=["property"]))
    assert recovered.loc[0, "formula"] == "Fe0.5Co0.2Ni0.3"


def test_search_space_enforces_required_and_minimum_active_components():
    space = CompositionSearchSpace(
        components=["Fe", "Ni", "Co"],
        steps={"Fe": 0.01, "Ni": 0.01, "Co": 0.01},
        min_active_components=2,
        max_active_components=2,
        required_components=["Co"],
    )
    repaired = space.repair({"Fe": 1.0, "Ni": 0.0, "Co": 0.0})
    assert repaired["Co"] >= 0.01
    assert sum(value > 0 for value in repaired.values()) == 2
    assert not space.validate(repaired)


def test_inverse_transform_converts_weight_fraction_to_atomic_formula():
    transformer = CompositionTransformer(
        elements=["H", "O"],
        normalization="weight_fraction",
        representation="fractions",
        precision=4,
    )
    transformed = transformer.fit_transform(pd.Series(["H2O"]))
    assert transformer.inverse_transform(transformed).iloc[0] == "H0.6667O0.3333"


def test_inverse_transform_omits_pseudocount_zero_components():
    transformer = CompositionTransformer(
        elements=["Fe", "Co", "Ni"],
        representation="ilr",
        precision=6,
    )
    transformed = transformer.fit_transform(pd.Series(["Fe0.5Ni0.5"]))
    assert transformer.inverse_transform(transformed).iloc[0] == "Fe0.5Ni0.5"


def test_search_space_rejects_non_activatable_required_component():
    with pytest.raises(ValueError, match="cannot be active"):
        CompositionSearchSpace(
            components=["Fe", "Ni"],
            bounds={"Fe": (0.0, 1.0), "Ni": (0.0, 0.0)},
            required_components=["Ni"],
        )


def test_search_space_handles_small_positive_upper_without_step():
    space = CompositionSearchSpace(
        components=["Fe", "Ni"],
        bounds={"Fe": (0.0, 1.0), "Ni": (0.0, 5e-8)},
        required_components=["Ni"],
        tolerance=1e-8,
    )
    repaired = space.repair({"Fe": 1.0, "Ni": 0.0})
    assert repaired["Ni"] > space.tolerance
    assert not space.validate(repaired)
