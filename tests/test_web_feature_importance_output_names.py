from types import SimpleNamespace

from bochan.serving.webapp.feature_importance_outputs import (
    relabel_feature_importance_outputs,
)


def _result(*names: str) -> SimpleNamespace:
    outputs = {
        name: SimpleNamespace(output_name=name, value=index)
        for index, name in enumerate(names)
    }
    return SimpleNamespace(outputs=outputs, metadata={})


def test_relabels_native_multitask_outputs_with_target_columns() -> None:
    result = _result("output_0", "output_1")

    returned = relabel_feature_importance_outputs(
        result,
        ["strength", "density"],
    )

    assert returned is result
    assert list(result.outputs) == ["strength", "density"]
    assert result.outputs["strength"].output_name == "strength"
    assert result.outputs["density"].output_name == "density"
    assert result.outputs["strength"].value == 0
    assert result.outputs["density"].value == 1
    assert result.metadata["output_name_map"] == {
        "output_0": "strength",
        "output_1": "density",
    }
    assert result.metadata["output_names_source"] == "target_columns"


def test_output_count_mismatch_is_left_unchanged() -> None:
    result = _result("output_0")

    relabel_feature_importance_outputs(
        result,
        ["strength", "density"],
    )

    assert list(result.outputs) == ["output_0"]
    assert result.outputs["output_0"].output_name == "output_0"
    assert result.metadata == {}


def test_duplicate_target_names_are_left_unchanged() -> None:
    result = _result("output_0", "output_1")

    relabel_feature_importance_outputs(
        result,
        ["property", "property"],
    )

    assert list(result.outputs) == ["output_0", "output_1"]
    assert result.metadata == {}
