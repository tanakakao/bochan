from pathlib import Path


def test_composition_transport_is_owned_by_typed_api_request_building() -> None:
    api = Path("web/src/api.ts").read_text(encoding="utf-8")
    extension = Path("web/src/compositionExtension.ts").read_text(encoding="utf-8")

    assert "modelKwargs.web_composition = compositionSettingsToBackend" in api
    assert "const searchSpace = input.searchSpace.map" in api
    assert "markFormulaLikeColumns(dataset)" in api
    assert "formulaLikeColumn" in extension
    assert "compositionSettingsToBackend" in extension
    assert not Path("web/src/compositionRuntime.ts").exists()
    assert "window.fetch =" not in api
    assert "window.fetch =" not in extension

    for component_path in (
        "web/src/components/CompositionKindControl.tsx",
        "web/src/components/CompositionModelSettings.tsx",
        "web/src/components/CompositionCandidateConstraints.tsx",
    ):
        component = Path(component_path).read_text(encoding="utf-8")
        assert "saveCompositionSettings" in component
        assert "bochan-web-composition-settings" not in component
