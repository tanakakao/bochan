from pathlib import Path


SETTINGS_SOURCE = Path("web/src/compositionExtension.ts")
MODEL_SOURCE = Path("web/src/components/CompositionModelSettings.tsx")
CANDIDATE_SOURCE = Path("web/src/components/CompositionCandidateConstraints.tsx")


def test_composition_extension_is_settings_only() -> None:
    source = SETTINGS_SOURCE.read_text(encoding="utf-8")

    assert "loadCompositionSettings" in source
    assert "MutationObserver" not in source
    assert "document.querySelector" not in source
    assert "document.createElement" not in source
    assert "innerHTML" not in source
    assert "replaceWith" not in source
    assert "installCompositionExtension" not in source


def test_composition_panels_are_rendered_by_react_components() -> None:
    model = MODEL_SOURCE.read_text(encoding="utf-8")
    candidate = CANDIDATE_SOURCE.read_text(encoding="utf-8")

    assert "composition-model-settings-react" in model
    assert "組成式のモデル変換" in model
    assert "composition-search-space-constraints-react" in candidate
    assert "composition-linear-constraints-react" in candidate

    for source in (model, candidate):
        assert "MutationObserver" not in source
        assert "document.querySelector" not in source
        assert "document.createElement" not in source
        assert "replaceWith" not in source
