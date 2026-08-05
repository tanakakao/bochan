from pathlib import Path


def test_model_to_suggest_navigation_has_no_legacy_page_hosts() -> None:
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    settings = Path("web/src/pages/SettingsPage.tsx").read_text(encoding="utf-8")
    runtime = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")

    assert "<CompositionModelSettings />" in settings
    assert "<CompositionSearchSpaceConstraints />" in Path(
        "web/src/components/SearchVariableSettings.tsx"
    ).read_text(encoding="utf-8")
    assert "<CompositionLinearConstraints />" in Path(
        "web/src/components/FeatureConstraints.tsx"
    ).read_text(encoding="utf-8")

    assert "installCompositionExtension" not in main
    assert "installWorkflowLayoutExtension" not in main
    assert "composition-model-settings-host" not in runtime
    assert "composition-constraint-settings-host" not in runtime
    assert "model-primary-grid" not in runtime
    assert "feature-constraint-panel" not in runtime
