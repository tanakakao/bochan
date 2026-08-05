from pathlib import Path


def test_model_and_suggest_composition_ui_are_react_owned() -> None:
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    model = Path("web/src/components/CompositionModelSettings.tsx").read_text(
        encoding="utf-8"
    )
    candidate = Path(
        "web/src/components/CompositionCandidateConstraints.tsx"
    ).read_text(encoding="utf-8")
    runtime = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")
    results = Path("web/src/resultsLayoutExtension.ts").read_text(encoding="utf-8")

    assert "installCompositionRuntime" in main
    assert "installCompositionExtension" not in main
    assert "installResultsLayoutExtension" in main
    assert "installWorkflowLayoutExtension" not in main

    for source in (model, candidate):
        assert "MutationObserver" not in source
        assert "cloneNode" not in source
        assert "replaceWith" not in source
        assert "insertAdjacentElement" not in source

    assert ".feature-variable-choice" in runtime
    assert ".model-primary-grid" not in runtime
    assert ".feature-constraint-panel" not in runtime
    assert "composition-model-settings-host" not in runtime
    assert "composition-constraint-settings-host" not in runtime

    assert ".model-primary-grid" not in results
    assert ".feature-constraint-panel" not in results
    assert "composition-model-settings-host" not in results
    assert "composition-constraint-settings-host" not in results
