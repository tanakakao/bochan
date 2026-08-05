from pathlib import Path


def test_workflow_composition_ui_is_react_owned() -> None:
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    prepare = Path("web/src/pages/PreparePage.tsx").read_text(encoding="utf-8")
    kind = Path("web/src/components/CompositionKindControl.tsx").read_text(
        encoding="utf-8"
    )
    model = Path("web/src/components/CompositionModelSettings.tsx").read_text(
        encoding="utf-8"
    )
    candidate = Path(
        "web/src/components/CompositionCandidateConstraints.tsx"
    ).read_text(encoding="utf-8")
    runtime = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")
    results = Path("web/src/pages/ResultsPage.tsx").read_text(encoding="utf-8")
    result_plots = Path("web/src/InteractiveResultPlots.tsx").read_text(
        encoding="utf-8"
    )

    assert "installCompositionRuntime" in main
    assert "installCompositionPrepareControls" not in main
    assert "installCompositionExtension" not in main
    assert "installResultsLayoutExtension" not in main
    assert "installWorkflowLayoutExtension" not in main
    assert "<CompositionKindControl" in prepare

    for source in (kind, model, candidate, runtime, results, result_plots):
        assert "MutationObserver" not in source
        assert "cloneNode" not in source
        assert "replaceWith" not in source
        assert "insertAdjacentElement" not in source

    assert ".feature-variable-choice" not in runtime
    assert ".model-primary-grid" not in runtime
    assert ".feature-constraint-panel" not in runtime
    assert "composition-model-settings-host" not in runtime
    assert "composition-constraint-settings-host" not in runtime

    assert 'className="results-dashboard-layout"' in results
    assert "results-candidates-panel" in results
    assert "results-accuracy-slot" in results
    assert "results-importance-slot" in results
    assert "results-interactive-section" in result_plots
    assert "results-yy-card" in result_plots
    assert "results-relationship-card" in result_plots
