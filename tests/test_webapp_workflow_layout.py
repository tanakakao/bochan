from pathlib import Path


LAYOUT_CSS = Path("web/src/workflow-layout-extension.css")
MAIN_SOURCE = Path("web/src/main.tsx")
SETTINGS_SOURCE = Path("web/src/pages/SettingsPage.tsx")
RESULTS_SOURCE = Path("web/src/pages/ResultsPage.tsx")
RESULT_PLOTS_SOURCE = Path("web/src/InteractiveResultPlots.tsx")
RESULTS_EXTENSION_SOURCE = Path("web/src/resultsLayoutExtension.ts")


def test_model_settings_cards_follow_requested_react_order() -> None:
    source = SETTINGS_SOURCE.read_text(encoding="utf-8")

    preprocessing_index = source.index("feature-preprocessing-panel")
    composition_index = source.index("<CompositionModelSettings />")
    accuracy_index = source.index("<h3>精度評価</h3>")
    importance_index = source.index("<h3>特徴量重要度</h3>")

    assert preprocessing_index < composition_index < accuracy_index < importance_index
    assert "<FeatureMissingSettings />" in source
    preprocessing_start = source.index('className="panel feature-preprocessing-panel"')
    preprocessing_end = source.index("<CompositionModelSettings />", preprocessing_start)
    assert preprocessing_start < source.index("<FeatureMissingSettings />") < preprocessing_end


def test_workflow_pages_do_not_use_dom_reordering_extensions() -> None:
    main = MAIN_SOURCE.read_text(encoding="utf-8")
    composition_runtime = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")
    candidate_controls = Path(
        "web/src/components/CompositionCandidateConstraints.tsx"
    ).read_text(encoding="utf-8")
    model_controls = Path(
        "web/src/components/CompositionModelSettings.tsx"
    ).read_text(encoding="utf-8")
    results = RESULTS_SOURCE.read_text(encoding="utf-8")
    result_plots = RESULT_PLOTS_SOURCE.read_text(encoding="utf-8")

    assert 'from "./workflowLayoutExtension"' not in main
    assert "installWorkflowLayoutExtension" not in main
    assert 'from "./resultsLayoutExtension"' not in main
    assert "installResultsLayoutExtension" not in main
    assert not RESULTS_EXTENSION_SOURCE.exists()

    for source in (
        composition_runtime,
        candidate_controls,
        model_controls,
        results,
        result_plots,
    ):
        assert "MutationObserver" not in source
        assert "replaceWith" not in source
        assert "insertAdjacentElement" not in source

    for source in (composition_runtime, candidate_controls, model_controls):
        assert "composition-model-settings-host" not in source
        assert "composition-constraint-settings-host" not in source
        assert "composition-search-space-constraints-proxy" not in source
        assert "composition-linear-constraints-proxy" not in source


def test_results_dashboard_is_owned_by_react() -> None:
    results = RESULTS_SOURCE.read_text(encoding="utf-8")
    plots = RESULT_PLOTS_SOURCE.read_text(encoding="utf-8")
    css = LAYOUT_CSS.read_text(encoding="utf-8")

    assert 'className="results-dashboard-layout"' in results
    assert "results-candidates-panel" in results
    assert "results-accuracy-slot" in results
    assert "results-importance-slot" in results
    assert "<InteractiveResultPlots" in results
    assert "<FeatureImportancePanel" in results
    assert "results-interactive-section" in plots
    assert "results-yy-card" in plots
    assert "results-relationship-card" in plots

    assert '"candidates candidates"' in css
    assert '"yy accuracy"' in css
    assert '"importance relationship"' in css
    assert '"candidates"\n      "yy"\n      "accuracy"' in css


def test_results_to_data_navigation_cannot_remove_reparented_nodes() -> None:
    results = RESULTS_SOURCE.read_text(encoding="utf-8")
    main = MAIN_SOURCE.read_text(encoding="utf-8")

    for source in (results, main):
        assert "removeChild" not in source
        assert "append(...desired)" not in source
        assert "parent.insertBefore" not in source
    assert "resultsLayoutExtension" not in main
