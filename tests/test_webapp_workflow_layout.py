from pathlib import Path


RESULTS_LAYOUT_SOURCE = Path("web/src/resultsLayoutExtension.ts")
LAYOUT_CSS = Path("web/src/workflow-layout-extension.css")
MAIN_SOURCE = Path("web/src/main.tsx")
SETTINGS_SOURCE = Path("web/src/pages/SettingsPage.tsx")


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


def test_model_and_suggest_pages_do_not_use_layout_dom_reordering() -> None:
    main = MAIN_SOURCE.read_text(encoding="utf-8")
    composition_runtime = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")
    candidate_controls = Path(
        "web/src/components/CompositionCandidateConstraints.tsx"
    ).read_text(encoding="utf-8")
    model_controls = Path(
        "web/src/components/CompositionModelSettings.tsx"
    ).read_text(encoding="utf-8")

    assert 'from "./workflowLayoutExtension"' not in main
    assert "installWorkflowLayoutExtension" not in main
    assert 'from "./resultsLayoutExtension"' in main
    assert "installResultsLayoutExtension();" in main

    for source in (composition_runtime, candidate_controls, model_controls):
        assert "composition-model-settings-host" not in source
        assert "composition-constraint-settings-host" not in source
        assert "composition-search-space-constraints-proxy" not in source
        assert "composition-linear-constraints-proxy" not in source

    for source in (candidate_controls, model_controls):
        assert "MutationObserver" not in source
        assert "appendChild" not in source
        assert "replaceWith" not in source


def test_results_observer_is_isolated_from_model_and_suggest_pages() -> None:
    source = RESULTS_LAYOUT_SOURCE.read_text(encoding="utf-8")

    assert "RESULTS_ANCHOR_SELECTOR" in source
    assert "article.recommended-first" in source
    assert ".interactive-visualization-section" in source
    assert ".feature-importance-panel" in source
    assert ".model-primary-grid" not in source
    assert ".feature-constraint-panel" not in source
    assert "composition-model-settings-host" not in source
    assert "composition-constraint-settings-host" not in source
    assert "resultsObserver?.disconnect();" in source
    assert "observeResultsMutations();" in source


def test_results_dashboard_uses_requested_grid_areas() -> None:
    source = RESULTS_LAYOUT_SOURCE.read_text(encoding="utf-8")
    css = LAYOUT_CSS.read_text(encoding="utf-8")

    assert "results-candidates-panel" in source
    assert "results-yy-card" in source
    assert "results-accuracy-slot" in source
    assert "results-importance-slot" in source
    assert "results-relationship-card" in source
    assert '"candidates candidates"' in css
    assert '"yy accuracy"' in css
    assert '"importance relationship"' in css
    assert '"candidates"\n      "yy"\n      "accuracy"' in css
