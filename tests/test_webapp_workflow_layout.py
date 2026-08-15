from pathlib import Path


LAYOUT_CSS = Path("web/src/result-interactions.css")
MAIN_SOURCE = Path("web/src/main.tsx")
SETTINGS_SOURCE = Path("web/src/pages/SettingsPage.tsx")
RESULTS_SOURCE = Path("web/src/pages/ResultsPage.tsx")
RESULT_PLOTS_SOURCE = Path("web/src/InteractiveResultPlots.tsx")
RESULTS_EXTENSION_SOURCE = Path("web/src/resultsLayoutExtension.ts")


OLD_PATCH_STYLES = (
    "web/src/constraint-settings.css",
    "web/src/constraint-selection.css",
    "web/src/data-dropzone.css",
    "web/src/workflow-separation.css",
    "web/src/ui-adjustments.css",
    "web/src/readability.css",
    "web/src/ux-simplification.css",
    "web/src/ux-enhancements.css",
    "web/src/ux-corrections.css",
    "web/src/workflow-layout-extension.css",
)


def test_model_settings_follow_problem_then_primary_then_details_order() -> None:
    source = SETTINGS_SOURCE.read_text(encoding="utf-8")

    target_task_index = source.index("<TargetModelSettings")
    model_card_index = source.index('className="panel model-workbench-card"')
    model_selection_index = source.index("model-selection-column", model_card_index)
    basic_settings_index = source.index("model-basic-settings", model_card_index)
    details_index = source.index('className="model-card-details model-output-details"')
    training_index = source.index("<h4>学習</h4>", details_index)
    robustness_index = source.index("<h4>頑健化</h4>", details_index)
    composition_index = source.index("<CompositionModelSettings />", details_index)
    missing_index = source.index("<h4>欠損値</h4>", details_index)
    noise_index = source.index("<h4>観測ノイズ</h4>", details_index)
    accuracy_index = source.index("<h3>精度評価</h3>", details_index)
    importance_index = source.index("<h3>特徴量重要度</h3>", details_index)

    assert target_task_index < model_card_index
    assert model_card_index < model_selection_index < details_index
    assert model_card_index < basic_settings_index < details_index
    assert details_index < training_index < robustness_index
    assert robustness_index < composition_index < missing_index < noise_index
    assert noise_index < accuracy_index < importance_index

    normalize_index = source.index("checked={normalize}", basic_settings_index)
    perturbation_index = source.index("checked={inputPerturbation}", basic_settings_index)
    missing_strategy_index = source.index("<FeatureMissingStrategySettings", basic_settings_index)
    cv_toggle_index = source.index("checked={crossValidation.enabled}", model_selection_index)
    importance_toggle_index = source.index("checked={featureImportance.enabled}", model_selection_index)
    alpha_index = source.index("<NoiseAlphaSettings", details_index)
    missing_detail_index = source.index("<FeatureMissingImputationSettings", details_index)

    assert basic_settings_index < normalize_index < details_index
    assert basic_settings_index < perturbation_index < details_index
    assert basic_settings_index < missing_strategy_index < details_index
    assert model_selection_index < cv_toggle_index < details_index
    assert model_selection_index < importance_toggle_index < details_index
    assert details_index < missing_detail_index < alpha_index


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


def test_workbench_css_uses_canonical_ownership_files() -> None:
    main = MAIN_SOURCE.read_text(encoding="utf-8")
    target_css = Path("web/src/target-settings.css").read_text(encoding="utf-8")

    assert 'import "./styles/workflow.css";' in main
    assert 'import "./styles/typography.css";' in main
    assert 'import "./styles/workbench-design.css";' in main
    assert 'import "./advanced-settings.css";' in main
    assert 'import "./model-artifact.css";' in main
    assert 'import "./result-interactions.css";' in main

    for path in OLD_PATCH_STYLES:
        assert not Path(path).exists()
        assert Path(path).name not in main

    assert "body {" not in target_css
    assert ".brand-wordmark" not in target_css


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
