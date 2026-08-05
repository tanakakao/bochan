from pathlib import Path


LAYOUT_SOURCE = Path("web/src/workflowLayoutExtension.ts")
LAYOUT_CSS = Path("web/src/workflow-layout-extension.css")
MAIN_SOURCE = Path("web/src/main.tsx")


def test_model_settings_cards_follow_requested_order() -> None:
    source = LAYOUT_SOURCE.read_text(encoding="utf-8")

    assert 'panelByHeading("説明変数の前処理")' in source
    assert 'panelByHeading("精度評価")' in source
    assert 'panelByHeading("特徴量重要度")' in source
    assert "preprocessingGrid.appendChild(missingPanel)" in source
    assert "placeAfter(preprocessingPanel, modelGrid)" in source
    assert "placeAfter(compositionHost, preprocessingPanel)" in source
    assert "placeAfter(accuracyPanel, accuracyAnchor)" in source
    assert "placeAfter(importancePanel, accuracyPanel)" in source
    assert "compositionHost.hidden = !compositionEnabled" in source


def test_composition_constraints_are_embedded_in_matching_feature_cards() -> None:
    source = LAYOUT_SOURCE.read_text(encoding="utf-8")
    css = LAYOUT_CSS.read_text(encoding="utf-8")

    assert 'panelByHeading("説明変数の探索範囲")' in source
    assert 'document.querySelector<HTMLElement>(\n    ".feature-constraint-panel"' in source
    assert "composition-search-space-constraints-proxy" in source
    assert "組成候補の元素制約" in source
    assert "元素数の制約" in source
    assert "composition-ratio-constraints-proxy" in source
    assert "composition-linear-constraints-proxy" in source
    assert "bindCompositionProxy(proxy)" in source
    assert ".composition-constraint-settings-source[hidden]" in css


def test_results_dashboard_uses_requested_grid_areas() -> None:
    source = LAYOUT_SOURCE.read_text(encoding="utf-8")
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


def test_workflow_layout_extension_is_installed() -> None:
    main = MAIN_SOURCE.read_text(encoding="utf-8")

    assert 'from "./workflowLayoutExtension"' in main
    assert 'import "./workflow-layout-extension.css";' in main
    assert "installWorkflowLayoutExtension();" in main
