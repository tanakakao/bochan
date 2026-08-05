from pathlib import Path


def test_composition_settings_use_react_owned_candidate_controls() -> None:
    extension = Path("web/src/compositionExtension.ts").read_text(encoding="utf-8")
    candidate_controls = Path(
        "web/src/components/CompositionCandidateConstraints.tsx"
    ).read_text(encoding="utf-8")
    search_variables = Path("web/src/components/SearchVariableSettings.tsx").read_text(
        encoding="utf-8"
    )
    feature_constraints = Path("web/src/components/FeatureConstraints.tsx").read_text(
        encoding="utf-8"
    )
    feature_missing = Path("web/src/components/FeatureMissingSettings.tsx").read_text(
        encoding="utf-8"
    )

    model_start = extension.index("function modelPanelHtml")
    constraint_start = extension.index("function constraintPanelHtml")
    model_source = extension[model_start:constraint_start]

    assert 'document.querySelector<HTMLElement>(".model-primary-grid")' in extension
    assert 'host.className = "composition-model-settings-host"' in extension
    assert 'modelGrid.insertAdjacentElement("afterend", host)' in extension
    assert "組成式のモデル変換" in model_source
    assert "変換方法" in model_source
    assert "組成基準" in model_source
    assert "composition-min-components" not in model_source
    assert "composition-add-constraint" not in model_source

    assert "CompositionSearchSpaceConstraints" in search_variables
    assert "CompositionLinearConstraints" in feature_constraints
    assert 'className="panel candidate-feature-constraint-panel"' in feature_constraints
    assert 'className="panel feature-constraint-panel"' not in feature_constraints
    assert 'className="panel feature-missing-panel"' in feature_missing
    assert "candidate-feature-constraint-panel" not in feature_missing

    assert "function useCompositionSettings" in candidate_controls
    assert "bochan-web-composition-settings" in candidate_controls
    assert "bochan-composition-settings-change" in candidate_controls
    assert "組成候補の元素制約" in candidate_controls
    assert "元素数の制約" in candidate_controls
    assert "元素ごとの比率制約" in candidate_controls
    assert "元素間の線形制約" in candidate_controls
    assert "MutationObserver" not in candidate_controls
    assert "cloneNode" not in candidate_controls
    assert "replaceWith" not in candidate_controls


def test_search_and_composition_tables_share_compact_aligned_columns() -> None:
    css = Path("web/src/composition-extension.css").read_text(encoding="utf-8")

    assert ".table-wrap:has(> .search-variable-table)" in css
    assert ".table-wrap:has(> .composition-element-table)" in css
    assert "width: fit-content;" in css
    assert ".search-variable-table {\n  width: 1020px;" in css
    assert ".composition-element-table {\n  width: 720px;" in css
    assert ".search-variable-table th:nth-child(1),\n.composition-element-table th:nth-child(1)" in css
    assert ".search-variable-table th:nth-child(3)," in css
    assert ".composition-element-table th:nth-child(2)," in css
    assert ".search-variable-table th:nth-child(6),\n.composition-element-table th:nth-child(5)" in css
