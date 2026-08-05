from pathlib import Path


def test_composition_settings_are_owned_by_react_pages() -> None:
    model_controls = Path(
        "web/src/components/CompositionModelSettings.tsx"
    ).read_text(encoding="utf-8")
    candidate_controls = Path(
        "web/src/components/CompositionCandidateConstraints.tsx"
    ).read_text(encoding="utf-8")
    settings_page = Path("web/src/pages/SettingsPage.tsx").read_text(encoding="utf-8")
    search_variables = Path("web/src/components/SearchVariableSettings.tsx").read_text(
        encoding="utf-8"
    )
    feature_constraints = Path("web/src/components/FeatureConstraints.tsx").read_text(
        encoding="utf-8"
    )
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    runtime = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")

    assert "CompositionModelSettings" in settings_page
    assert "CompositionSearchSpaceConstraints" in search_variables
    assert "CompositionLinearConstraints" in feature_constraints
    assert 'className="panel candidate-feature-constraint-panel"' in feature_constraints
    assert 'className="panel feature-constraint-panel"' not in feature_constraints

    assert "組成式のモデル変換" in model_controls
    assert "変換方法" in model_controls
    assert "組成基準" in model_controls
    assert "候補元素" in model_controls
    assert "composition-model-settings-react" in model_controls

    assert "組成候補の元素制約" in candidate_controls
    assert "元素数の制約" in candidate_controls
    assert "元素ごとの比率制約" in candidate_controls
    assert "元素間の線形制約" in candidate_controls

    for source in (model_controls, candidate_controls):
        assert "MutationObserver" not in source
        assert "cloneNode" not in source
        assert "replaceWith" not in source
        assert "insertAdjacentElement" not in source

    assert 'from "./compositionRuntime"' in main
    assert "installCompositionRuntime" in main
    assert 'from "./compositionExtension"' not in main
    assert "installCompositionExtension" not in main
    assert "synchronizeModelPanel" not in runtime
    assert "synchronizeConstraintPanel" not in runtime
    assert "composition-model-settings-host" not in runtime
    assert "composition-constraint-settings-host" not in runtime


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
