from pathlib import Path


def test_composition_settings_are_split_between_model_and_constraint_pages() -> None:
    source = Path("web/src/compositionExtension.ts").read_text(encoding="utf-8")
    feature_constraints = Path("web/src/components/FeatureConstraints.tsx").read_text(
        encoding="utf-8"
    )
    feature_missing = Path("web/src/components/FeatureMissingSettings.tsx").read_text(
        encoding="utf-8"
    )

    model_start = source.index("function modelPanelHtml")
    constraint_start = source.index("function constraintPanelHtml")
    update_start = source.index("function updateElements")
    model_source = source[model_start:constraint_start]
    constraint_source = source[constraint_start:update_start]

    assert 'document.querySelector<HTMLElement>(".model-primary-grid")' in source
    assert 'host.className = "composition-model-settings-host"' in source
    assert 'modelGrid.insertAdjacentElement("afterend", host)' in source
    assert 'document.querySelector<HTMLElement>(".feature-constraint-panel")' in source
    assert 'host.className = "composition-constraint-settings-host"' in source
    assert 'className="panel feature-constraint-panel"' in feature_constraints
    assert 'className="panel feature-missing-panel"' in feature_missing
    assert 'feature-constraint-panel' not in feature_missing
    assert "組成式のモデル変換" in model_source
    assert "変換方法" in model_source
    assert "組成基準" in model_source
    assert "composition-min-components" not in model_source
    assert "composition-add-constraint" not in model_source
    assert "組成候補の元素制約" in constraint_source
    assert "composition-min-components" in constraint_source
    assert "composition-max-components" in constraint_source
    assert "renderElementTable(settings)" in constraint_source
    assert "composition-add-constraint" in constraint_source


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
