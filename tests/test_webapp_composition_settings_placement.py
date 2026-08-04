from pathlib import Path


def test_composition_settings_are_split_between_model_and_constraint_pages() -> None:
    source = Path("web/src/compositionExtension.ts").read_text(encoding="utf-8")

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
