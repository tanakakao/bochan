from __future__ import annotations

from pathlib import Path


def test_muted_red_theme_is_loaded_after_typography_and_domain_styles() -> None:
    main_source = Path("web/src/main.tsx").read_text(encoding="utf-8")

    typography_import = 'import "./styles/typography.css";'
    composition_import = 'import "./composition-extension.css";'
    theme_import = 'import "./red-theme.css";'
    assert typography_import in main_source
    assert composition_import in main_source
    assert theme_import in main_source
    assert (
        main_source.index(typography_import)
        < main_source.index(composition_import)
        < main_source.index(theme_import)
    )


def test_theme_defines_red_selection_orange_categories_and_yellow_compositions() -> None:
    source = Path("web/src/red-theme.css").read_text(encoding="utf-8")

    assert "--bg: #fcfbfb" in source
    assert "--text: #302929" in source
    assert "--primary: #b94f57" in source
    assert "--primary-soft-strong: #f5d9dc" in source
    assert "--category: #c36a2d" in source
    assert "--category-soft: #fff0e3" in source
    assert "--composition: #a77d00" in source
    assert "--composition-soft: #fff9d9" in source
    assert ".variable-choice.selected:not(.selected-categorical)" in source
    assert ".feature-variable-choice.selected-categorical" in source
    assert ".feature-variable-choice.selected-composition" in source
    assert (
        'button.composition-kind-option[data-composition-kind="normal"].active'
        in source
    )
    assert (
        'button.composition-kind-option[data-composition-kind="composition"].active'
        in source
    )
    assert ".feature-type-toggle input:checked + span::after" in source
    assert "background: var(--category)" in source
    assert "border-color: var(--category)" in source
    assert ".feature-type-toggle input" in source
    assert "accent-color: var(--category)" in source
    assert ".feature-missing-panel" in source


def test_composition_selector_is_react_owned_without_dom_observer() -> None:
    source = Path("web/src/components/CompositionKindControl.tsx").read_text(
        encoding="utf-8"
    )
    css = Path("web/src/composition-extension.css").read_text(encoding="utf-8")

    assert "composition-kind-control-segmented" in source
    assert "composition-kind-option" in source
    assert "selectComposition" in source
    assert "selectNormal" in source
    assert "MutationObserver" not in source
    assert "document.querySelector" not in source
    assert "document.createElement" not in source
    assert "appendChild" not in source
    assert ".composition-kind-control" in css


def test_select_page_explains_the_red_and_orange_states() -> None:
    source = Path("web/src/pages/PreparePage.tsx").read_text(encoding="utf-8")

    assert "淡い赤は数値、オレンジはカテゴリ扱いです。" in source
    assert "青は数値、紫はカテゴリ扱いです。" not in source
