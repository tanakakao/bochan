from __future__ import annotations

from pathlib import Path


def test_muted_red_theme_is_loaded_last() -> None:
    main_source = Path("web/src/main.tsx").read_text(encoding="utf-8")

    theme_import = 'import "./red-theme.css";'
    composition_import = 'import "./composition-extension.css";'
    assert theme_import in main_source
    assert main_source.index(theme_import) > main_source.index(composition_import)


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


def test_composition_selector_observer_is_narrow_and_does_not_replace_global_observer() -> None:
    source = Path("web/src/compositionPrepareControls.ts").read_text(encoding="utf-8")
    css = Path("web/src/composition-extension.css").read_text(encoding="utf-8")

    assert "withCompositionMutationGuard" not in source
    assert 'Object.defineProperty(window, "MutationObserver"' not in source
    assert "mutationAddsCompositionControl" not in source
    assert "addedCompositionControls(record)" in source
    assert "controls.forEach(upgradeControl)" in source
    assert "control.replaceWith" not in source
    assert "control.appendChild(group)" in source
    assert "checkbox?.checked === true" in source
    assert "control.hidden = !visible" in source
    assert "turnOffCompositionWhenCategoryIsDisabled" in source
    assert "observer.observe(document.documentElement, { subtree: true, childList: true })" in source
    assert ".composition-kind-control[hidden]" in css
    assert "display: none !important" in css


def test_select_page_explains_the_red_and_orange_states() -> None:
    source = Path("web/src/pages/PreparePage.tsx").read_text(encoding="utf-8")

    assert "淡い赤は数値、オレンジはカテゴリ扱いです。" in source
    assert "青は数値、紫はカテゴリ扱いです。" not in source
