from __future__ import annotations

from pathlib import Path


def test_muted_red_theme_is_loaded_last() -> None:
    main_source = Path("web/src/main.tsx").read_text(encoding="utf-8")

    theme_import = 'import "./red-theme.css";'
    composition_import = 'import "./composition-extension.css";'
    assert theme_import in main_source
    assert main_source.index(theme_import) > main_source.index(composition_import)


def test_muted_red_theme_defines_red_selection_and_orange_categories() -> None:
    source = Path("web/src/red-theme.css").read_text(encoding="utf-8")

    assert "--bg: #fcfbfb" in source
    assert "--text: #302929" in source
    assert "--primary: #b94f57" in source
    assert "--primary-soft-strong: #f5d9dc" in source
    assert "--category: #c36a2d" in source
    assert "--category-soft: #fff0e3" in source
    assert ".variable-choice.selected:not(.selected-categorical)" in source
    assert ".feature-variable-choice.selected-categorical" in source
    assert (
        'button.composition-kind-option[data-composition-kind="composition"].active'
        in source
    )


def test_select_page_explains_the_red_and_orange_states() -> None:
    source = Path("web/src/pages/PreparePage.tsx").read_text(encoding="utf-8")

    assert "淡い赤は数値、オレンジはカテゴリ扱いです。" in source
    assert "青は数値、紫はカテゴリ扱いです。" not in source
