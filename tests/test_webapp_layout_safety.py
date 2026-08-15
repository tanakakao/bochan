from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "web/src/main.tsx"
SAFETY = ROOT / "web/src/styles/layout-safety.css"
PREPARE = ROOT / "web/src/pages/PreparePage.css"
TARGET_SETTINGS = ROOT / "web/src/target-settings.css"


def test_layout_safety_is_an_explicit_geometry_layer() -> None:
    main = MAIN.read_text(encoding="utf-8")
    source = SAFETY.read_text(encoding="utf-8")

    composition_import = 'import "./composition-extension.css";'
    safety_import = 'import "./styles/layout-safety.css";'
    theme_import = 'import "./red-theme.css";'

    assert safety_import in main
    assert main.index(composition_import) < main.index(safety_import) < main.index(theme_import)

    # This file is a geometry/containment owner, not another visual-theme patch.
    assert "color:" not in source
    assert "background:" not in source
    assert "box-shadow:" not in source
    assert "!important" not in source


def test_dynamic_card_content_can_shrink_or_wrap() -> None:
    source = SAFETY.read_text(encoding="utf-8")

    assert ".panel," in source
    assert ".side-card," in source
    assert ".constraint-card," in source
    assert ".conversation-action-card," in source
    assert "min-width: 0;" in source
    assert "max-width: 100%;" in source
    assert "overflow-wrap: anywhere;" in source

    context_start = source.index(".context-list strong")
    context_block = source[context_start : source.index("}", context_start)]
    assert "white-space: normal;" in context_block
    assert "overflow-wrap: anywhere;" in context_block


def test_select_cards_share_one_minimum_height() -> None:
    source = TARGET_SETTINGS.read_text(encoding="utf-8")

    assert "--select-variable-card-min-height: 56px;" in source
    variable_start = source.index(".variable-choice {")
    variable_block = source[variable_start : source.index("}", variable_start)]
    assert "min-height: var(--select-variable-card-min-height);" in variable_block
    assert "min-height: 54px;" not in variable_block


def test_select_metadata_badges_wrap_instead_of_overflowing() -> None:
    source = PREPARE.read_text(encoding="utf-8")
    safety = SAFETY.read_text(encoding="utf-8")
    selector = (
        ".feature-variable-choice.selected-with-missing\n"
        "  .variable-choice-main > small:first-of-type,"
    )
    start = source.index(selector)
    end = source.index("}", start)
    block = source[start:end]

    assert "flex-wrap: wrap;" in block
    assert "width: 100%;" in block
    assert "white-space: normal;" in block

    assert 'content: "⚠ 欠損あり";' in source
    assert 'content: "⚠ ID列候補";' in source

    # A broad descendant selector here previously made badges inherit the
    # column-name sizing and broke the entire card. Keep safety rules scoped.
    assert ".variable-choice-main span" not in safety


def test_narrow_feature_cards_stack_controls() -> None:
    source = SAFETY.read_text(encoding="utf-8")

    assert "@media (max-width: 640px)" in source
    narrow = source[source.index("@media (max-width: 640px)") :]
    assert ".feature-variable-choice" in narrow
    assert "grid-template-columns: minmax(0, 1fr);" in narrow
    assert ".feature-type-toggle," in narrow
    assert ".composition-kind-control" in narrow
    assert "width: 100%;" in narrow
