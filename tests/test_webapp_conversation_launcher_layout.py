from pathlib import Path


def test_conversation_launcher_owns_compact_two_line_sizing() -> None:
    source = Path("web/src/styles/typography.css").read_text(encoding="utf-8")

    assert "button.conversation-launcher {\n  min-height: 66px;" in source
    assert ".conversation-launcher-copy strong {\n  font-size: 13px !important;\n  line-height: 1.25;" in source
    assert ".conversation-launcher-copy small {\n  font-size: 11px !important;\n  line-height: 1.3;" in source

    shared_primary = source[
        source.index(".workflow-step strong,") : source.index(".workflow-step span")
    ]
    assert ".conversation-launcher-copy strong," not in shared_primary
