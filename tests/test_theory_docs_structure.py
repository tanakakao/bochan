from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY_ROOT = REPO_ROOT / "docs" / "theory"


def _chapter_files(language: str) -> set[str]:
    return {
        path.name
        for path in (THEORY_ROOT / language).glob("*.md")
        if path.name != "README.md"
    }


def _indexed_chapters(language: str) -> set[str]:
    readme = (THEORY_ROOT / language / "README.md").read_text(encoding="utf-8")
    return set(re.findall(r"`([^`]+\.md)`", readme))


def test_theory_root_is_only_a_language_selector() -> None:
    root_readme = (THEORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "[English theory reference](en/README.md)" in root_readme
    assert "[日本語 理論リファレンス](ja/README.md)" in root_readme
    assert {path.name for path in THEORY_ROOT.glob("*.md")} == {"README.md"}


def test_theory_language_trees_are_mirrored() -> None:
    english = _chapter_files("en")
    japanese = _chapter_files("ja")

    assert english
    assert english == japanese


def test_theory_language_indexes_cover_every_chapter() -> None:
    for language in ("en", "ja"):
        assert _indexed_chapters(language) == _chapter_files(language)


def test_theory_language_indexes_link_to_each_other() -> None:
    english = (THEORY_ROOT / "en" / "README.md").read_text(encoding="utf-8")
    japanese = (THEORY_ROOT / "ja" / "README.md").read_text(encoding="utf-8")

    assert "[日本語](../ja/README.md)" in english
    assert "[English](../en/README.md)" in japanese


def test_theory_migration_scaffolding_is_removed() -> None:
    migration_workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-bilingual-theory-docs.yml"
    )

    assert not (THEORY_ROOT / "root_readme_new.md").exists()
    assert not (THEORY_ROOT / "selector.txt").exists()
    assert not migration_workflow.exists()
    assert not list((REPO_ROOT / ".github").glob("translation-trigger*.txt"))
