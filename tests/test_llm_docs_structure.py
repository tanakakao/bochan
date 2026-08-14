from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLM_DOCS = ROOT / "docs" / "llm"


OLD_ROOT_GUIDES = {
    "README_LLM_CANDIDATE_EXPLANATION.md",
    "README_LLM_CANDIDATE_OVERALL_EXPLANATION.md",
    "README_LLM_HYBRID_CONSTRAINTS.md",
}

EXPECTED_GUIDES = {
    "candidate_explanation.md",
    "hybrid_constraints.md",
}


def test_llm_guides_live_under_docs_llm() -> None:
    assert LLM_DOCS.is_dir()
    for filename in EXPECTED_GUIDES:
        assert (LLM_DOCS / filename).is_file(), filename
    assert not (LLM_DOCS / "candidate_overall_explanation.md").exists()


def test_legacy_root_llm_guides_are_absent() -> None:
    for filename in OLD_ROOT_GUIDES:
        assert not (ROOT / filename).exists(), filename


def test_candidate_overall_explanation_is_integrated() -> None:
    candidate_guide = (LLM_DOCS / "candidate_explanation.md").read_text(encoding="utf-8")
    assert "overall_interpretation" in candidate_guide
    assert "## 13. 候補群・候補ごとの総合説明" in candidate_guide


def test_llm_docs_index_links_all_guides() -> None:
    index = (LLM_DOCS / "README.md").read_text(encoding="utf-8")
    for filename in EXPECTED_GUIDES:
        assert f"({filename})" in index, filename
    assert "candidate_overall_explanation.md" not in index
    assert "../../README_LLM.md" in index
    assert "../llm_selected_acquisition.md" in index


def test_llm_smoke_tracks_docs_directory() -> None:
    workflow = (ROOT / ".github" / "workflows" / "llm-client-smoke.yml").read_text(
        encoding="utf-8"
    )
    assert '"docs/llm/**"' in workflow
    assert '"tests/test_llm_docs_structure.py"' in workflow
    for filename in OLD_ROOT_GUIDES:
        assert filename not in workflow
