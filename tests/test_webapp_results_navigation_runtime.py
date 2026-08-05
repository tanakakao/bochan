from pathlib import Path


def test_results_runtime_has_no_external_reparenting_extension() -> None:
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    results = Path("web/src/pages/ResultsPage.tsx").read_text(encoding="utf-8")

    assert "resultsLayoutExtension" not in main
    assert "installResultsLayoutExtension" not in main
    assert "results-dashboard-layout" in results
    assert "MutationObserver" not in results
