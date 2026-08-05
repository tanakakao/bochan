from pathlib import Path


def test_results_to_data_navigation_uses_react_owned_parentage() -> None:
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    results = Path("web/src/pages/ResultsPage.tsx").read_text(encoding="utf-8")
    plots = Path("web/src/InteractiveResultPlots.tsx").read_text(encoding="utf-8")

    assert "resultsLayoutExtension" not in main
    assert "installResultsLayoutExtension" not in main
    assert 'className="results-dashboard-layout"' in results
    assert "results-candidates-panel" in results
    assert "results-accuracy-slot" in results
    assert "results-importance-slot" in results
    assert "results-interactive-section" in plots
    assert "results-yy-card" in plots
    assert "results-relationship-card" in plots

    for source in (results, plots):
        assert "MutationObserver" not in source
        assert "removeChild" not in source
        assert "insertBefore" not in source
        assert "append(...desired)" not in source
