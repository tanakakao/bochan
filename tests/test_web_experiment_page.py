"""Static integration checks for the Web experiment-result workflow."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_result_page_is_connected() -> None:
    app = (ROOT / "web/src/App.tsx").read_text(encoding="utf-8")
    results = (ROOT / "web/src/pages/ResultsPage.tsx").read_text(encoding="utf-8")
    experiment = (ROOT / "web/src/pages/ExperimentPage.tsx").read_text(encoding="utf-8")
    history = (ROOT / "web/src/components/ExperimentHistoryPanel.tsx").read_text(encoding="utf-8")
    data_page = (ROOT / "web/src/pages/DataPage.tsx").read_text(encoding="utf-8")
    data_helpers = (ROOT / "web/src/experimentData.ts").read_text(encoding="utf-8")
    history_api = (ROOT / "web/src/experimentHistory.ts").read_text(encoding="utf-8")
    project_api = (ROOT / "web/src/experimentProject.ts").read_text(encoding="utf-8")
    main = (ROOT / "web/src/main.tsx").read_text(encoding="utf-8")

    assert 'window.location.hash = "experiment"' in results
    assert 'import ExperimentPage from "./pages/ExperimentPage"' in app
    assert "strong>Experiment</strong><small>実験結果追加" in app
    assert "appendExperimentRows" in experiment
    assert "appendExperimentFile" in experiment
    assert "recordExperimentCycle" in experiment
    assert "ExperimentHistoryPanel" in experiment
    assert "実験条件" in experiment
    assert "実験結果" in experiment
    assert "COMPLETE_DATASET_LIMIT" in data_helpers
    assert "stale_after_data_append" in experiment
    assert "サイクル内ベスト" in history
    assert "fetchExperimentHistory" in history
    assert "downloadExperimentProject" in history
    assert "履歴込みプロジェクトを保存" in history
    assert ".bochan-project.zip" in data_page
    assert "履歴付きプロジェクトを開く" in data_page
    assert 'request<ExperimentHistoryResponse>' in history_api
    assert '"/experiment-projects/export"' in project_api
    assert "model_included" not in project_api
    assert 'import "./experiment-results.css"' in main
    assert 'import "./experiment-history.css"' in main


def test_experiment_import_preserves_existing_columns() -> None:
    data_helpers = (ROOT / "web/src/experimentData.ts").read_text(encoding="utf-8")

    assert "normalizedRows" in data_helpers
    assert "未登録列があります" in data_helpers
    assert "missingRequired" in data_helpers
    assert "currentComplete.preview" in data_helpers
    assert "rows" in data_helpers
