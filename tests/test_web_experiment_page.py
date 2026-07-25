"""Static integration checks for the Web experiment-result workflow."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_result_page_is_connected() -> None:
    app = (ROOT / "web/src/App.tsx").read_text(encoding="utf-8")
    results = (ROOT / "web/src/pages/ResultsPage.tsx").read_text(encoding="utf-8")
    experiment = (ROOT / "web/src/pages/ExperimentPage.tsx").read_text(encoding="utf-8")
    history = (ROOT / "web/src/components/ExperimentHistoryPanel.tsx").read_text(encoding="utf-8")
    interactive = (ROOT / "web/src/InteractiveResultPlots.tsx").read_text(encoding="utf-8")
    context = (ROOT / "web/src/context/WorkbenchContext.tsx").read_text(encoding="utf-8")
    history_css = (ROOT / "web/src/experiment-history.css").read_text(encoding="utf-8")
    history_cycle_css = (
        ROOT / "web/src/experiment-history-cycle-plots.css"
    ).read_text(encoding="utf-8")
    data_page = (ROOT / "web/src/pages/DataPage.tsx").read_text(encoding="utf-8")
    data_helpers = (ROOT / "web/src/experimentData.ts").read_text(encoding="utf-8")
    history_api = (ROOT / "web/src/experimentHistory.ts").read_text(encoding="utf-8")
    project_api = (ROOT / "web/src/experimentProject.ts").read_text(encoding="utf-8")
    main = (ROOT / "web/src/main.tsx").read_text(encoding="utf-8")

    assert 'window.location.hash = "experiment"' in results
    assert 'import ExperimentPage from "./pages/ExperimentPage"' in app
    assert "strong>Experiment</strong><small>実験結果追加" in app
    assert "const experimentAvailable = Boolean(dataset && result);" in app
    assert 'auxiliaryPage === "experiment" && experimentAvailable' in app
    assert 'if (auxiliaryPage === "experiment" && !experimentAvailable)' in app
    assert 'setStep("data");\n      clearAuxiliaryHash();' in app
    assert 'disabled={!experimentAvailable}' in app
    assert 'STEPS.filter(([id]) => id !== "logs")' in app
    assert 'setStep("logs")' not in results

    assert "experimentValueControl" in experiment
    assert "categoryValues(column)" in experiment
    assert '<option value="">選択してください</option>' in experiment
    assert "visualization_uses_latest_saved_model" in experiment
    assert "delete result.visualization_run_id" not in experiment
    assert "stale_after_data_append" in experiment
    assert "appendExperimentRows" in experiment
    assert "appendExperimentFile" in experiment
    assert "recordExperimentCycle" in experiment
    assert "ExperimentHistoryPanel" in experiment
    assert "実験条件" in experiment
    assert "実験結果" in experiment

    assert "目的変数同士の関係" in history
    assert "selectedTargetXTask" in history
    assert '{ type: "category" as const }' in history
    assert "累積Pareto front" not in history
    assert "多目的パレート推移" not in history
    assert "cycleScatterTraces" in history
    assert "説明変数の探索推移" in history
    assert "サイクル内ベスト" in history
    assert "fetchExperimentHistory" in history
    assert "downloadExperimentProject" in history
    assert "履歴込みプロジェクトを保存" in history
    assert "最新モデルを含める" in history
    assert "過去サイクルのモデルも含める（標準OFF）" in history
    assert 'aria-label="プロジェクト保存名"' in history
    assert "defaultExperimentProjectFilename" in history
    assert "normalizeExperimentProjectFilename" in history
    assert "filename" in history
    assert "cycleSearchMethod" in history
    assert "requested_search_method" in history
    assert "effective_optimizer" in history
    assert "Optimizer backend" in history

    assert '"target_relation"' in interactive
    assert "目的変数同士" in interactive
    assert "Pareto図" not in interactive
    assert "result.visualizations.find" in interactive
    assert "!result?.metadata?.stale_after_data_append" in context
    assert "グラフは引き続き確認できます" in results

    assert "COMPLETE_DATASET_LIMIT" in data_helpers
    assert ".history-export-filename" in history_css
    assert "history-axis-controls" in history_cycle_css
    assert 'accept=".bochan-project.zip,.zip,application/zip"' in data_page
    assert 'endsWith(".bochan-project.zip")' not in data_page
    assert "内部のプロジェクト情報を検証" in data_page
    assert "履歴付きプロジェクトを開く" in data_page
    assert "含まれるモデルを信頼" in data_page
    assert 'request<ExperimentHistoryResponse>' in history_api
    assert '"/experiment-projects/export"' in project_api
    assert "include_latest_model: options.includeLatestModel ?? true" in project_api
    assert "include_past_models: options.includePastModels ?? false" in project_api
    assert "filename?: string" in project_api
    assert "normalizeExperimentProjectFilename" in project_api
    assert "anchor.download = filename" in project_api
    assert 'import "./experiment-results.css"' in main
    assert 'import "./experiment-history.css"' in main


def test_experiment_import_preserves_existing_columns() -> None:
    data_helpers = (ROOT / "web/src/experimentData.ts").read_text(encoding="utf-8")

    assert "normalizedRows" in data_helpers
    assert "未登録列があります" in data_helpers
    assert "missingRequired" in data_helpers
    assert "currentComplete.preview" in data_helpers
    assert "rows" in data_helpers
