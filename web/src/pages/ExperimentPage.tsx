import { useMemo, useState } from "react";
import { EmptyState, SectionHeader } from "../components/Common";
import ExperimentHistoryPanel from "../components/ExperimentHistoryPanel";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  appendExperimentFile,
  appendExperimentRows,
  downloadExperimentTemplate
} from "../experimentData";
import { recordExperimentCycle } from "../experimentHistory";
import type {
  AcquisitionFamily,
  ColumnProfile,
  DatasetResponse,
  RegressionResult,
  TargetSetting
} from "../types";

interface DraftRow {
  id: string;
  source: string;
  selected: boolean;
  values: Record<string, string>;
}

interface SourceConfiguration {
  modelType: string;
  acquisitionFamily: AcquisitionFamily;
  acquisition: string;
  beta: number;
  q: number;
  numRestarts: number;
  rawSamples: number;
  targetSettings: TargetSetting[];
}

function textValue(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}

function closeAuxiliaryPage(): void {
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

function resultTargetColumns(result: RegressionResult): string[] {
  if (result.target_columns?.length) return result.target_columns;
  return result.target_column ? [result.target_column] : [];
}

function initialDraftRows(result: RegressionResult | null): DraftRow[] {
  if (!result) return [];
  const targetColumns = resultTargetColumns(result);
  return [...result.candidates]
    .sort((left, right) => left.rank - right.rank)
    .map((candidate) => ({
      id: `candidate-${candidate.rank}`,
      source: `候補 ${candidate.rank}`,
      selected: true,
      values: {
        ...Object.fromEntries(
          result.feature_columns.map((column) => [column, textValue(candidate.values[column])])
        ),
        ...Object.fromEntries(targetColumns.map((column) => [column, ""]))
      }
    }));
}

function coerceValue(column: ColumnProfile | undefined, value: string): unknown {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (column?.kind !== "numeric") return trimmed;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${column.name}には数値を入力してください。`);
  }
  return parsed;
}

function uniqueCategoryValues(values: unknown[]): Array<string | number> {
  const unique = new Map<string, string | number>();
  for (const value of values) {
    if (typeof value !== "string" && typeof value !== "number") continue;
    const key = `${typeof value}:${String(value)}`;
    if (!unique.has(key)) unique.set(key, value);
  }
  return [...unique.values()];
}

function markPreviousResultStale(result: RegressionResult, appendedRows: number): void {
  result.metadata = {
    ...result.metadata,
    stale_after_data_append: true,
    appended_experiment_rows: appendedRows,
    visualization_uses_latest_saved_model: Boolean(result.visualization_run_id)
  };
}

function bestObservedByTarget(
  result: RegressionResult,
  targetColumns: string[]
): Record<string, number | null> {
  const bestObserved = result.best_observed;
  if (typeof bestObserved === "number") {
    return Object.fromEntries(targetColumns.map((target, index) => [
      target,
      index === 0 && Number.isFinite(bestObserved) ? bestObserved : null
    ]));
  }
  return Object.fromEntries(targetColumns.map((target) => {
    const value = Number(bestObserved[target]);
    return [target, Number.isFinite(value) ? value : null];
  }));
}

function metadataRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

/** Records experimental outcomes against editable proposed conditions or an imported result file. */
export default function ExperimentPage() {
  const {
    dataset,
    result,
    variables,
    modelType,
    acquisitionFamily,
    acquisition,
    beta,
    q,
    numRestarts,
    rawSamples,
    selectedTargetSettings,
    setError,
    setStep
  } = useWorkbench();
  const [sourceResult] = useState<RegressionResult | null>(() => result);
  const [sourceConfiguration] = useState<SourceConfiguration>(() => ({
    modelType,
    acquisitionFamily,
    acquisition,
    beta,
    q,
    numRestarts,
    rawSamples,
    targetSettings: selectedTargetSettings
  }));
  const [rows, setRows] = useState<DraftRow[]>(() => initialDraftRows(result));
  const [importFile, setImportFile] = useState<File | null>(null);
  const [saving, setSaving] = useState(false);
  const [completedMessage, setCompletedMessage] = useState<string | null>(null);
  const [historyVersion, setHistoryVersion] = useState(0);

  const featureColumns = sourceResult?.feature_columns ?? [];
  const targetColumns = sourceResult ? resultTargetColumns(sourceResult) : [];
  const columnsByName = useMemo(
    () => Object.fromEntries((dataset?.profile.columns ?? []).map((column) => [column.name, column])),
    [dataset]
  );
  const selectedCount = rows.filter((row) => row.selected).length;

  if (!dataset || !sourceResult) {
    return (
      <>
        <SectionHeader
          step="EXPERIMENT DATA"
          title="実験結果をデータへ追加する"
          text="候補生成後にこのページを開いてください。"
        />
        <EmptyState>追加対象のデータまたは候補がありません。</EmptyState>
      </>
    );
  }

  const activeDataset = dataset;
  const activeResult = sourceResult;
  const targetSettings = activeResult.target_settings?.length
    ? activeResult.target_settings
    : sourceConfiguration.targetSettings;

  function isCategoricalColumn(column: string): boolean {
    const targetSetting = targetSettings.find((setting) => setting.target === column);
    const internalTask = activeResult.target_metadata?.[column]?.internal_task;
    return variables[column]?.type === "categorical" ||
      columnsByName[column]?.kind === "categorical" ||
      targetSetting?.task_type === "classification" ||
      targetSetting?.task_type === "ordinal" ||
      internalTask === "binary" || internalTask === "multiclass" || internalTask === "ordinal";
  }

  function categoryValues(column: string): Array<string | number> {
    if (!isCategoricalColumn(column)) return [];
    const targetSetting = targetSettings.find((setting) => setting.target === column);
    const metadata = activeResult.target_metadata?.[column];
    return uniqueCategoryValues([
      ...(variables[column]?.categories ?? []),
      ...(metadata?.classes ?? []),
      ...(targetSetting?.class_order ?? []),
      ...(targetSetting?.target_classes ?? []),
      ...(columnsByName[column]?.values ?? []),
      ...activeDataset.preview.map((row) => row[column])
    ]);
  }

  function patchRow(id: string, column: string, value: string) {
    setRows((current) => current.map((row) => (
      row.id === id ? { ...row, values: { ...row.values, [column]: value } } : row
    )));
  }

  function experimentValueControl(
    row: DraftRow,
    column: string,
    role: "condition" | "result"
  ) {
    const categorical = isCategoricalColumn(column);
    const disabled = saving || Boolean(completedMessage);
    const label = `${row.source} ${column} ${role === "condition" ? "実験条件" : "実験結果"}`;
    const className = `experiment-cell-input${role === "result" ? " result-input" : ""}`;
    if (categorical) {
      return (
        <select
          className={className}
          value={row.values[column] ?? ""}
          onChange={(event) => patchRow(row.id, column, event.target.value)}
          disabled={disabled}
          aria-label={label}
        >
          <option value="">選択してください</option>
          {categoryValues(column).map((value) => (
            <option key={`${typeof value}:${String(value)}`} value={String(value)}>{String(value)}</option>
          ))}
        </select>
      );
    }
    return (
      <input
        className={className}
        type={columnsByName[column]?.kind === "numeric" ? "number" : "text"}
        step={role === "condition" ? variables[column]?.step ?? "any" : "any"}
        value={row.values[column] ?? ""}
        onChange={(event) => patchRow(row.id, column, event.target.value)}
        disabled={disabled}
        placeholder={role === "result" ? "実測値" : undefined}
        aria-label={label}
      />
    );
  }

  function toggleRow(id: string) {
    setRows((current) => current.map((row) => (
      row.id === id ? { ...row, selected: !row.selected } : row
    )));
  }

  function addBlankRow() {
    const index = rows.length + 1;
    setRows((current) => [
      ...current,
      {
        id: `manual-${Date.now()}-${index}`,
        source: `追加入力 ${index}`,
        selected: true,
        values: Object.fromEntries([...featureColumns, ...targetColumns].map((column) => [column, ""]))
      }
    ]);
  }

  function removeRow(id: string) {
    setRows((current) => current.filter((row) => row.id !== id));
  }

  async function registerCycle(
    updated: DatasetResponse,
    appendedRows: Record<string, unknown>[],
    appendMode: "manual" | "import"
  ) {
    const metadata = activeResult.metadata ?? {};
    const modelDetails = metadataRecord(metadata.model_details);
    await recordExperimentCycle({
      parent_dataset_id: activeDataset.dataset_id,
      dataset_id: updated.dataset_id,
      dataset_name: updated.name,
      source_run_id: activeResult.visualization_run_id,
      append_mode: appendMode,
      n_rows_before: activeDataset.profile.n_rows,
      n_rows_after: updated.profile.n_rows,
      rows: appendedRows,
      feature_columns: featureColumns,
      target_columns: targetColumns,
      target_settings: targetSettings,
      model: {
        type: activeResult.model_type || sourceConfiguration.modelType,
        n_train: activeResult.n_train,
        n_features: activeResult.n_features,
        details: modelDetails
      },
      acquisition: {
        name: String(
          modelDetails.effective_acquisition ??
          metadata.acquisition ??
          sourceConfiguration.acquisition
        ),
        family: sourceConfiguration.acquisitionFamily,
        beta: sourceConfiguration.beta
      },
      optimizer: {
        backend: String(
          modelDetails.optimizer_backend ??
          metadata.optimizer_backend ??
          metadata.optimizer ??
          "—"
        ),
        q: sourceConfiguration.q,
        num_restarts: sourceConfiguration.numRestarts,
        raw_samples: sourceConfiguration.rawSamples
      },
      best_observed_before: bestObservedByTarget(activeResult, targetColumns),
      candidate_count: activeResult.candidates.length
    });
  }

  function applyUpdatedDataset(updated: DatasetResponse, appendedRows: number) {
    Object.assign(activeDataset, updated);
    markPreviousResultStale(activeResult, appendedRows);
    setHistoryVersion((current) => current + 1);
    setCompletedMessage(
      `${appendedRows}件の実験データを追加しました。Resultsでは追加前の最新モデルによるグラフを確認でき、候補更新には再学習が必要です。`
    );
  }

  async function saveManualRows() {
    const selected = rows.filter((row) => row.selected);
    if (!selected.length) {
      setError("追加する行を1件以上選択してください。");
      return;
    }
    try {
      const payload = selected.map((row) => {
        if (!targetColumns.some((column) => row.values[column]?.trim())) {
          throw new Error(`${row.source}: 実験結果を1項目以上入力してください。`);
        }
        return Object.fromEntries(
          [...featureColumns, ...targetColumns].map((column) => [
            column,
            coerceValue(columnsByName[column], row.values[column] ?? "")
          ])
        );
      });
      setSaving(true);
      setError(null);
      const updated = await appendExperimentRows(activeDataset, payload);
      await registerCycle(updated, payload, "manual");
      applyUpdatedDataset(updated, payload.length);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSaving(false);
    }
  }

  async function importRows() {
    if (!importFile) {
      setError("CSVまたはExcelファイルを選択してください。");
      return;
    }
    try {
      setSaving(true);
      setError(null);
      const imported = await appendExperimentFile(activeDataset, importFile, targetColumns);
      await registerCycle(imported.dataset, imported.rows, "import");
      applyUpdatedDataset(imported.dataset, imported.appendedRows);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSaving(false);
    }
  }

  function goToPrepare() {
    closeAuxiliaryPage();
    setStep("prepare");
  }

  function backToResults() {
    closeAuxiliaryPage();
    setStep("results");
  }

  function downloadTemplate() {
    downloadExperimentTemplate(
      activeDataset,
      featureColumns,
      targetColumns,
      rows.map((row) => ({
        ...Object.fromEntries(featureColumns.map((column) => [column, row.values[column] ?? ""])),
        ...Object.fromEntries(targetColumns.map((column) => [column, ""]))
      }))
    );
  }

  return (
    <>
      <SectionHeader
        step="EXPERIMENT DATA"
        title="実験条件と結果をデータへ追加する"
        text="推奨条件は実験時の変更を反映できるよう編集可能です。結果を直接入力するか、CSV・Excelからまとめて追加します。"
        action={
          <>
            <button className="secondary" onClick={backToResults}>候補結果へ戻る</button>
            <button className="secondary" onClick={downloadTemplate}>入力テンプレートCSV</button>
          </>
        }
      />

      {completedMessage && (
        <div className="alert success experiment-complete">
          <div>{completedMessage}</div>
          <button onClick={goToPrepare}>再学習の準備へ</button>
        </div>
      )}

      <article className="panel experiment-entry-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">MANUAL ENTRY</span>
            <h3>提案候補に実験結果を入力</h3>
            <p>条件列は実際に使用した値へ書き換えられます。チェックした行だけを追加します。</p>
          </div>
          <div className="experiment-panel-actions">
            <span className="status-chip">{selectedCount} selected</span>
            <button className="secondary" onClick={addBlankRow} disabled={saving || Boolean(completedMessage)}>
              空の実験行を追加
            </button>
          </div>
        </div>

        <div className="table-wrap experiment-table-wrap">
          <table className="experiment-table">
            <thead>
              <tr>
                <th>追加</th>
                <th>元候補</th>
                {featureColumns.map((column) => <th key={column}>{column}<br /><small>実験条件</small></th>)}
                {targetColumns.map((column) => <th key={column}>{column}<br /><small>実験結果</small></th>)}
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className={!row.selected ? "experiment-row-disabled" : ""}>
                  <td>
                    <input
                      type="checkbox"
                      checked={row.selected}
                      onChange={() => toggleRow(row.id)}
                      aria-label={`${row.source}を追加対象にする`}
                      disabled={saving || Boolean(completedMessage)}
                    />
                  </td>
                  <td><strong>{row.source}</strong></td>
                  {featureColumns.map((column) => (
                    <td key={column}>{experimentValueControl(row, column, "condition")}</td>
                  ))}
                  {targetColumns.map((column) => (
                    <td key={column}>{experimentValueControl(row, column, "result")}</td>
                  ))}
                  <td>
                    <button
                      className="secondary compact-button"
                      onClick={() => removeRow(row.id)}
                      disabled={saving || Boolean(completedMessage)}
                    >
                      削除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="experiment-save-row">
          <p>追加後は旧モデルを再利用せず、更新データで再学習してください。</p>
          <button
            onClick={() => void saveManualRows()}
            disabled={saving || selectedCount === 0 || Boolean(completedMessage)}
          >
            {saving ? "追加中" : `${selectedCount}件をデータへ追加`}
          </button>
        </div>
      </article>

      <article className="panel compact-panel experiment-import-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">FILE IMPORT</span>
            <h3>実験結果ファイルをインポート</h3>
            <p>既存データと同じ列名を使用してください。未使用列は省略でき、目的変数列は必須です。</p>
          </div>
        </div>
        <div className="experiment-import-controls">
          <label className="experiment-file-picker">
            <span>CSV / Excel</span>
            <input
              type="file"
              accept=".csv,.xlsx,.xls,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
              onChange={(event) => setImportFile(event.target.files?.[0] ?? null)}
              disabled={saving || Boolean(completedMessage)}
            />
            <strong>{importFile?.name ?? "ファイル未選択"}</strong>
          </label>
          <button
            onClick={() => void importRows()}
            disabled={!importFile || saving || Boolean(completedMessage)}
          >
            {saving ? "インポート中" : "ファイルの行を追加"}
          </button>
        </div>
      </article>

      <ExperimentHistoryPanel datasetId={activeDataset.dataset_id} refreshKey={historyVersion} />
    </>
  );
}
