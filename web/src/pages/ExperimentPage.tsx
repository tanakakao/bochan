import { useMemo, useState } from "react";
import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  appendExperimentFile,
  appendExperimentRows,
  downloadExperimentTemplate
} from "../experimentData";
import type { ColumnProfile, DatasetResponse, RegressionResult } from "../types";

interface DraftRow {
  id: string;
  source: string;
  selected: boolean;
  values: Record<string, string>;
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

function markPreviousResultStale(result: RegressionResult, appendedRows: number): void {
  delete result.visualization_run_id;
  result.metadata = {
    ...result.metadata,
    stale_after_data_append: true,
    appended_experiment_rows: appendedRows
  };
}

/** Records experimental outcomes against editable proposed conditions or an imported result file. */
export default function ExperimentPage() {
  const {
    dataset,
    result,
    variables,
    setError,
    setStep
  } = useWorkbench();
  const [sourceResult] = useState<RegressionResult | null>(() => result);
  const [rows, setRows] = useState<DraftRow[]>(() => initialDraftRows(result));
  const [importFile, setImportFile] = useState<File | null>(null);
  const [saving, setSaving] = useState(false);
  const [completedMessage, setCompletedMessage] = useState<string | null>(null);

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

  function patchRow(id: string, column: string, value: string) {
    setRows((current) => current.map((row) => (
      row.id === id ? { ...row, values: { ...row.values, [column]: value } } : row
    )));
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

  function applyUpdatedDataset(updated: DatasetResponse, appendedRows: number) {
    Object.assign(activeDataset, updated);
    markPreviousResultStale(activeResult, appendedRows);
    setCompletedMessage(
      `${appendedRows}件の実験データを追加しました。現在の変数・モデル設定を保持したまま再学習できます。`
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
                    <td key={column}>
                      <input
                        className="experiment-cell-input"
                        type={columnsByName[column]?.kind === "numeric" ? "number" : "text"}
                        step={variables[column]?.step ?? "any"}
                        value={row.values[column] ?? ""}
                        onChange={(event) => patchRow(row.id, column, event.target.value)}
                        disabled={saving || Boolean(completedMessage)}
                        aria-label={`${row.source} ${column} 実験条件`}
                      />
                    </td>
                  ))}
                  {targetColumns.map((column) => (
                    <td key={column}>
                      <input
                        className="experiment-cell-input result-input"
                        type={columnsByName[column]?.kind === "numeric" ? "number" : "text"}
                        step="any"
                        value={row.values[column] ?? ""}
                        onChange={(event) => patchRow(row.id, column, event.target.value)}
                        disabled={saving || Boolean(completedMessage)}
                        placeholder="実測値"
                        aria-label={`${row.source} ${column} 実験結果`}
                      />
                    </td>
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
    </>
  );
}
