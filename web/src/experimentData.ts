import { uploadDataset } from "./api";
import type { DatasetResponse } from "./types";

const RAW_API_BASE = String(import.meta.env.VITE_API_BASE ?? "/api/v1").trim();
const API_BASE = (RAW_API_BASE || "/api/v1").replace(/\/+$/, "");
const COMPLETE_DATASET_LIMIT = 2_147_483_647;

async function responsePayload(response: Response): Promise<any> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function fetchDataset(datasetId: string): Promise<DatasetResponse> {
  const url = `${API_BASE}/datasets/${encodeURIComponent(datasetId)}?limit=${COMPLETE_DATASET_LIMIT}`;
  const response = await fetch(url);
  const payload = await responsePayload(response);
  if (!response.ok) {
    const detail = payload?.detail ?? payload ?? `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload as DatasetResponse;
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function normalizedRows(
  dataset: DatasetResponse,
  rows: Record<string, unknown>[]
): Record<string, unknown>[] {
  const columns = dataset.profile.columns.map((column) => column.name);
  const columnSet = new Set(columns);
  return rows.map((row, index) => {
    const extraColumns = Object.keys(row).filter((column) => !columnSet.has(column));
    if (extraColumns.length) {
      throw new Error(`追加データの${index + 1}行目に未登録列があります: ${extraColumns.join(", ")}`);
    }
    return Object.fromEntries(columns.map((column) => [column, row[column] ?? null]));
  });
}

function datasetFile(
  dataset: DatasetResponse,
  rows: Record<string, unknown>[]
): File {
  const columns = dataset.profile.columns.map((column) => column.name);
  const csvRows = [
    columns,
    ...rows.map((row) => columns.map((column) => row[column] ?? null))
  ];
  const csv = `\uFEFF${csvRows.map((row) => row.map(csvCell).join(",")).join("\r\n")}`;
  const stem = dataset.name.replace(/\.[^.]+$/, "") || "bochan_dataset";
  return new File([csv], `${stem}_updated.csv`, { type: "text/csv;charset=utf-8" });
}

/** Append manually entered rows and register the merged data as a new Web dataset. */
export async function appendExperimentRows(
  dataset: DatasetResponse,
  rows: Record<string, unknown>[]
): Promise<DatasetResponse> {
  if (!rows.length) throw new Error("追加する実験データがありません。");
  const complete = await fetchDataset(dataset.dataset_id);
  const appended = normalizedRows(complete, rows);
  return uploadDataset(datasetFile(complete, [...complete.preview, ...appended]));
}

/** Import CSV/Excel experiment rows, merge them with the current data, and re-register the dataset. */
export async function appendExperimentFile(
  dataset: DatasetResponse,
  file: File,
  requiredColumns: string[]
): Promise<{ dataset: DatasetResponse; appendedRows: number }> {
  const imported = await uploadDataset(file);
  const importedComplete = await fetchDataset(imported.dataset_id);
  const importedColumns = new Set(importedComplete.profile.columns.map((column) => column.name));
  const missingRequired = requiredColumns.filter((column) => !importedColumns.has(column));
  if (missingRequired.length) {
    throw new Error(`インポートファイルに結果列がありません: ${missingRequired.join(", ")}`);
  }
  if (!importedComplete.preview.length) {
    throw new Error("インポートファイルに追加できる行がありません。");
  }
  return {
    dataset: await appendExperimentRows(dataset, importedComplete.preview),
    appendedRows: importedComplete.preview.length
  };
}

/** Export one candidate-oriented CSV template for recording experimental outcomes. */
export function downloadExperimentTemplate(
  dataset: DatasetResponse,
  featureColumns: string[],
  targetColumns: string[],
  rows: Record<string, unknown>[]
): void {
  const columns = [...featureColumns, ...targetColumns];
  const csvRows = [
    columns,
    ...rows.map((row) => columns.map((column) => row[column] ?? null))
  ];
  const csv = `\uFEFF${csvRows.map((row) => row.map(csvCell).join(",")).join("\r\n")}`;
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${dataset.name.replace(/\.[^.]+$/, "") || "bochan"}_experiment_results.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}
