import type { WorkbenchStep } from "../../context/WorkbenchContext";
import type { AuxiliaryPage } from "./workbenchPages";

export type PersistedWorkbenchPage = WorkbenchStep | AuxiliaryPage;
export type WorkbenchPageResetKeys = Record<PersistedWorkbenchPage, string>;

interface WorkbenchPageResetInput {
  datasetRevision: number;
  datasetId: string | null;
  resultRevision: number;
}

/**
 * Defines when mounted page-local UI state must be discarded.
 *
 * Dataset-bound setup pages survive navigation but reset when a new dataset
 * workspace replaces the current one. Results and experiment-entry state are
 * bound to the current generated result instead. A stale result created by
 * appending experiment rows intentionally keeps the same result revision so
 * the completed experiment entry remains reviewable until a new result exists.
 */
export function createWorkbenchPageResetKeys({
  datasetRevision,
  datasetId,
  resultRevision
}: WorkbenchPageResetInput): WorkbenchPageResetKeys {
  const datasetKey = `dataset:${datasetRevision}:${datasetId ?? "none"}`;
  const resultKey = `result:${resultRevision}`;

  return {
    data: datasetKey,
    prepare: datasetKey,
    settings: datasetKey,
    optimize: datasetKey,
    results: resultKey,
    logs: "logs",
    conversation: datasetKey,
    experiment: resultKey
  };
}
