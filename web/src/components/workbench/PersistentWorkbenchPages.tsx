import type { WorkbenchStep } from "../../context/WorkbenchContext";
import { useWorkbench } from "../../context/WorkbenchContext";
import PersistentWorkbenchPage from "./PersistentWorkbenchPage";
import {
  AUXILIARY_PAGES,
  WORKBENCH_PAGES,
  type AuxiliaryPage
} from "./workbenchPages";
import { createWorkbenchPageResetKeys } from "./workbenchPagePersistence";

const WORKFLOW_PAGE_IDS: WorkbenchStep[] = [
  "data",
  "prepare",
  "settings",
  "optimize",
  "results",
  "logs"
];
const AUXILIARY_PAGE_IDS: AuxiliaryPage[] = ["conversation", "experiment"];

interface PersistentWorkbenchPagesProps {
  step: WorkbenchStep;
  activeAuxiliaryPage: AuxiliaryPage | null;
}

/** Renders only visited pages and preserves their local UI state between navigation changes. */
export default function PersistentWorkbenchPages({
  step,
  activeAuxiliaryPage
}: PersistentWorkbenchPagesProps) {
  const {
    dataset,
    datasetRevision,
    resultRevision
  } = useWorkbench();
  const resetKeys = createWorkbenchPageResetKeys({
    datasetRevision,
    datasetId: dataset?.dataset_id ?? null,
    resultRevision
  });

  return (
    <>
      {WORKFLOW_PAGE_IDS.map((pageId) => (
        <PersistentWorkbenchPage
          key={pageId}
          pageId={pageId}
          Page={WORKBENCH_PAGES[pageId]}
          active={!activeAuxiliaryPage && step === pageId}
          cacheKey={resetKeys[pageId]}
        />
      ))}
      {AUXILIARY_PAGE_IDS.map((pageId) => (
        <PersistentWorkbenchPage
          key={pageId}
          pageId={pageId}
          Page={AUXILIARY_PAGES[pageId]}
          active={activeAuxiliaryPage === pageId}
          cacheKey={resetKeys[pageId]}
        />
      ))}
    </>
  );
}
