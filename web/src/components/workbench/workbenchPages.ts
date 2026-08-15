import type { ComponentType } from "react";
import type { WorkbenchStep } from "../../context/WorkbenchContext";
import ConversationPage from "../../pages/ConversationPage";
import DataPage from "../../pages/DataPage";
import ExperimentPage from "../../pages/ExperimentPage";
import LogsPage from "../../pages/LogsPage";
import OptimizePage from "../../pages/OptimizePage";
import PreparePage from "../../pages/PreparePage";
import ResultsPage from "../../pages/ResultsPage";
import SettingsPage from "../../pages/SettingsPage";

export type AuxiliaryPage = "conversation" | "experiment";

export const WORKBENCH_PAGES: Record<WorkbenchStep, ComponentType> = {
  data: DataPage,
  prepare: PreparePage,
  settings: SettingsPage,
  optimize: OptimizePage,
  results: ResultsPage,
  logs: LogsPage
};

export const AUXILIARY_PAGES: Record<AuxiliaryPage, ComponentType> = {
  conversation: ConversationPage,
  experiment: ExperimentPage
};

export const WORKBENCH_ICONS: Record<WorkbenchStep, string> = {
  data: "▦",
  prepare: "◇",
  settings: "⌘",
  optimize: "↗",
  results: "◎",
  logs: "≡"
};

export function currentAuxiliaryPage(): AuxiliaryPage | null {
  if (window.location.hash === "#conversation") return "conversation";
  if (window.location.hash === "#experiment") return "experiment";
  return null;
}

export function clearAuxiliaryHash(): void {
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}
