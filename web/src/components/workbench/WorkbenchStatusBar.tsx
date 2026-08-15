import { useWorkbench } from "../../context/WorkbenchContext";
import type { WorkbenchMode } from "../../workbenchMode";
import type { AuxiliaryPage } from "./workbenchPages";
import { API_STATUS_LABELS } from "./workbenchPresentation";

interface WorkbenchStatusBarProps {
  mode: WorkbenchMode;
  activeAuxiliaryPage: AuxiliaryPage | null;
}

export default function WorkbenchStatusBar({
  mode,
  activeAuxiliaryPage
}: WorkbenchStatusBarProps) {
  const { health, dataset, result } = useWorkbench();
  const apiStatusLabel = API_STATUS_LABELS[health.status];
  const resultStale = Boolean(result?.metadata?.stale_after_data_append);

  return (
    <footer className="statusbar" data-tutorial="status">
      <span><span className={`dot ${health.status}`} /> API接続 {apiStatusLabel}</span>
      <span>{activeAuxiliaryPage === "conversation" ? "Conversation mode" : mode === "simple" ? "Simple mode" : "Advanced mode"}</span>
      <span>{dataset ? `${dataset.profile.n_rows} rows` : "No data"}</span>
      <span>{result ? `${result.candidates.length} candidates${resultStale ? " · stale" : ""}` : "No result"}</span>
      <span className="privacy-status">React · FastAPI · BoTorch</span>
    </footer>
  );
}
