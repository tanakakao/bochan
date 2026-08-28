import { useEffect, useState } from "react";
import { WORKBENCH_RUN_SETTINGS_CHANGE_EVENT } from "../workbenchSettingsEvents";

/** Re-render WorkbenchProvider when localStorage-backed run settings change in this tab. */
export function useWorkbenchExternalSettingsRevision(): number {
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    const refresh = () => setRevision((current) => current + 1);
    window.addEventListener(WORKBENCH_RUN_SETTINGS_CHANGE_EVENT, refresh);
    return () => window.removeEventListener(WORKBENCH_RUN_SETTINGS_CHANGE_EVENT, refresh);
  }, []);

  return revision;
}
