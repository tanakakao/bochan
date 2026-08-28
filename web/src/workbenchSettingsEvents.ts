export const WORKBENCH_RUN_SETTINGS_CHANGE_EVENT = "bochan:workbench-run-settings-change";

/** Notify the active workbench that a persisted run setting changed in this tab. */
export function notifyWorkbenchRunSettingsChanged(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(WORKBENCH_RUN_SETTINGS_CHANGE_EVENT));
}
