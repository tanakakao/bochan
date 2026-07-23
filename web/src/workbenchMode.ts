import { useSyncExternalStore } from "react";

export type WorkbenchMode = "simple" | "advanced";

const WORKBENCH_MODE_KEY = "bochan-web-workbench-mode";
const listeners = new Set<() => void>();

function readWorkbenchMode(): WorkbenchMode {
  if (typeof window === "undefined") return "advanced";
  return window.localStorage.getItem(WORKBENCH_MODE_KEY) === "simple" ? "simple" : "advanced";
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const handleStorage = (event: StorageEvent) => {
    if (event.key === WORKBENCH_MODE_KEY) listener();
  };
  window.addEventListener("storage", handleStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", handleStorage);
  };
}

export function setWorkbenchMode(mode: WorkbenchMode): void {
  window.localStorage.setItem(WORKBENCH_MODE_KEY, mode);
  listeners.forEach((listener) => listener());
}

export function useWorkbenchMode(): WorkbenchMode {
  return useSyncExternalStore(subscribe, readWorkbenchMode, () => "advanced");
}
