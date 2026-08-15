import { useEffect, useState } from "react";
import { fetchHealth } from "../api";
import type { HealthState, Theme, WorkbenchStep } from "./workbenchTypes";

/** Owns application-shell state that is independent of analysis configuration. */
export function useWorkbenchRuntimeState() {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = window.localStorage.getItem("bochan-theme");
    if (stored === "dark" || stored === "light") return stored;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const [step, setStepState] = useState<WorkbenchStep>("data");
  const [health, setHealth] = useState<HealthState>({ status: "loading", text: "接続確認中" });
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("bochan-theme", theme);
  }, [theme]);

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((response) => {
        if (active) setHealth({ status: "ready", text: response.application || "bochan-web" });
      })
      .catch(() => {
        if (active) setHealth({ status: "error", text: "FastAPIに接続できません" });
      });
    return () => {
      active = false;
    };
  }, []);

  return {
    theme,
    setTheme,
    step,
    setStepState,
    health,
    busy,
    setBusy,
    error,
    setError
  };
}
