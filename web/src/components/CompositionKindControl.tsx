import { useCallback, useEffect, useState } from "react";
import { loadCompositionSettings } from "../compositionExtension";

const STORAGE_KEY = "bochan-web-composition-settings";
const CHANGE_EVENT = "bochan-composition-settings-change";

type CompositionSettings = ReturnType<typeof loadCompositionSettings>;

type Props = {
  column: string;
  preview: Record<string, unknown>[];
  categorical: boolean;
};

function elementSymbols(formula: unknown): string[] {
  if (typeof formula !== "string") return [];
  const compact = formula.replace(/\s+/g, "");
  if (!compact || !/^[A-Za-z0-9.()[\]·]+$/.test(compact)) return [];
  return [...new Set(compact.match(/[A-Z][a-z]?/g) ?? [])];
}

function inferElements(column: string, preview: Record<string, unknown>[]): string[] {
  return [...new Set(preview.flatMap((row) => elementSymbols(row[column])))];
}

function saveCompositionSettings(settings: CompositionSettings): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

/** Selects normal categorical input or chemical-composition parsing for one feature. */
export default function CompositionKindControl({ column, preview, categorical }: Props) {
  const [settings, setSettings] = useState<CompositionSettings>(() => loadCompositionSettings());
  const selected = settings.enabled && settings.column === column;

  const refresh = useCallback(() => {
    setSettings(loadCompositionSettings());
  }, []);

  useEffect(() => {
    window.addEventListener(CHANGE_EVENT, refresh);
    return () => window.removeEventListener(CHANGE_EVENT, refresh);
  }, [refresh]);

  useEffect(() => {
    if (categorical || !selected) return;
    saveCompositionSettings({ ...loadCompositionSettings(), enabled: false, column: "" });
  }, [categorical, selected]);

  if (!categorical) return null;

  function selectComposition(): void {
    const current = loadCompositionSettings();
    const elements = current.column === column && current.elements.length
      ? current.elements
      : inferElements(column, preview);
    saveCompositionSettings({
      ...current,
      enabled: true,
      column,
      elements,
      bounds: Object.fromEntries(elements.map((element) => [
        element,
        current.bounds[element] ?? [0, 1]
      ])),
      steps: Object.fromEntries(elements.map((element) => [
        element,
        current.steps[element] ?? null
      ])),
      maxComponents: elements.length || null
    });
  }

  function selectNormal(): void {
    const current = loadCompositionSettings();
    if (current.column !== column) return;
    saveCompositionSettings({ ...current, enabled: false, column: "" });
  }

  return (
    <div className="composition-kind-control composition-kind-control-segmented">
      <span>入力表記</span>
      <div
        className="composition-kind-segment"
        role="group"
        aria-label={`${column}の入力表記`}
      >
        <button
          type="button"
          className={`composition-kind-option ${selected ? "" : "active"}`}
          aria-pressed={!selected}
          onClick={selectNormal}
        >
          通常
        </button>
        <button
          type="button"
          className={`composition-kind-option ${selected ? "active" : ""}`}
          aria-pressed={selected}
          onClick={selectComposition}
        >
          組成式
        </button>
      </div>
    </div>
  );
}
