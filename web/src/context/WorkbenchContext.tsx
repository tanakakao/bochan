import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode
} from "react";
import { fetchHealth, runRegression, uploadDataset } from "../api";
import { getColumnClassValues } from "../targetSettingUtils";
import type {
  AcquisitionFamily,
  ColumnProfile,
  DatasetResponse,
  Direction,
  RegressionResult,
  SearchVariable,
  TargetClassValue,
  TargetSetting
} from "../types";

export type WorkbenchStep = "data" | "prepare" | "settings" | "optimize" | "results" | "logs";
export type Theme = "light" | "dark";
export type HealthState = {
  status: "loading" | "ready" | "error";
  text: string;
};

export const STEPS: Array<[WorkbenchStep, string, string]> = [
  ["data", "Data", "データ読込"],
  ["prepare", "Select", "変数選択"],
  ["settings", "Settings", "目的・探索設定"],
  ["optimize", "Optimize", "モデルと候補生成"],
  ["results", "Results", "候補と可視化"],
  ["logs", "Logs", "実行履歴"]
];

function numberOrUndefined(value: string): number | undefined {
  if (value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function createVariable(column: ColumnProfile): SearchVariable {
  if (column.kind === "categorical") {
    return {
      name: column.name,
      type: "categorical",
      fixed: false,
      categories: column.values ?? []
    };
  }
  return {
    name: column.name,
    type: "numeric",
    lower: column.min ?? undefined,
    upper: column.max ?? undefined,
    fixed: false
  };
}

function createTargetSetting(
  column: ColumnProfile,
  preview: Record<string, unknown>[]
): TargetSetting {
  const common = {
    target: column.name,
    optimize: true,
    direction: "maximize" as Direction,
    goal: "none" as const,
    value: null
  };
  if (column.kind === "numeric") {
    return { ...common, task_type: "regression" };
  }

  const classes = getColumnClassValues(column, preview);
  if (classes.length === 2) {
    return {
      ...common,
      task_type: "classification",
      target_class: classes[1],
      target_classes: [classes[1]]
    };
  }
  return {
    ...common,
    task_type: "classification",
    target_classes: classes.length ? [classes[0]] : []
  };
}

function finiteNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function classKey(value: TargetClassValue): string {
  return String(value);
}

function containsClass(values: TargetClassValue[], value: TargetClassValue | null | undefined): boolean {
  if (value === null || value === undefined) return false;
  const requested = classKey(value);
  return values.some((candidate) => classKey(candidate) === requested);
}

function sameClassSet(left: TargetClassValue[], right: TargetClassValue[]): boolean {
  if (left.length !== right.length) return false;
  const leftKeys = new Set(left.map(classKey));
  const rightKeys = new Set(right.map(classKey));
  return leftKeys.size === left.length && rightKeys.size === right.length &&
    [...leftKeys].every((value) => rightKeys.has(value));
}

function validateVariable(variable: SearchVariable): boolean {
  if (variable.type === "categorical") {
    if (!variable.fixed) return true;
    return variable.fixed_value !== undefined && String(variable.fixed_value).trim() !== "";
  }
  const lower = finiteNumber(variable.lower);
  const upper = finiteNumber(variable.upper);
  if (lower === null || upper === null || lower >= upper) return false;
  if (variable.step !== undefined && (finiteNumber(variable.step) ?? 0) <= 0) return false;
  if (!variable.fixed) return true;
  const fixed = finiteNumber(variable.fixed_value);
  return fixed !== null && fixed >= lower && fixed <= upper;
}

function validateTargetSetting(
  setting: TargetSetting,
  column: ColumnProfile | undefined,
  preview: Record<string, unknown>[]
): boolean {
  if (!column) return false;
  if (setting.goal === "target" && !setting.optimize) return false;

  if (setting.task_type === "regression") {
    if (column.kind !== "numeric") return false;
    return setting.goal === "none" || finiteNumber(setting.value) !== null;
  }

  const classes = getColumnClassValues(column, preview);
  if (classes.length < 2) return false;

  if (setting.task_type === "classification") {
    if (classes.length === 2) {
      if (!containsClass(classes, setting.target_class)) return false;
    } else {
      const selected = setting.target_classes ?? [];
      if (selected.length === 0 || !selected.every((value) => containsClass(classes, value))) return false;
    }
    if (setting.goal === "above" || setting.goal === "below") {
      const threshold = finiteNumber(setting.value);
      return threshold !== null && threshold >= 0 && threshold <= 1;
    }
    return setting.goal === "none";
  }

  const order = setting.class_order ?? [];
  if (!sameClassSet(order, classes)) return false;
  if (setting.goal === "none") return true;
  if (setting.goal === "above" || setting.goal === "below") {
    return containsClass(order, setting.value as TargetClassValue);
  }
  const targets = setting.target_values ?? [];
  return targets.length > 0 && targets.every((value) => containsClass(order, value));
}

interface WorkbenchContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  step: WorkbenchStep;
  setStep: (step: WorkbenchStep) => void;
  canOpenStep: (step: WorkbenchStep) => boolean;
  health: HealthState;
  busy: string | null;
  error: string | null;
  setError: (error: string | null) => void;
  dataset: DatasetResponse | null;
  columns: ColumnProfile[];
  selectableColumns: ColumnProfile[];
  targetCandidates: ColumnProfile[];
  featureColumns: string[];
  targetColumn: string;
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  selectedTargetSettings: TargetSetting[];
  optimizedTargetSettings: TargetSetting[];
  targetDirections: Record<string, Direction>;
  direction: Direction;
  variables: Record<string, SearchVariable>;
  selectedVariables: SearchVariable[];
  modelType: string;
  setModelType: (modelType: string) => void;
  acquisitionFamily: AcquisitionFamily;
  setAcquisitionFamily: (family: AcquisitionFamily) => void;
  acquisition: string;
  setAcquisition: (acquisition: string) => void;
  beta: number;
  setBeta: (beta: number) => void;
  fitMaxiter: number;
  setFitMaxiter: (fitMaxiter: number) => void;
  q: number;
  setQ: (q: number) => void;
  numRestarts: number;
  setNumRestarts: (numRestarts: number) => void;
  rawSamples: number;
  setRawSamples: (rawSamples: number) => void;
  result: RegressionResult | null;
  canConfigure: boolean;
  settingsValid: boolean;
  handleFile: (file: File | null) => Promise<void>;
  toggleFeature: (name: string) => void;
  toggleTarget: (name: string) => void;
  patchTargetSetting: (target: string, patch: Partial<TargetSetting>) => void;
  patchVariable: (name: string, patch: Partial<SearchVariable>) => void;
  execute: () => Promise<void>;
  numberOrUndefined: (value: string) => number | undefined;
}

const WorkbenchContext = createContext<WorkbenchContextValue | null>(null);

export function WorkbenchProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = window.localStorage.getItem("bochan-theme");
    if (stored === "dark" || stored === "light") return stored;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const [step, setStepState] = useState<WorkbenchStep>("data");
  const [health, setHealth] = useState<HealthState>({ status: "loading", text: "接続確認中" });
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [targetColumns, setTargetColumns] = useState<string[]>([]);
  const [targetSettings, setTargetSettings] = useState<Record<string, TargetSetting>>({});
  const [variables, setVariables] = useState<Record<string, SearchVariable>>({});
  const [modelType, setModelType] = useState("base");
  const [acquisitionFamily, setAcquisitionFamily] = useState<AcquisitionFamily>("bayesian_optimization");
  const [acquisition, setAcquisition] = useState("EI");
  const [beta, setBeta] = useState(2);
  const [fitMaxiter, setFitMaxiter] = useState(128);
  const [q, setQ] = useState(3);
  const [numRestarts, setNumRestarts] = useState(10);
  const [rawSamples, setRawSamples] = useState(256);
  const [result, setResult] = useState<RegressionResult | null>(null);

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

  const columns = dataset?.profile.columns ?? [];
  const preview = dataset?.preview ?? [];
  const selectableColumns = columns.filter(
    (column) => column.kind === "numeric" || column.kind === "categorical"
  );
  const targetCandidates = selectableColumns;
  const selectedTargetSettings = useMemo(
    () => targetColumns
      .map((name) => targetSettings[name])
      .filter((setting): setting is TargetSetting => Boolean(setting)),
    [targetColumns, targetSettings]
  );
  const optimizedTargetSettings = useMemo(
    () => selectedTargetSettings.filter((setting) => setting.optimize),
    [selectedTargetSettings]
  );
  const targetColumn = optimizedTargetSettings[0]?.target ?? targetColumns[0] ?? "";
  const selectedVariables = useMemo(
    () => featureColumns
      .map((name) => variables[name])
      .filter((value): value is SearchVariable => Boolean(value)),
    [featureColumns, variables]
  );
  const targetDirections = useMemo(
    () => Object.fromEntries(selectedTargetSettings.map((setting) => [
      setting.target,
      setting.goal === "target" ? "maximize" : setting.direction
    ])) as Record<string, Direction>,
    [selectedTargetSettings]
  );
  const direction = targetDirections[targetColumn] ?? "maximize";
  const canConfigure = Boolean(
    dataset &&
    targetColumns.length > 0 &&
    featureColumns.length > 0 &&
    targetColumns.every((target) => !featureColumns.includes(target))
  );
  const settingsValid = Boolean(
    canConfigure &&
    optimizedTargetSettings.length > 0 &&
    selectedTargetSettings.length === targetColumns.length &&
    selectedTargetSettings.every((setting) => validateTargetSetting(
      setting,
      columns.find((column) => column.name === setting.target),
      preview
    )) &&
    selectedVariables.every(validateVariable)
  );

  function setTheme(nextTheme: Theme) {
    setThemeState(nextTheme);
  }

  function canOpenStep(nextStep: WorkbenchStep): boolean {
    if (nextStep === "data" || nextStep === "logs") return true;
    if (nextStep === "prepare") return Boolean(dataset);
    if (nextStep === "settings") return canConfigure;
    if (nextStep === "optimize") return settingsValid;
    if (nextStep === "results") return Boolean(result);
    return false;
  }

  function setStep(nextStep: WorkbenchStep) {
    if (canOpenStep(nextStep)) setStepState(nextStep);
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    setBusy("データを読み込んでいます");
    setError(null);
    setResult(null);
    try {
      const loaded = await uploadDataset(file);
      setDataset(loaded);
      const candidates = loaded.profile.columns.filter(
        (column) => column.kind === "numeric" || column.kind === "categorical"
      );
      const initialTarget = candidates.at(-1);
      const initialFeatures = candidates
        .filter((column) => column.name !== initialTarget?.name)
        .map((column) => column.name);
      setTargetColumns(initialTarget ? [initialTarget.name] : []);
      setTargetSettings(initialTarget ? {
        [initialTarget.name]: createTargetSetting(initialTarget, loaded.preview)
      } : {});
      setFeatureColumns(initialFeatures);
      setVariables(Object.fromEntries(candidates.map((column) => [column.name, createVariable(column)])));
      setModelType("base");
      setAcquisitionFamily("bayesian_optimization");
      setAcquisition("EI");
      setStepState("prepare");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  function toggleFeature(name: string) {
    if (targetColumns.includes(name)) return;
    setFeatureColumns((current) =>
      current.includes(name)
        ? current.filter((column) => column !== name)
        : [...current, name]
    );
  }

  function toggleTarget(name: string) {
    const profile = columns.find((column) => column.name === name);
    if (!profile) return;
    setTargetColumns((current) => {
      const selected = current.includes(name);
      const next = selected ? current.filter((column) => column !== name) : [...current, name];
      setFeatureColumns((features) => features.filter((column) => !next.includes(column)));
      setTargetSettings((settings) => {
        const updated = { ...settings };
        if (selected) delete updated[name];
        else updated[name] = updated[name] ?? createTargetSetting(profile, preview);
        return updated;
      });
      return next;
    });
  }

  function patchTargetSetting(target: string, patch: Partial<TargetSetting>) {
    setTargetSettings((current) => ({
      ...current,
      [target]: { ...current[target], ...patch, target }
    }));
  }

  function patchVariable(name: string, patch: Partial<SearchVariable>) {
    setVariables((current) => ({
      ...current,
      [name]: { ...current[name], ...patch }
    }));
  }

  async function execute() {
    if (!dataset || !settingsValid) return;
    setBusy("モデル学習と候補探索を実行しています");
    setError(null);
    try {
      const response = await runRegression({
        datasetId: dataset.dataset_id,
        featureColumns,
        targetColumn,
        targetColumns,
        targetSettings: selectedTargetSettings,
        targetDirections,
        direction,
        modelType,
        fitMaxiter,
        acquisitionFamily,
        acquisition,
        beta,
        q,
        numRestarts,
        rawSamples,
        searchSpace: selectedVariables
      });
      setResult(response);
      setStepState("results");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  const value: WorkbenchContextValue = {
    theme,
    setTheme,
    step,
    setStep,
    canOpenStep,
    health,
    busy,
    error,
    setError,
    dataset,
    columns,
    selectableColumns,
    targetCandidates,
    featureColumns,
    targetColumn,
    targetColumns,
    targetSettings,
    selectedTargetSettings,
    optimizedTargetSettings,
    targetDirections,
    direction,
    variables,
    selectedVariables,
    modelType,
    setModelType,
    acquisitionFamily,
    setAcquisitionFamily,
    acquisition,
    setAcquisition,
    beta,
    setBeta,
    fitMaxiter,
    setFitMaxiter,
    q,
    setQ,
    numRestarts,
    setNumRestarts,
    rawSamples,
    setRawSamples,
    result,
    canConfigure,
    settingsValid,
    handleFile,
    toggleFeature,
    toggleTarget,
    patchTargetSetting,
    patchVariable,
    execute,
    numberOrUndefined
  };

  return <WorkbenchContext.Provider value={value}>{children}</WorkbenchContext.Provider>;
}

export function useWorkbench(): WorkbenchContextValue {
  const value = useContext(WorkbenchContext);
  if (!value) throw new Error("useWorkbench must be used inside WorkbenchProvider.");
  return value;
}
