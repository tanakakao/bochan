import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode
} from "react";
import { fetchHealth, runRegression, uploadDataset } from "../api";
import type {
  ColumnProfile,
  DatasetResponse,
  RegressionResult,
  SearchVariable
} from "../types";

export type WorkbenchStep = "data" | "prepare" | "optimize" | "results" | "logs";
export type Theme = "light" | "dark";
export type HealthState = {
  status: "loading" | "ready" | "error";
  text: string;
};

export const STEPS: Array<[WorkbenchStep, string, string]> = [
  ["data", "Data", "データ読込"],
  ["prepare", "Prepare", "変数設定"],
  ["optimize", "Optimize", "モデルと探索"],
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
  variables: Record<string, SearchVariable>;
  selectedVariables: SearchVariable[];
  direction: "maximize" | "minimize";
  setDirection: (direction: "maximize" | "minimize") => void;
  modelType: string;
  setModelType: (modelType: string) => void;
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
  handleFile: (file: File | null) => Promise<void>;
  toggleFeature: (name: string) => void;
  changeTarget: (name: string) => void;
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
  const [targetColumn, setTargetColumn] = useState("");
  const [variables, setVariables] = useState<Record<string, SearchVariable>>({});
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [modelType, setModelType] = useState("base");
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
  const selectableColumns = columns.filter(
    (column) => column.kind === "numeric" || column.kind === "categorical"
  );
  const targetCandidates = columns.filter((column) => column.kind === "numeric");
  const selectedVariables = useMemo(
    () => featureColumns
      .map((name) => variables[name])
      .filter((value): value is SearchVariable => Boolean(value)),
    [featureColumns, variables]
  );
  const canConfigure = Boolean(dataset && targetColumn && featureColumns.length > 0);

  function setTheme(nextTheme: Theme) {
    setThemeState(nextTheme);
  }

  function canOpenStep(nextStep: WorkbenchStep): boolean {
    if (nextStep === "data" || nextStep === "logs") return true;
    if (nextStep === "prepare") return Boolean(dataset);
    if (nextStep === "optimize") return canConfigure;
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
      const numeric = loaded.profile.columns.filter((column) => column.kind === "numeric");
      const initialTarget = numeric.at(-1)?.name ?? "";
      const initialFeatures = loaded.profile.columns
        .filter(
          (column) =>
            (column.kind === "numeric" || column.kind === "categorical") &&
            column.name !== initialTarget
        )
        .map((column) => column.name);
      setTargetColumn(initialTarget);
      setFeatureColumns(initialFeatures);
      setVariables(
        Object.fromEntries(
          loaded.profile.columns
            .filter((column) => column.kind === "numeric" || column.kind === "categorical")
            .map((column) => [column.name, createVariable(column)])
        )
      );
      setStepState("prepare");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  function toggleFeature(name: string) {
    setFeatureColumns((current) =>
      current.includes(name)
        ? current.filter((column) => column !== name)
        : [...current, name]
    );
  }

  function changeTarget(name: string) {
    setTargetColumn(name);
    setFeatureColumns((current) => current.filter((column) => column !== name));
  }

  function patchVariable(name: string, patch: Partial<SearchVariable>) {
    setVariables((current) => ({
      ...current,
      [name]: { ...current[name], ...patch }
    }));
  }

  async function execute() {
    if (!dataset || !canConfigure) return;
    setBusy("モデル学習と候補探索を実行しています");
    setError(null);
    try {
      const response = await runRegression({
        datasetId: dataset.dataset_id,
        featureColumns,
        targetColumn,
        direction,
        modelType,
        fitMaxiter,
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
    variables,
    selectedVariables,
    direction,
    setDirection,
    modelType,
    setModelType,
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
    handleFile,
    toggleFeature,
    changeTarget,
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
