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
  SearchVariable,
  AcquisitionFamily,
  KSparseConfig,
  LinearConstraint,
  OutcomeConstraint,
  TaskType
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

/**
 * Converts an input string to a finite number or undefined.
 *
 * Args:
 *   value: Raw input value from a form control.
 *
 * Returns:
 *   A finite number when parsing succeeds; otherwise undefined.
 */
function numberOrUndefined(value: string): number | undefined {
  if (value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/**
 * Creates an editable search-space variable from a column profile.
 *
 * Args:
 *   column: Dataset column metadata returned by the API.
 *
 * Returns:
 *   A search-space variable initialized from observed column values.
 */
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
  taskType: TaskType;
  setTaskType: (taskType: TaskType) => void;
  ordinalOrder: string[];
  setOrdinalOrder: (ordinalOrder: string[]) => void;
  targetColumns: string[];
  toggleTarget: (name: string) => void;
  direction: "maximize" | "minimize";
  setDirection: (direction: "maximize" | "minimize") => void;
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
  outcomeConstraints: OutcomeConstraint[];
  addOutcomeConstraint: () => void;
  patchOutcomeConstraint: (id: string, patch: Partial<OutcomeConstraint>) => void;
  removeOutcomeConstraint: (id: string) => void;
  linearConstraints: LinearConstraint[];
  addLinearConstraint: (kind: LinearConstraint["kind"]) => void;
  patchLinearConstraint: (id: string, patch: Partial<LinearConstraint>) => void;
  patchLinearConstraintTerm: (id: string, variable: string, coefficient: number | undefined) => void;
  removeLinearConstraint: (id: string) => void;
  kSparse: KSparseConfig;
  setKSparse: (kSparse: KSparseConfig) => void;
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
  const [targetColumns, setTargetColumns] = useState<string[]>([]);
  const [variables, setVariables] = useState<Record<string, SearchVariable>>({});
  const [taskType, setTaskType] = useState<TaskType>("regression");
  const [ordinalOrder, setOrdinalOrder] = useState<string[]>([]);
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [modelType, setModelType] = useState("base");
  const [acquisitionFamily, setAcquisitionFamily] = useState<AcquisitionFamily>("bayesian_optimization");
  const [acquisition, setAcquisition] = useState("EI");
  const [beta, setBeta] = useState(2);
  const [fitMaxiter, setFitMaxiter] = useState(128);
  const [q, setQ] = useState(3);
  const [numRestarts, setNumRestarts] = useState(10);
  const [rawSamples, setRawSamples] = useState(256);
  const [outcomeConstraints, setOutcomeConstraints] = useState<OutcomeConstraint[]>([]);
  const [linearConstraints, setLinearConstraints] = useState<LinearConstraint[]>([]);
  const [kSparse, setKSparse] = useState<KSparseConfig>({ enabled: false, k: 1, variables: [] });
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
  const targetCandidates = columns.filter((column) => taskType === "regression" ? column.kind === "numeric" : column.kind === "numeric" || column.kind === "categorical");
  const targetColumn = targetColumns[0] ?? "";
  const selectedVariables = useMemo(
    () => featureColumns
      .map((name) => variables[name])
      .filter((value): value is SearchVariable => Boolean(value)),
    [featureColumns, variables]
  );
  const canConfigure = Boolean(dataset && targetColumns.length > 0 && featureColumns.length > 0);

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
      setTargetColumns(initialTarget ? [initialTarget] : []);
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

  /**
   * Replaces target selection with a single target.
   *
   * Args:
   *   name: Column name to use as the sole target.
   */
  function changeTarget(name: string) {
    setTargetColumns(name ? [name] : []);
    setFeatureColumns((current) => current.filter((column) => column !== name));
  }

  /**
   * Toggles a target column for multi-objective configuration.
   *
   * Args:
   *   name: Column name to add or remove from selected targets.
   */
  function toggleTarget(name: string) {
    setTargetColumns((current) => {
      const next = current.includes(name) ? current.filter((column) => column !== name) : [...current, name];
      setFeatureColumns((features) => features.filter((column) => !next.includes(column)));
      return next;
    });
  }

  /**
   * Updates one search-space variable without replacing the rest.
   *
   * Args:
   *   name: Variable name to update.
   *   patch: Partial variable settings to merge.
   */
  function patchVariable(name: string, patch: Partial<SearchVariable>) {
    setVariables((current) => ({
      ...current,
      [name]: { ...current[name], ...patch }
    }));
  }

  /**
   * Adds a blank outcome constraint row.
   *
   * Returns:
   *   Nothing. The context state receives one new default constraint.
   */
  function addOutcomeConstraint() {
    const target = targetColumns[0] ?? targetCandidates[0]?.name ?? "";
    setOutcomeConstraints((current) => [...current, { id: crypto.randomUUID(), target, operator: ">=", value: 0 }]);
  }

  /**
   * Updates one outcome constraint row.
   *
   * Args:
   *   id: Constraint identifier.
   *   patch: Partial settings to merge into the constraint.
   */
  function patchOutcomeConstraint(id: string, patch: Partial<OutcomeConstraint>) {
    setOutcomeConstraints((current) => current.map((constraint) => constraint.id === id ? { ...constraint, ...patch } : constraint));
  }

  /**
   * Removes one outcome constraint row.
   *
   * Args:
   *   id: Constraint identifier to remove.
   */
  function removeOutcomeConstraint(id: string) {
    setOutcomeConstraints((current) => current.filter((constraint) => constraint.id !== id));
  }

  /**
   * Adds a blank linear input constraint row.
   *
   * Args:
   *   kind: Whether the new row is an equality or inequality constraint.
   */
  function addLinearConstraint(kind: LinearConstraint["kind"]) {
    setLinearConstraints((current) => [...current, { id: crypto.randomUUID(), kind, terms: {}, operator: kind === "equality" ? "=" : "<=", rhs: 0 }]);
  }

  /**
   * Updates one linear input constraint row.
   *
   * Args:
   *   id: Constraint identifier.
   *   patch: Partial settings to merge into the constraint.
   */
  function patchLinearConstraint(id: string, patch: Partial<LinearConstraint>) {
    setLinearConstraints((current) => current.map((constraint) => constraint.id === id ? { ...constraint, ...patch } : constraint));
  }

  /**
   * Updates a coefficient in a linear input constraint.
   *
   * Args:
   *   id: Constraint identifier.
   *   variable: Search-space variable name.
   *   coefficient: New coefficient, or undefined to remove the term.
   */
  function patchLinearConstraintTerm(id: string, variable: string, coefficient: number | undefined) {
    setLinearConstraints((current) => current.map((constraint) => {
      if (constraint.id !== id) return constraint;
      const terms = { ...constraint.terms };
      if (coefficient === undefined) delete terms[variable];
      else terms[variable] = coefficient;
      return { ...constraint, terms };
    }));
  }

  /**
   * Removes one linear input constraint row.
   *
   * Args:
   *   id: Constraint identifier to remove.
   */
  function removeLinearConstraint(id: string) {
    setLinearConstraints((current) => current.filter((constraint) => constraint.id !== id));
  }

  /**
   * Runs optimization with the current workbench configuration.
   *
   * Returns:
   *   A promise that resolves after candidates are generated or an error is shown.
   */
  async function execute() {
    if (!dataset || !canConfigure) return;
    setBusy("モデル学習と候補探索を実行しています");
    setError(null);
    try {
      const response = await runRegression({
        datasetId: dataset.dataset_id,
        featureColumns,
        targetColumn,
        targetColumns,
        taskType,
        ordinalOrder,
        direction,
        modelType,
        fitMaxiter,
        acquisitionFamily,
        acquisition,
        beta,
        q,
        numRestarts,
        rawSamples,
        searchSpace: selectedVariables,
        outcomeConstraints,
        linearConstraints,
        kSparse
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
    variables,
    selectedVariables,
    taskType,
    setTaskType,
    ordinalOrder,
    setOrdinalOrder,
    toggleTarget,
    direction,
    setDirection,
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
    outcomeConstraints,
    addOutcomeConstraint,
    patchOutcomeConstraint,
    removeOutcomeConstraint,
    linearConstraints,
    addLinearConstraint,
    patchLinearConstraint,
    patchLinearConstraintTerm,
    removeLinearConstraint,
    kSparse,
    setKSparse,
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
