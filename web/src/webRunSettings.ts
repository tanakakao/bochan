export type FeatureConstraintOperator = ">" | "<" | "=";

/** One linear constraint: sum(coefficients[name] * name) operator value. */
export interface FeatureConstraint {
  id: string;
  variables: string[];
  coefficients: Record<string, number>;
  operator: FeatureConstraintOperator;
  value: number;
}

/** Limits how many selected numeric variables remain non-zero in one candidate. */
export interface SelectionCountConstraint {
  enabled: boolean;
  variables: string[];
  k: number;
}

export type FeatureMissingStrategy = "drop" | "impute";
export type ContinuousImputeStrategy = "mean" | "iterative";

/** Missing-value handling applied only to explanatory variables. */
export interface FeatureMissingSettings {
  strategy: FeatureMissingStrategy;
  continuousStrategy: ContinuousImputeStrategy;
  categoricalStrategy: "mode";
  imputeMaxIter: number;
  imputeRandomState: number | null;
  multipleImputeSamplePosterior: boolean;
}

export type SearchMethod =
  | "normal"
  | "torch"
  | "ga"
  | "sa"
  | "pso"
  | "cmaes"
  | "thompson_sampling"
  | "nsgaii";

const CONSTRAINTS_KEY = "bochan-web-feature-constraints";
const SELECTION_COUNT_KEY = "bochan-web-selection-count-constraint";
const SEARCH_METHOD_KEY = "bochan-web-search-method";
const FEATURE_MISSING_KEY = "bochan-web-feature-missing";

function storage(): Storage | null {
  return typeof window === "undefined" ? null : window.localStorage;
}

export function newConstraintId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `constraint-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function finiteNumber(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeConstraint(raw: unknown): FeatureConstraint | null {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Record<string, unknown>;
  const id = typeof value.id === "string" && value.id ? value.id : newConstraintId();
  const operator = value.operator === ">" || value.operator === "=" ? value.operator : "<";

  // Migrate the previous single-term shape without discarding user settings.
  if (typeof value.variable === "string" && value.variable) {
    return {
      id,
      variables: [value.variable],
      coefficients: { [value.variable]: finiteNumber(value.coefficient, 1) },
      operator,
      value: finiteNumber(value.value, 0)
    };
  }

  const variables = Array.isArray(value.variables)
    ? value.variables.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
  const rawCoefficients = value.coefficients && typeof value.coefficients === "object"
    ? value.coefficients as Record<string, unknown>
    : {};
  const coefficients = Object.fromEntries(
    variables.map((name) => [name, finiteNumber(rawCoefficients[name], 1)])
  );
  return {
    id,
    variables: [...new Set(variables)],
    coefficients,
    operator,
    value: finiteNumber(value.value, 0)
  };
}

export function loadFeatureConstraints(): FeatureConstraint[] {
  const value = storage()?.getItem(CONSTRAINTS_KEY);
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.map(normalizeConstraint).filter((item): item is FeatureConstraint => Boolean(item));
  } catch {
    return [];
  }
}

export function saveFeatureConstraints(constraints: FeatureConstraint[]): void {
  storage()?.setItem(CONSTRAINTS_KEY, JSON.stringify(constraints));
}

export function loadSelectionCountConstraint(): SelectionCountConstraint {
  const value = storage()?.getItem(SELECTION_COUNT_KEY);
  if (!value) return { enabled: false, variables: [], k: 1 };
  try {
    const parsed = JSON.parse(value) as Partial<SelectionCountConstraint>;
    return {
      enabled: Boolean(parsed.enabled),
      variables: Array.isArray(parsed.variables)
        ? parsed.variables.filter((item): item is string => typeof item === "string" && item.length > 0)
        : [],
      k: Math.max(1, Math.trunc(finiteNumber(parsed.k, 1)))
    };
  } catch {
    return { enabled: false, variables: [], k: 1 };
  }
}

export function saveSelectionCountConstraint(value: SelectionCountConstraint): void {
  storage()?.setItem(SELECTION_COUNT_KEY, JSON.stringify(value));
}

export function loadFeatureMissingSettings(): FeatureMissingSettings {
  // Historical behavior remains the default until imputation is selected.
  const fallback: FeatureMissingSettings = {
    strategy: "drop",
    continuousStrategy: "mean",
    categoricalStrategy: "mode",
    imputeMaxIter: 10,
    imputeRandomState: null,
    multipleImputeSamplePosterior: false
  };
  const value = storage()?.getItem(FEATURE_MISSING_KEY);
  if (!value) return fallback;
  try {
    const parsed = JSON.parse(value) as Partial<FeatureMissingSettings>;
    return {
      strategy: parsed.strategy === "impute" ? "impute" : "drop",
      continuousStrategy: parsed.continuousStrategy === "iterative" ? "iterative" : "mean",
      categoricalStrategy: "mode",
      imputeMaxIter: Math.max(1, Math.trunc(finiteNumber(parsed.imputeMaxIter, 10))),
      imputeRandomState: parsed.imputeRandomState === null || parsed.imputeRandomState === undefined
        ? null
        : Math.trunc(finiteNumber(parsed.imputeRandomState, 0)),
      multipleImputeSamplePosterior: Boolean(parsed.multipleImputeSamplePosterior)
    };
  } catch {
    return fallback;
  }
}

export function saveFeatureMissingSettings(value: FeatureMissingSettings): void {
  storage()?.setItem(FEATURE_MISSING_KEY, JSON.stringify(value));
}

export function loadSearchMethod(): SearchMethod {
  const value = storage()?.getItem(SEARCH_METHOD_KEY) as SearchMethod | null;
  return value && [
    "normal",
    "torch",
    "ga",
    "sa",
    "pso",
    "cmaes",
    "thompson_sampling",
    "nsgaii"
  ].includes(value) ? value : "normal";
}

export function saveSearchMethod(method: SearchMethod): void {
  storage()?.setItem(SEARCH_METHOD_KEY, method);
}
