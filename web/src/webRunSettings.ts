export type FeatureConstraintOperator = ">" | "<" | "=";

export interface FeatureConstraint {
  id: string;
  variable: string;
  coefficient: number;
  operator: FeatureConstraintOperator;
  value: number;
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
const SEARCH_METHOD_KEY = "bochan-web-search-method";

function storage(): Storage | null {
  return typeof window === "undefined" ? null : window.localStorage;
}

export function newConstraintId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `constraint-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function loadFeatureConstraints(): FeatureConstraint[] {
  const value = storage()?.getItem(CONSTRAINTS_KEY);
  if (!value) return [];
  try {
    const parsed = JSON.parse(value) as FeatureConstraint[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveFeatureConstraints(constraints: FeatureConstraint[]): void {
  storage()?.setItem(CONSTRAINTS_KEY, JSON.stringify(constraints));
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
