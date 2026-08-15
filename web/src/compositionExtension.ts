const STORAGE_KEY = "bochan-web-composition-settings";

type Representation = "fractions" | "clr" | "alr" | "ilr";
type Normalization = "atomic_fraction" | "weight_fraction";
type ConstraintOperator = "=" | "<=" | ">=";
type ConstraintBasis = "atomic_amount" | "weight_amount";

interface ConstraintTerm {
  element: string;
  coefficient: number;
}

interface ElementConstraint {
  id: string;
  terms: ConstraintTerm[];
  operator: ConstraintOperator;
  rhs: number;
  basis: ConstraintBasis;
}

export interface CompositionSettings {
  enabled: boolean;
  column: string;
  elements: string[];
  normalization: Normalization;
  representation: Representation;
  referenceElement: string;
  pseudocount: number;
  precision: number;
  coordinateLower: number;
  coordinateUpper: number;
  minComponents: number;
  maxComponents: number | null;
  requiredComponents: string[];
  bounds: Record<string, [number, number]>;
  steps: Record<string, number | null>;
  constraints: ElementConstraint[];
}

const DEFAULT_SETTINGS: CompositionSettings = {
  enabled: false,
  column: "",
  elements: [],
  normalization: "atomic_fraction",
  representation: "ilr",
  referenceElement: "",
  pseudocount: 1e-12,
  precision: 6,
  coordinateLower: -8,
  coordinateUpper: 8,
  minComponents: 1,
  maxComponents: null,
  requiredComponents: [],
  bounds: {},
  steps: {},
  constraints: []
};

function newId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `composition-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function finiteNumber(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function uniqueStrings(value: unknown): string[] {
  const values = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? value.split(/[,\s]+/)
      : [];
  return [...new Set(values.map((item) => String(item).trim()).filter(Boolean))];
}

function normalizeConstraint(value: unknown, elements: Set<string>): ElementConstraint | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const terms = Array.isArray(raw.terms)
    ? raw.terms.flatMap((item): ConstraintTerm[] => {
        if (!item || typeof item !== "object") return [];
        const term = item as Record<string, unknown>;
        const element = String(term.element ?? "").trim();
        if (!elements.has(element)) return [];
        return [{ element, coefficient: finiteNumber(term.coefficient, 1) }];
      })
    : [];
  const operator: ConstraintOperator = raw.operator === "<=" || raw.operator === ">="
    ? raw.operator
    : "=";
  const basis: ConstraintBasis = raw.basis === "weight_amount"
    ? "weight_amount"
    : "atomic_amount";
  return {
    id: String(raw.id || newId()),
    terms,
    operator,
    rhs: finiteNumber(raw.rhs, 0),
    basis
  };
}

function normalizeSettings(value: unknown): CompositionSettings {
  if (!value || typeof value !== "object") return { ...DEFAULT_SETTINGS };
  const raw = value as Record<string, unknown>;
  const elements = uniqueStrings(raw.elements);
  const elementSet = new Set(elements);
  const rawBounds = raw.bounds && typeof raw.bounds === "object"
    ? raw.bounds as Record<string, unknown>
    : {};
  const rawSteps = raw.steps && typeof raw.steps === "object"
    ? raw.steps as Record<string, unknown>
    : {};
  const bounds = Object.fromEntries(elements.map((element) => {
    const rawPair = Array.isArray(rawBounds[element]) ? rawBounds[element] as unknown[] : [0, 1];
    const lower = Math.max(0, finiteNumber(rawPair[0], 0));
    const upper = Math.max(lower, finiteNumber(rawPair[1], 1));
    return [element, [lower, upper] as [number, number]];
  }));
  const steps = Object.fromEntries(elements.map((element) => {
    const value = rawSteps[element];
    if (value === null || value === undefined || value === "") return [element, null];
    const step = finiteNumber(value, 0);
    return [element, step > 0 ? step : null];
  }));
  const constraints = Array.isArray(raw.constraints)
    ? raw.constraints
        .map((item) => normalizeConstraint(item, elementSet))
        .filter((item): item is ElementConstraint => item !== null)
    : [];
  const representationValue = String(raw.representation ?? "ilr");
  const representation: Representation = ["fractions", "clr", "alr", "ilr"].includes(representationValue)
    ? representationValue as Representation
    : "ilr";
  const maxRaw = raw.maxComponents;
  const maxComponents = maxRaw === null || maxRaw === undefined || maxRaw === ""
    ? null
    : Math.max(1, Math.trunc(finiteNumber(maxRaw, elements.length || 1)));
  const column = String(raw.column ?? "");
  return {
    enabled: Boolean(raw.enabled && column),
    column,
    elements,
    normalization: raw.normalization === "weight_fraction" ? "weight_fraction" : "atomic_fraction",
    representation,
    referenceElement: elementSet.has(String(raw.referenceElement ?? ""))
      ? String(raw.referenceElement)
      : "",
    pseudocount: Math.max(1e-15, finiteNumber(raw.pseudocount, 1e-12)),
    precision: Math.max(1, Math.min(12, Math.trunc(finiteNumber(raw.precision, 6)))),
    coordinateLower: finiteNumber(raw.coordinateLower, -8),
    coordinateUpper: finiteNumber(raw.coordinateUpper, 8),
    minComponents: Math.max(1, Math.trunc(finiteNumber(raw.minComponents, 1))),
    maxComponents,
    requiredComponents: uniqueStrings(raw.requiredComponents).filter((element) => elementSet.has(element)),
    bounds,
    steps,
    constraints
  };
}

/** Load and normalize the composition configuration used by React controls and API transport. */
export function loadCompositionSettings(): CompositionSettings {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored ? normalizeSettings(JSON.parse(stored)) : { ...DEFAULT_SETTINGS };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}