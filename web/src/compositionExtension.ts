const STORAGE_KEY = "bochan-web-composition-settings";
const ACTIVE_DATASET_KEY = "bochan-web-composition-dataset-id";

export const COMPOSITION_SETTINGS_CHANGE_EVENT = "bochan-composition-settings-change";

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

/** Persist normalized composition settings and notify React subscribers. */
export function saveCompositionSettings(settings: CompositionSettings): void {
  const normalized = normalizeSettings(settings);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
  window.dispatchEvent(new CustomEvent(COMPOSITION_SETTINGS_CHANGE_EVENT));
}

/** Clear a stale composition selection when a newly uploaded dataset becomes active. */
export function activateCompositionDataset(datasetId: string): void {
  const normalizedId = String(datasetId).trim();
  if (!normalizedId) return;
  const previousId = window.localStorage.getItem(ACTIVE_DATASET_KEY);
  if (previousId === normalizedId) return;
  window.localStorage.setItem(ACTIVE_DATASET_KEY, normalizedId);
  saveCompositionSettings({
    ...loadCompositionSettings(),
    enabled: false,
    column: "",
    elements: []
  });
}

/** Convert React-owned settings to the canonical Web backend payload. */
export function compositionSettingsToBackend(
  settings: CompositionSettings
): Record<string, unknown> {
  const bounds = Object.fromEntries(settings.elements.map((element) => [
    element,
    settings.bounds[element] ?? [0, 1]
  ]));
  const steps = Object.fromEntries(settings.elements.flatMap((element) => {
    const step = settings.steps[element];
    return step && step > 0 ? [[element, step]] : [];
  }));
  return {
    enabled: settings.enabled,
    column: settings.column,
    elements: settings.elements,
    normalization: settings.normalization,
    representation: settings.representation,
    reference_element: settings.referenceElement || null,
    pseudocount: settings.pseudocount,
    precision: settings.precision,
    total: 1,
    coordinate_bounds: [settings.coordinateLower, settings.coordinateUpper],
    min_components: settings.minComponents,
    max_components: settings.maxComponents,
    required_components: settings.requiredComponents,
    bounds,
    steps,
    element_constraints: settings.constraints.map((constraint) => ({
      terms: constraint.terms.map((term) => ({
        element: term.element,
        coefficient: term.coefficient
      })),
      operator: constraint.operator,
      rhs: constraint.rhs,
      basis: constraint.basis
    }))
  };
}

/** Restore a saved backend composition payload to the React settings shape. */
export function compositionSettingsFromBackend(value: unknown): CompositionSettings {
  if (!value || typeof value !== "object") return { ...DEFAULT_SETTINGS };
  const raw = value as Record<string, unknown>;
  const coordinateBounds = Array.isArray(raw.coordinate_bounds)
    ? raw.coordinate_bounds
    : [-8, 8];
  return normalizeSettings({
    enabled: raw.enabled ?? Boolean(raw.column),
    column: raw.column,
    elements: raw.elements,
    normalization: raw.normalization,
    representation: raw.representation,
    referenceElement: raw.reference_element,
    pseudocount: raw.pseudocount,
    precision: raw.precision,
    coordinateLower: coordinateBounds[0],
    coordinateUpper: coordinateBounds[1],
    minComponents: raw.min_components,
    maxComponents: raw.max_components,
    requiredComponents: raw.required_components,
    bounds: raw.bounds,
    steps: raw.steps,
    constraints: raw.element_constraints
  });
}

interface FormulaDatasetPayload {
  profile?: { columns?: Array<{ name?: string; kind?: string }> };
  preview?: Record<string, unknown>[];
}

function elementSymbols(formula: unknown): string[] {
  if (typeof formula !== "string") return [];
  const compact = formula.replace(/\s+/g, "");
  if (!compact || !/^[A-Za-z0-9.()[\]·]+$/.test(compact)) return [];
  return [...new Set(compact.match(/[A-Z][a-z]?/g) ?? [])];
}

function formulaLikeColumn(column: string, preview: Record<string, unknown>[]): boolean {
  const values = preview
    .map((row) => row[column])
    .filter((value) => value !== null && value !== undefined);
  if (!values.length) return false;
  const matches = values.filter(
    (value) => typeof value === "string" && elementSymbols(value).length > 0
  );
  return matches.length / values.length >= 0.7;
}

/** Mark formula-like or explicitly configured composition columns as selectable inputs. */
export function markFormulaLikeColumns<T extends FormulaDatasetPayload>(
  payload: T,
  configuredColumn = ""
): T {
  const preview = payload.preview ?? [];
  for (const column of payload.profile?.columns ?? []) {
    if (
      column.kind === "string" &&
      column.name &&
      (column.name === configuredColumn || formulaLikeColumn(column.name, preview))
    ) {
      column.kind = "categorical";
    }
  }
  return payload;
}
