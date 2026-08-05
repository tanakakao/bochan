import { loadCompositionSettings } from "./compositionExtension";

type CompositionSettings = ReturnType<typeof loadCompositionSettings>;

interface DatasetPayload {
  profile?: { columns?: Array<{ name?: string; kind?: string }> };
  preview?: Record<string, unknown>[];
}

let installed = false;
let originalFetch: typeof window.fetch | null = null;

function elementSymbols(formula: unknown): string[] {
  if (typeof formula !== "string") return [];
  const compact = formula.replace(/\s+/g, "");
  if (!compact || !/^[A-Za-z0-9.()[\]·]+$/.test(compact)) return [];
  return [...new Set(compact.match(/[A-Z][a-z]?/g) ?? [])];
}

function looksLikeFormula(value: unknown): boolean {
  return typeof value === "string" && elementSymbols(value).length > 0;
}

function formulaLikeColumn(
  column: string,
  preview: Record<string, unknown>[]
): boolean {
  const values = preview
    .map((row) => row[column])
    .filter((value) => value !== null && value !== undefined);
  return values.length > 0 && values.filter(looksLikeFormula).length / values.length >= 0.7;
}

function responseWithJson(response: Response, payload: unknown): Response {
  const headers = new Headers(response.headers);
  headers.set("Content-Type", "application/json");
  return new Response(JSON.stringify(payload), {
    status: response.status,
    statusText: response.statusText,
    headers
  });
}

function backendSettings(settings: CompositionSettings): Record<string, unknown> {
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

function installFetchAdapter(): void {
  if (originalFetch) return;
  originalFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const rawUrl = typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
    const url = new URL(rawUrl, window.location.href);
    let nextInit = init;

    if (url.pathname.endsWith("/regression/run") && typeof init?.body === "string") {
      const settings = loadCompositionSettings();
      if (settings.enabled && settings.column) {
        const payload = JSON.parse(init.body) as Record<string, any>;
        payload.model_kwargs = {
          ...(payload.model_kwargs ?? {}),
          web_composition: backendSettings(settings)
        };
        delete payload.model_kwargs.web_reuse_model_run_id;
        payload.search_space = Array.isArray(payload.search_space)
          ? payload.search_space.map((spec: Record<string, unknown>) => (
              spec.name === settings.column
                ? {
                    name: settings.column,
                    type: "auto",
                    fixed: false,
                    fixed_value: null,
                    categories: null,
                    lower: null,
                    upper: null,
                    step: null
                  }
                : spec
            ))
          : payload.search_space;
        nextInit = { ...init, body: JSON.stringify(payload) };
      }
    }

    const response = await originalFetch!(input, nextInit);
    const method = String(nextInit?.method ?? "GET").toUpperCase();
    if (url.pathname.endsWith("/datasets") && method === "POST" && response.ok) {
      try {
        const payload = await response.clone().json() as DatasetPayload;
        const preview = payload.preview ?? [];
        for (const column of payload.profile?.columns ?? []) {
          if (
            column.kind === "string" &&
            column.name &&
            formulaLikeColumn(column.name, preview)
          ) {
            column.kind = "categorical";
          }
        }
        return responseWithJson(response, payload);
      } catch {
        return response;
      }
    }
    return response;
  };
}

/** Installs composition-aware dataset and regression API transport without DOM mutation. */
export function installCompositionRuntime(): void {
  if (installed) return;
  installed = true;
  installFetchAdapter();
}
