import { loadCompositionSettings } from "./compositionExtension";

const STORAGE_KEY = "bochan-web-composition-settings";
const CHANGE_EVENT = "bochan-composition-settings-change";
const FEATURE_CARD_SELECTOR = ".feature-variable-choice";

type CompositionSettings = ReturnType<typeof loadCompositionSettings>;

interface DatasetPayload {
  profile?: { columns?: Array<{ name?: string; kind?: string }> };
  preview?: Record<string, unknown>[];
}

let installed = false;
let originalFetch: typeof window.fetch | null = null;
let latestDataset: DatasetPayload | null = null;
let synchronizationScheduled = false;

function saveCompositionSettings(settings: CompositionSettings): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

function elementSymbols(formula: unknown): string[] {
  if (typeof formula !== "string") return [];
  const compact = formula.replace(/\s+/g, "");
  if (!compact || !/^[A-Za-z0-9.()[\]·]+$/.test(compact)) return [];
  return [...new Set(compact.match(/[A-Z][a-z]?/g) ?? [])];
}

function looksLikeFormula(value: unknown): boolean {
  return typeof value === "string" && elementSymbols(value).length > 0;
}

function inferElements(column: string): string[] {
  const values = latestDataset?.preview?.map((row) => row[column]).filter(looksLikeFormula) ?? [];
  return [...new Set(values.flatMap(elementSymbols))];
}

function formulaLikeColumn(column: string): boolean {
  const values = latestDataset?.preview
    ?.map((row) => row[column])
    .filter((value) => value !== null && value !== undefined) ?? [];
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
        latestDataset = payload;
        for (const column of payload.profile?.columns ?? []) {
          if (column.kind === "string" && column.name && formulaLikeColumn(column.name)) {
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

function escapeHtml(value: unknown): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function cardColumn(card: Element): string {
  return card.querySelector(".variable-choice-main span")?.textContent?.trim() ?? "";
}

function synchronizePrepareControls(): void {
  const settings = loadCompositionSettings();
  document.querySelectorAll<HTMLElement>(FEATURE_CARD_SELECTOR).forEach((card) => {
    const column = cardColumn(card);
    if (!column) return;
    card.classList.toggle("selected-composition", settings.enabled && settings.column === column);

    let control = card.querySelector<HTMLElement>(":scope > .composition-kind-control");
    if (!control) {
      control = document.createElement("label");
      control.className = "composition-kind-control";
      control.innerHTML = `<span>入力表記</span><select aria-label="${escapeHtml(column)}の入力表記"><option value="normal">通常カテゴリ</option><option value="composition">組成式</option></select>`;
      card.appendChild(control);
      control.querySelector("select")?.addEventListener("change", (event) => {
        const select = event.currentTarget as HTMLSelectElement;
        const current = loadCompositionSettings();
        if (select.value === "composition") {
          const elements = current.column === column && current.elements.length
            ? current.elements
            : inferElements(column);
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
        } else if (current.column === column) {
          saveCompositionSettings({ ...current, enabled: false, column: "" });
        }
      });
    }

    const select = control.querySelector<HTMLSelectElement>("select");
    if (select) {
      select.value = settings.enabled && settings.column === column ? "composition" : "normal";
    }
  });
}

function nodeContainsFeatureCard(node: Node): boolean {
  if (!(node instanceof Element)) return false;
  return node.matches(FEATURE_CARD_SELECTOR) || Boolean(node.querySelector(FEATURE_CARD_SELECTOR));
}

function scheduleSynchronization(): void {
  if (synchronizationScheduled) return;
  synchronizationScheduled = true;
  queueMicrotask(() => {
    synchronizationScheduled = false;
    synchronizePrepareControls();
  });
}

/** Installs composition data transport and Select-page controls only. */
export function installCompositionRuntime(): void {
  if (installed) return;
  installed = true;
  installFetchAdapter();

  const observer = new MutationObserver((records) => {
    if (records.some((record) => Array.from(record.addedNodes).some(nodeContainsFeatureCard))) {
      scheduleSynchronization();
    }
  });
  observer.observe(document.documentElement, { subtree: true, childList: true });

  window.addEventListener(CHANGE_EVENT, scheduleSynchronization);
  scheduleSynchronization();
}
