const STORAGE_KEY = "bochan-web-composition-settings";
const CHANGE_EVENT = "bochan-composition-settings-change";
const OBSERVER_OPTIONS: MutationObserverInit = { subtree: true, childList: true };
const COMPOSITION_ANCHOR_SELECTOR = [
  ".feature-variable-choice",
  ".model-primary-grid",
  ".feature-constraint-panel",
  ".composition-model-settings-host",
  ".composition-constraint-settings-host"
].join(", ");

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

interface CompositionSettings {
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

interface DatasetPayload {
  profile?: { columns?: Array<{ name?: string; kind?: string }> };
  preview?: Record<string, unknown>[];
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

let latestDataset: DatasetPayload | null = null;
let installed = false;
let originalFetch: typeof window.fetch | null = null;
let renderScheduled = false;
let compositionObserver: MutationObserver | null = null;
const panelRenderSignatures = new WeakMap<HTMLElement, string>();

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

export function loadCompositionSettings(): CompositionSettings {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored ? normalizeSettings(JSON.parse(stored)) : { ...DEFAULT_SETTINGS };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function saveCompositionSettings(value: CompositionSettings): void {
  const settings = normalizeSettings(value);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

function patchSettings(patch: Partial<CompositionSettings>): void {
  saveCompositionSettings({ ...loadCompositionSettings(), ...patch });
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

function cardColumn(card: Element): string {
  return card.querySelector(".variable-choice-main span")?.textContent?.trim() ?? "";
}

function synchronizePrepareControls(): void {
  const settings = loadCompositionSettings();
  document.querySelectorAll<HTMLElement>(".feature-variable-choice").forEach((card) => {
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
    if (select) select.value = settings.enabled && settings.column === column ? "composition" : "normal";
  });
}

function escapeHtml(value: unknown): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function elementOptions(elements: string[], selected: string): string {
  return elements.map((element) => (
    `<option value="${escapeHtml(element)}"${element === selected ? " selected" : ""}>${escapeHtml(element)}</option>`
  )).join("");
}

function renderElementTable(settings: CompositionSettings): string {
  if (!settings.elements.length) return "";
  const rows = settings.elements.map((element) => {
    const pair = settings.bounds[element] ?? [0, 1];
    const step = settings.steps[element];
    const required = settings.requiredComponents.includes(element);
    return `<tr>
      <td><strong>${escapeHtml(element)}</strong></td>
      <td><input data-element="${escapeHtml(element)}" data-field="lower" type="number" min="0" max="1" step="any" value="${pair[0]}"></td>
      <td><input data-element="${escapeHtml(element)}" data-field="upper" type="number" min="0" max="1" step="any" value="${pair[1]}"></td>
      <td><input data-element="${escapeHtml(element)}" data-field="step" type="number" min="0" max="1" step="any" value="${step ?? ""}" placeholder="任意"></td>
      <td><input data-element="${escapeHtml(element)}" data-field="required" class="table-checkbox" type="checkbox"${required ? " checked" : ""}></td>
    </tr>`;
  }).join("");
  return `<section class="composition-element-section">
    <div class="constraint-section-heading"><div><h4>元素ごとの比率制約</h4><p>上下限と刻みは、合計1に正規化した比率で指定します。</p></div></div>
    <div class="table-wrap"><table class="composition-element-table">
      <thead><tr><th>元素</th><th>下限</th><th>上限</th><th>刻み</th><th>必須</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </section>`;
}

function renderConstraint(settings: CompositionSettings, constraint: ElementConstraint, index: number): string {
  const terms = constraint.terms.map((term, termIndex) => `<div class="composition-term-row">
    <input data-constraint="${constraint.id}" data-term="${termIndex}" data-term-field="coefficient" type="number" step="any" value="${term.coefficient}">
    <span>×</span>
    <select data-constraint="${constraint.id}" data-term="${termIndex}" data-term-field="element">${elementOptions(settings.elements, term.element)}</select>
    <button data-action="remove-term" data-constraint="${constraint.id}" data-term="${termIndex}" type="button" class="secondary compact"${constraint.terms.length <= 1 ? " disabled" : ""}>削除</button>
  </div>`).join("");
  return `<div class="constraint-card">
    <div class="constraint-card-heading">
      <div><span class="constraint-index">${index + 1}</span><strong>Σ（係数 × 元素比）</strong></div>
      <button data-action="remove-constraint" data-constraint="${constraint.id}" type="button" class="constraint-delete">×</button>
    </div>
    <div class="composition-term-list">${terms}
      <button data-action="add-term" data-constraint="${constraint.id}" type="button" class="secondary compact">項を追加</button>
    </div>
    <div class="composition-constraint-relation">
      <select data-constraint="${constraint.id}" data-constraint-field="operator">
        <option value="="${constraint.operator === "=" ? " selected" : ""}>=</option>
        <option value="<="${constraint.operator === "<=" ? " selected" : ""}>≤</option>
        <option value=">="${constraint.operator === ">=" ? " selected" : ""}>≥</option>
      </select>
      <input data-constraint="${constraint.id}" data-constraint-field="rhs" type="number" step="any" value="${constraint.rhs}">
      <select data-constraint="${constraint.id}" data-constraint-field="basis">
        <option value="atomic_amount"${constraint.basis === "atomic_amount" ? " selected" : ""}>原子比・mol比基準</option>
        <option value="weight_amount"${constraint.basis === "weight_amount" ? " selected" : ""}>重量基準</option>
      </select>
    </div>
  </div>`;
}

function modelPanelHtml(settings: CompositionSettings): string {
  if (!settings.enabled || !settings.column) {
    return `<article class="panel composition-settings-panel composition-model-settings-panel">
      <div class="panel-title"><div><span class="panel-kicker">COMPOSITION MODEL</span><h3>組成式のモデル変換</h3><p>Select画面で説明変数の入力表記を「組成式」にすると設定できます。</p></div><span class="status-chip">Off</span></div>
    </article>`;
  }
  const reference = settings.representation === "alr"
    ? `<label><span>参照元素</span><select id="composition-reference"><option value="">自動</option>${elementOptions(settings.elements, settings.referenceElement)}</select></label>`
    : "";
  const coordinate = settings.representation !== "fractions"
    ? `<label><span>変換座標の下限</span><input id="composition-coordinate-lower" type="number" step="any" value="${settings.coordinateLower}"></label>
       <label><span>変換座標の上限</span><input id="composition-coordinate-upper" type="number" step="any" value="${settings.coordinateUpper}"></label>`
    : "";
  return `<article class="panel composition-settings-panel composition-model-settings-panel">
    <div class="panel-title">
      <div><span class="panel-kicker">COMPOSITION MODEL</span><h3>組成式のモデル変換</h3><p>${escapeHtml(settings.column)}を合計1の組成比へ変換し、学習モデルの入力座標を作成します。</p></div>
      <span class="status-chip ${settings.elements.length >= 2 ? "success" : "warning"}">${settings.elements.length} elements</span>
    </div>
    <div class="composition-basic-grid">
      <label><span>組成式列</span><input value="${escapeHtml(settings.column)}" disabled></label>
      <label><span>変換方法</span><select id="composition-representation">
        <option value="fractions"${settings.representation === "fractions" ? " selected" : ""}>Fraction</option>
        <option value="clr"${settings.representation === "clr" ? " selected" : ""}>CLR</option>
        <option value="alr"${settings.representation === "alr" ? " selected" : ""}>ALR</option>
        <option value="ilr"${settings.representation === "ilr" ? " selected" : ""}>ILR</option>
      </select></label>
      <label><span>組成基準</span><select id="composition-normalization">
        <option value="atomic_fraction"${settings.normalization === "atomic_fraction" ? " selected" : ""}>原子比・mol比</option>
        <option value="weight_fraction"${settings.normalization === "weight_fraction" ? " selected" : ""}>重量比</option>
      </select></label>
      <label><span>候補元素</span><input id="composition-elements" value="${escapeHtml(settings.elements.join(", "))}" placeholder="Fe, Co, Ni"></label>
      ${reference}
      <label><span>表示桁数</span><input id="composition-precision" type="number" min="1" max="12" value="${settings.precision}"></label>
      ${coordinate}
    </div>
    ${settings.elements.length < 2 ? '<p class="settings-note warning-text">候補元素を2種類以上指定してください。</p>' : ""}
    <p class="settings-note">元素比率の上下限、刻み、必須元素、使用元素数、元素間制約は候補提案画面の「組成候補の元素制約」で設定します。</p>
  </article>`;
}

function constraintPanelHtml(settings: CompositionSettings): string {
  const constraints = settings.constraints.length
    ? `<div class="constraint-list">${settings.constraints.map((constraint, index) => renderConstraint(settings, constraint, index)).join("")}</div>`
    : `<div class="constraint-empty">元素間制約は設定されていません。</div>`;
  return `<article class="panel composition-settings-panel composition-constraint-settings-panel">
    <div class="panel-title">
      <div><span class="panel-kicker">COMPOSITION CONSTRAINTS</span><h3>組成候補の元素制約</h3><p>${escapeHtml(settings.column)}の候補生成時に、使用元素数と各元素の比率制約を適用します。</p></div>
      <span class="status-chip ${settings.elements.length >= 2 ? "success" : "warning"}">${settings.constraints.length} constraints</span>
    </div>
    <div class="composition-basic-grid">
      <label><span>最小使用元素数</span><input id="composition-min-components" type="number" min="1" max="${Math.max(settings.elements.length, 1)}" value="${settings.minComponents}"></label>
      <label><span>最大使用元素数</span><input id="composition-max-components" type="number" min="${settings.minComponents}" max="${Math.max(settings.elements.length, 1)}" value="${settings.maxComponents ?? settings.elements.length}"></label>
    </div>
    ${renderElementTable(settings)}
    <section class="composition-element-section">
      <div class="constraint-section-heading"><div><h4>元素間の線形制約</h4><p>例: Sr − 0.5 × La = 0 とすると、SrをLaの半分に固定できます。</p></div><button id="composition-add-constraint" type="button" class="secondary"${settings.elements.length ? "" : " disabled"}>制約を追加</button></div>
      ${constraints}
    </section>
    <p class="settings-note">比率は合計1を維持してrepairされます。A/Bサイト分離と複数組成式列は今回の対象外です。</p>
  </article>`;
}

function updateElements(value: string): void {
  const current = loadCompositionSettings();
  const elements = uniqueStrings(value);
  saveCompositionSettings({
    ...current,
    elements,
    requiredComponents: current.requiredComponents.filter((element) => elements.includes(element)),
    bounds: Object.fromEntries(elements.map((element) => [element, current.bounds[element] ?? [0, 1]])),
    steps: Object.fromEntries(elements.map((element) => [element, current.steps[element] ?? null])),
    constraints: current.constraints.map((constraint) => ({
      ...constraint,
      terms: constraint.terms.filter((term) => elements.includes(term.element))
    })),
    referenceElement: elements.includes(current.referenceElement) ? current.referenceElement : "",
    maxComponents: elements.length || null
  });
}

function patchConstraint(id: string, patch: Partial<ElementConstraint>): void {
  const current = loadCompositionSettings();
  patchSettings({
    constraints: current.constraints.map((constraint) => (
      constraint.id === id ? { ...constraint, ...patch } : constraint
    ))
  });
}

function bindPanel(host: HTMLElement): void {
  host.querySelector<HTMLSelectElement>("#composition-representation")?.addEventListener("change", (event) => {
    patchSettings({ representation: (event.currentTarget as HTMLSelectElement).value as Representation });
  });
  host.querySelector<HTMLSelectElement>("#composition-normalization")?.addEventListener("change", (event) => {
    patchSettings({ normalization: (event.currentTarget as HTMLSelectElement).value as Normalization });
  });
  host.querySelector<HTMLInputElement>("#composition-elements")?.addEventListener("change", (event) => {
    updateElements((event.currentTarget as HTMLInputElement).value);
  });
  host.querySelector<HTMLSelectElement>("#composition-reference")?.addEventListener("change", (event) => {
    patchSettings({ referenceElement: (event.currentTarget as HTMLSelectElement).value });
  });
  host.querySelector<HTMLInputElement>("#composition-precision")?.addEventListener("change", (event) => {
    patchSettings({ precision: Math.max(1, Math.min(12, Number((event.currentTarget as HTMLInputElement).value))) });
  });
  host.querySelector<HTMLInputElement>("#composition-min-components")?.addEventListener("change", (event) => {
    patchSettings({ minComponents: Math.max(1, Number((event.currentTarget as HTMLInputElement).value)) });
  });
  host.querySelector<HTMLInputElement>("#composition-max-components")?.addEventListener("change", (event) => {
    patchSettings({ maxComponents: Math.max(1, Number((event.currentTarget as HTMLInputElement).value)) });
  });
  host.querySelector<HTMLInputElement>("#composition-coordinate-lower")?.addEventListener("change", (event) => {
    patchSettings({ coordinateLower: Number((event.currentTarget as HTMLInputElement).value) });
  });
  host.querySelector<HTMLInputElement>("#composition-coordinate-upper")?.addEventListener("change", (event) => {
    patchSettings({ coordinateUpper: Number((event.currentTarget as HTMLInputElement).value) });
  });
  host.querySelector<HTMLButtonElement>("#composition-add-constraint")?.addEventListener("click", () => {
    const current = loadCompositionSettings();
    const first = current.elements[0] ?? "";
    const second = current.elements[1] ?? first;
    patchSettings({
      constraints: [...current.constraints, {
        id: newId(),
        terms: [
          { element: first, coefficient: 1 },
          { element: second, coefficient: -1 }
        ].filter((term) => term.element),
        operator: "=",
        rhs: 0,
        basis: "atomic_amount"
      }]
    });
  });

  host.addEventListener("change", (event) => {
    const target = event.target as HTMLInputElement | HTMLSelectElement;
    const element = target.dataset.element;
    const field = target.dataset.field;
    if (element && field) {
      const current = loadCompositionSettings();
      if (field === "required") {
        const selected = current.requiredComponents.includes(element);
        patchSettings({
          requiredComponents: selected
            ? current.requiredComponents.filter((value) => value !== element)
            : [...current.requiredComponents, element]
        });
      } else if (field === "step") {
        patchSettings({
          steps: {
            ...current.steps,
            [element]: target.value.trim() ? Math.max(0, Number(target.value)) : null
          }
        });
      } else {
        const pair = current.bounds[element] ?? [0, 1];
        const value = Number(target.value);
        patchSettings({
          bounds: {
            ...current.bounds,
            [element]: field === "lower"
              ? [Math.max(0, value), Math.max(value, pair[1])]
              : [pair[0], Math.max(pair[0], value)]
          }
        });
      }
      return;
    }

    const constraintId = target.dataset.constraint;
    const termIndex = target.dataset.term === undefined ? null : Number(target.dataset.term);
    const termField = target.dataset.termField;
    const constraintField = target.dataset.constraintField;
    if (constraintId && termIndex !== null && termField) {
      const current = loadCompositionSettings();
      const constraint = current.constraints.find((item) => item.id === constraintId);
      if (!constraint) return;
      patchConstraint(constraintId, {
        terms: constraint.terms.map((term, index) => index === termIndex
          ? {
              ...term,
              [termField]: termField === "coefficient" ? Number(target.value) : target.value
            }
          : term)
      });
    } else if (constraintId && constraintField) {
      const value: unknown = constraintField === "rhs" ? Number(target.value) : target.value;
      patchConstraint(constraintId, { [constraintField]: value } as Partial<ElementConstraint>);
    }
  });

  host.addEventListener("click", (event) => {
    const target = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-action]");
    if (!target) return;
    const action = target.dataset.action;
    const constraintId = target.dataset.constraint;
    if (!constraintId) return;
    const current = loadCompositionSettings();
    const constraint = current.constraints.find((item) => item.id === constraintId);
    if (action === "remove-constraint") {
      patchSettings({ constraints: current.constraints.filter((item) => item.id !== constraintId) });
    } else if (action === "add-term" && constraint) {
      patchConstraint(constraintId, {
        terms: [...constraint.terms, { element: current.elements[0] ?? "", coefficient: 1 }]
      });
    } else if (action === "remove-term" && constraint) {
      const termIndex = Number(target.dataset.term);
      patchConstraint(constraintId, {
        terms: constraint.terms.filter((_term, index) => index !== termIndex)
      });
    }
  });
}

function renderPanel(host: HTMLElement, html: string): HTMLElement {
  if (panelRenderSignatures.get(host) === html) return host;
  const replacement = host.cloneNode(false) as HTMLElement;
  replacement.innerHTML = html;
  panelRenderSignatures.set(replacement, html);
  host.replaceWith(replacement);
  bindPanel(replacement);
  return replacement;
}

function synchronizeModelPanel(): void {
  const modelGrid = document.querySelector<HTMLElement>(".model-primary-grid");
  if (!modelGrid || !modelGrid.parentElement) {
    document.querySelector(".composition-model-settings-host")?.remove();
    return;
  }
  let host = modelGrid.parentElement.querySelector<HTMLElement>(":scope > .composition-model-settings-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "composition-model-settings-host";
    modelGrid.insertAdjacentElement("afterend", host);
  } else if (modelGrid.nextElementSibling !== host) {
    modelGrid.insertAdjacentElement("afterend", host);
  }
  renderPanel(host, modelPanelHtml(loadCompositionSettings()));
}

function synchronizeConstraintPanel(): void {
  const featurePanel = document.querySelector<HTMLElement>(".feature-constraint-panel");
  const settings = loadCompositionSettings();
  if (!featurePanel || !featurePanel.parentElement || !settings.enabled || !settings.column) {
    document.querySelector(".composition-constraint-settings-host")?.remove();
    return;
  }
  let host = featurePanel.parentElement.querySelector<HTMLElement>(":scope > .composition-constraint-settings-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "composition-constraint-settings-host";
    featurePanel.parentElement.insertBefore(host, featurePanel);
  } else if (host.nextElementSibling !== featurePanel) {
    featurePanel.parentElement.insertBefore(host, featurePanel);
  }
  renderPanel(host, constraintPanelHtml(settings));
}

function nodeContainsCompositionAnchor(node: Node): boolean {
  if (!(node instanceof Element)) return false;
  return node.matches(COMPOSITION_ANCHOR_SELECTOR)
    || Boolean(node.querySelector(COMPOSITION_ANCHOR_SELECTOR));
}

function mutationAffectsComposition(record: MutationRecord): boolean {
  return [...Array.from(record.addedNodes), ...Array.from(record.removedNodes)]
    .some(nodeContainsCompositionAnchor);
}

function observeCompositionMutations(): void {
  compositionObserver?.observe(document.documentElement, OBSERVER_OPTIONS);
}

function runSynchronization(): void {
  compositionObserver?.disconnect();
  try {
    synchronizePrepareControls();
    synchronizeModelPanel();
    synchronizeConstraintPanel();
  } finally {
    observeCompositionMutations();
  }
}

function synchronize(): void {
  if (renderScheduled) return;
  renderScheduled = true;
  queueMicrotask(() => {
    renderScheduled = false;
    runSynchronization();
  });
}

export function installCompositionExtension(): void {
  if (installed) return;
  installed = true;
  installFetchAdapter();
  compositionObserver = new MutationObserver((records) => {
    if (records.some(mutationAffectsComposition)) synchronize();
  });
  observeCompositionMutations();
  window.addEventListener(CHANGE_EVENT, synchronize);
  synchronize();
}
