import React, { useEffect, useMemo, useState } from "react";
import { createRoot, type Root } from "react-dom/client";

const STORAGE_KEY = "bochan-web-composition-settings";
const CHANGE_EVENT = "bochan-composition-settings-change";

type Representation = "fractions" | "clr" | "alr" | "ilr";
type Normalization = "atomic_fraction" | "weight_fraction";
type ConstraintOperator = "=" | "<=" | ">=";
type ConstraintBasis = "atomic_amount" | "weight_amount";

interface ElementTerm {
  element: string;
  coefficient: number;
}

interface ElementConstraint {
  id: string;
  terms: ElementTerm[];
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
let optimizeRoot: Root | null = null;
let optimizeHost: HTMLElement | null = null;

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
  const items = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? value.split(",")
      : [];
  return [...new Set(items.map((item) => String(item).trim()).filter(Boolean))];
}

function normalizeSettings(value: unknown): CompositionSettings {
  if (!value || typeof value !== "object") return { ...DEFAULT_SETTINGS };
  const raw = value as Partial<CompositionSettings>;
  const elements = uniqueStrings(raw.elements);
  const elementSet = new Set(elements);
  const bounds = Object.fromEntries(elements.map((element) => {
    const pair = raw.bounds?.[element];
    const lower = finiteNumber(pair?.[0], 0);
    const upper = finiteNumber(pair?.[1], 1);
    return [element, [Math.max(0, lower), Math.max(lower, upper)] as [number, number]];
  }));
  const steps = Object.fromEntries(elements.map((element) => {
    const step = raw.steps?.[element];
    const parsed = step === null || step === undefined ? null : finiteNumber(step, 0);
    return [element, parsed && parsed > 0 ? parsed : null];
  }));
  const constraints = Array.isArray(raw.constraints)
    ? raw.constraints.map((constraint) => ({
        id: constraint.id || newId(),
        terms: Array.isArray(constraint.terms)
          ? constraint.terms
              .map((term) => ({
                element: String(term.element ?? ""),
                coefficient: finiteNumber(term.coefficient, 1)
              }))
              .filter((term) => elementSet.has(term.element))
          : [],
        operator: constraint.operator === "<=" || constraint.operator === ">="
          ? constraint.operator
          : "=",
        rhs: finiteNumber(constraint.rhs, 0),
        basis: constraint.basis === "weight_amount" ? "weight_amount" : "atomic_amount"
      }))
    : [];
  const maxRaw = raw.maxComponents;
  const maxComponents = maxRaw === null || maxRaw === undefined
    ? null
    : Math.max(1, Math.trunc(finiteNumber(maxRaw, elements.length || 1)));
  return {
    enabled: Boolean(raw.enabled && raw.column),
    column: String(raw.column ?? ""),
    elements,
    normalization: raw.normalization === "weight_fraction" ? "weight_fraction" : "atomic_fraction",
    representation: ["fractions", "clr", "alr", "ilr"].includes(String(raw.representation))
      ? raw.representation as Representation
      : "ilr",
    referenceElement: elementSet.has(String(raw.referenceElement ?? ""))
      ? String(raw.referenceElement)
      : "",
    pseudocount: Math.max(finiteNumber(raw.pseudocount, 1e-12), 1e-15),
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

function saveCompositionSettings(settings: CompositionSettings): void {
  const normalized = normalizeSettings(settings);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: normalized }));
}

function elementSymbols(formula: unknown): string[] {
  if (typeof formula !== "string") return [];
  const compact = formula.replace(/\s+/g, "");
  if (!compact || !/^[A-Za-z0-9.()[\]·]+$/.test(compact)) return [];
  const symbols = compact.match(/[A-Z][a-z]?/g) ?? [];
  return [...new Set(symbols)];
}

function looksLikeFormula(value: unknown): boolean {
  const symbols = elementSymbols(value);
  return symbols.length > 0 && typeof value === "string" && /[A-Z]/.test(value);
}

function inferElements(column: string): string[] {
  const values = latestDataset?.preview?.map((row) => row[column]).filter(looksLikeFormula) ?? [];
  return [...new Set(values.flatMap(elementSymbols))];
}

function formulaLikeColumn(column: string): boolean {
  const values = latestDataset?.preview?.map((row) => row[column]).filter((value) => value !== null && value !== undefined) ?? [];
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
    const rawUrl = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
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
        // Reuse signatures in the current workbench do not yet include the
        // composition transform. Retrain instead of reusing an incompatible model.
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
    if (url.pathname.endsWith("/datasets") && response.ok && String(nextInit?.method ?? "GET").toUpperCase() === "POST") {
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

function updatePrepareCardClasses(): void {
  const settings = loadCompositionSettings();
  document.querySelectorAll(".feature-variable-choice").forEach((card) => {
    card.classList.toggle("selected-composition", settings.enabled && cardColumn(card) === settings.column);
  });
}

function installPrepareControls(): void {
  document.querySelectorAll<HTMLElement>(".feature-variable-choice").forEach((card) => {
    const column = cardColumn(card);
    if (!column || card.querySelector(":scope > .composition-kind-control")) return;
    const label = document.createElement("label");
    label.className = "composition-kind-control";
    const caption = document.createElement("span");
    caption.textContent = "入力表記";
    const select = document.createElement("select");
    select.setAttribute("aria-label", `${column}の入力表記`);
    select.innerHTML = "<option value=\"normal\">通常カテゴリ</option><option value=\"composition\">組成式</option>";
    const current = loadCompositionSettings();
    select.value = current.enabled && current.column === column ? "composition" : "normal";
    select.addEventListener("change", () => {
      const settings = loadCompositionSettings();
      if (select.value === "composition") {
        const inferred = inferElements(column);
        const elements = settings.column === column && settings.elements.length
          ? settings.elements
          : inferred;
        saveCompositionSettings({
          ...settings,
          enabled: true,
          column,
          elements,
          bounds: Object.fromEntries(elements.map((element) => [
            element,
            settings.bounds[element] ?? [0, 1]
          ])),
          steps: Object.fromEntries(elements.map((element) => [
            element,
            settings.steps[element] ?? null
          ])),
          maxComponents: settings.maxComponents ?? (elements.length || null)
        });
      } else if (settings.column === column) {
        saveCompositionSettings({ ...settings, enabled: false, column: "" });
      }
      document.querySelectorAll<HTMLSelectElement>(".composition-kind-control select").forEach((item) => {
        const itemColumn = cardColumn(item.closest(".feature-variable-choice")!);
        const next = loadCompositionSettings();
        item.value = next.enabled && next.column === itemColumn ? "composition" : "normal";
      });
      updatePrepareCardClasses();
    });
    label.append(caption, select);
    card.appendChild(label);
  });
  updatePrepareCardClasses();
}

function useSettings(): [CompositionSettings, (next: CompositionSettings) => void] {
  const [settings, setSettings] = useState(loadCompositionSettings);
  useEffect(() => {
    const listener = () => setSettings(loadCompositionSettings());
    window.addEventListener(CHANGE_EVENT, listener);
    window.addEventListener("storage", listener);
    return () => {
      window.removeEventListener(CHANGE_EVENT, listener);
      window.removeEventListener("storage", listener);
    };
  }, []);
  return [settings, saveCompositionSettings];
}

function splitElements(value: string): string[] {
  return [...new Set(value.split(/[,\s]+/).map((item) => item.trim()).filter(Boolean))];
}

function CompositionPanel() {
  const [settings, update] = useSettings();
  const elementText = settings.elements.join(", ");
  const elementSet = useMemo(() => new Set(settings.elements), [settings.elements]);

  function patch(patchValue: Partial<CompositionSettings>) {
    update({ ...settings, ...patchValue });
  }

  function setElements(text: string) {
    const elements = splitElements(text);
    patch({
      elements,
      requiredComponents: settings.requiredComponents.filter((element) => elements.includes(element)),
      bounds: Object.fromEntries(elements.map((element) => [element, settings.bounds[element] ?? [0, 1]])),
      steps: Object.fromEntries(elements.map((element) => [element, settings.steps[element] ?? null])),
      constraints: settings.constraints.map((constraint) => ({
        ...constraint,
        terms: constraint.terms.filter((term) => elements.includes(term.element))
      })),
      maxComponents: settings.maxComponents === null
        ? (elements.length || null)
        : Math.min(settings.maxComponents, Math.max(elements.length, 1))
    });
  }

  function patchElement(element: string, kind: "lower" | "upper" | "step", value: string) {
    if (kind === "step") {
      patch({
        steps: {
          ...settings.steps,
          [element]: value.trim() === "" ? null : Math.max(0, finiteNumber(value, 0))
        }
      });
      return;
    }
    const pair = settings.bounds[element] ?? [0, 1];
    const parsed = finiteNumber(value, kind === "lower" ? pair[0] : pair[1]);
    patch({
      bounds: {
        ...settings.bounds,
        [element]: kind === "lower"
          ? [Math.max(0, parsed), Math.max(parsed, pair[1])]
          : [pair[0], Math.max(pair[0], parsed)]
      }
    });
  }

  function toggleRequired(element: string) {
    const selected = settings.requiredComponents.includes(element);
    patch({
      requiredComponents: selected
        ? settings.requiredComponents.filter((value) => value !== element)
        : [...settings.requiredComponents, element]
    });
  }

  function addConstraint() {
    const first = settings.elements[0] ?? "";
    const second = settings.elements[1] ?? first;
    patch({
      constraints: [
        ...settings.constraints,
        {
          id: newId(),
          terms: [
            { element: first, coefficient: 1 },
            { element: second, coefficient: -1 }
          ].filter((term) => term.element),
          operator: "=",
          rhs: 0,
          basis: "atomic_amount"
        }
      ]
    });
  }

  function patchConstraint(id: string, value: Partial<ElementConstraint>) {
    patch({
      constraints: settings.constraints.map((constraint) => (
        constraint.id === id ? { ...constraint, ...value } : constraint
      ))
    });
  }

  function patchTerm(id: string, index: number, value: Partial<ElementTerm>) {
    const constraint = settings.constraints.find((item) => item.id === id);
    if (!constraint) return;
    patchConstraint(id, {
      terms: constraint.terms.map((term, termIndex) => (
        termIndex === index ? { ...term, ...value } : term
      ))
    });
  }

  if (!settings.enabled || !settings.column) {
    return (
      <article className="panel composition-settings-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">COMPOSITION</span>
            <h3>組成比設定</h3>
            <p>Select画面で説明変数の入力表記を「組成式」にすると設定できます。</p>
          </div>
          <span className="status-chip">Off</span>
        </div>
      </article>
    );
  }

  const validCoordinateBounds = settings.coordinateLower < settings.coordinateUpper;
  const validElements = settings.elements.length >= 2;
  const validConstraints = settings.constraints.every((constraint) => (
    constraint.terms.length > 0 &&
    constraint.terms.every((term) => elementSet.has(term.element) && Number.isFinite(term.coefficient)) &&
    Number.isFinite(constraint.rhs)
  ));

  return (
    <article className="panel composition-settings-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">COMPOSITION</span>
          <h3>単一組成式の比率探索</h3>
          <p>{settings.column}を合計1の組成比として変換し、候補を組成式へ戻します。</p>
        </div>
        <span className={`status-chip ${validElements && validCoordinateBounds && validConstraints ? "success" : "warning"}`}>
          {settings.elements.length} elements
        </span>
      </div>

      <div className="composition-basic-grid">
        <label>
          <span>組成式列</span>
          <input value={settings.column} disabled />
        </label>
        <label>
          <span>変換方法</span>
          <select value={settings.representation} onChange={(event) => patch({ representation: event.target.value as Representation })}>
            <option value="fractions">Fraction</option>
            <option value="clr">CLR</option>
            <option value="alr">ALR</option>
            <option value="ilr">ILR</option>
          </select>
        </label>
        <label>
          <span>組成基準</span>
          <select value={settings.normalization} onChange={(event) => patch({ normalization: event.target.value as Normalization })}>
            <option value="atomic_fraction">原子比・mol比</option>
            <option value="weight_fraction">重量比</option>
          </select>
        </label>
        <label>
          <span>候補元素</span>
          <input
            value={elementText}
            placeholder="Fe, Co, Ni"
            onChange={(event) => setElements(event.target.value)}
          />
        </label>
        {settings.representation === "alr" && (
          <label>
            <span>参照元素</span>
            <select value={settings.referenceElement} onChange={(event) => patch({ referenceElement: event.target.value })}>
              <option value="">自動</option>
              {settings.elements.map((element) => <option key={element} value={element}>{element}</option>)}
            </select>
          </label>
        )}
        <label>
          <span>表示桁数</span>
          <input type="number" min={1} max={12} value={settings.precision} onChange={(event) => patch({ precision: Math.max(1, Math.min(12, Number(event.target.value))) })} />
        </label>
        <label>
          <span>最小使用元素数</span>
          <input type="number" min={1} max={Math.max(settings.elements.length, 1)} value={settings.minComponents} onChange={(event) => patch({ minComponents: Math.max(1, Number(event.target.value)) })} />
        </label>
        <label>
          <span>最大使用元素数</span>
          <input type="number" min={settings.minComponents} max={Math.max(settings.elements.length, 1)} value={settings.maxComponents ?? settings.elements.length} onChange={(event) => patch({ maxComponents: Math.max(settings.minComponents, Number(event.target.value)) })} />
        </label>
        {settings.representation !== "fractions" && (
          <>
            <label>
              <span>変換座標の下限</span>
              <input type="number" step="any" value={settings.coordinateLower} onChange={(event) => patch({ coordinateLower: Number(event.target.value) })} />
            </label>
            <label>
              <span>変換座標の上限</span>
              <input type="number" step="any" value={settings.coordinateUpper} onChange={(event) => patch({ coordinateUpper: Number(event.target.value) })} />
            </label>
          </>
        )}
      </div>

      {!validElements && <p className="settings-note warning-text">候補元素を2種類以上指定してください。</p>}
      {!validCoordinateBounds && <p className="settings-note warning-text">変換座標は下限より上限を大きくしてください。</p>}

      {settings.elements.length > 0 && (
        <section className="composition-element-section">
          <div className="constraint-section-heading">
            <div>
              <h4>元素ごとの比率制約</h4>
              <p>上下限と刻みは、合計1に正規化した比率で指定します。</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="composition-element-table">
              <thead><tr><th>元素</th><th>下限</th><th>上限</th><th>刻み</th><th>必須</th></tr></thead>
              <tbody>
                {settings.elements.map((element) => {
                  const pair = settings.bounds[element] ?? [0, 1];
                  return (
                    <tr key={element}>
                      <td><strong>{element}</strong></td>
                      <td><input type="number" min={0} max={1} step="any" value={pair[0]} onChange={(event) => patchElement(element, "lower", event.target.value)} /></td>
                      <td><input type="number" min={0} max={1} step="any" value={pair[1]} onChange={(event) => patchElement(element, "upper", event.target.value)} /></td>
                      <td><input type="number" min={0} max={1} step="any" value={settings.steps[element] ?? ""} placeholder="任意" onChange={(event) => patchElement(element, "step", event.target.value)} /></td>
                      <td><input className="table-checkbox" type="checkbox" checked={settings.requiredComponents.includes(element)} onChange={() => toggleRequired(element)} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="composition-element-section">
        <div className="constraint-section-heading">
          <div>
            <h4>元素間の線形制約</h4>
            <p>例: Sr − 0.5 × La = 0 とすると、SrをLaの半分に固定できます。</p>
          </div>
          <button type="button" className="secondary" disabled={!settings.elements.length} onClick={addConstraint}>制約を追加</button>
        </div>
        {settings.constraints.length === 0 ? (
          <div className="constraint-empty">元素間制約は設定されていません。</div>
        ) : (
          <div className="constraint-list">
            {settings.constraints.map((constraint, constraintIndex) => (
              <div className="constraint-card" key={constraint.id}>
                <div className="constraint-card-heading">
                  <div><span className="constraint-index">{constraintIndex + 1}</span><strong>Σ（係数 × 元素比）</strong></div>
                  <button type="button" className="constraint-delete" onClick={() => patch({ constraints: settings.constraints.filter((item) => item.id !== constraint.id) })}>×</button>
                </div>
                <div className="composition-term-list">
                  {constraint.terms.map((term, termIndex) => (
                    <div className="composition-term-row" key={`${constraint.id}-${termIndex}`}>
                      <input type="number" step="any" value={term.coefficient} aria-label={`制約${constraintIndex + 1}項${termIndex + 1}の係数`} onChange={(event) => patchTerm(constraint.id, termIndex, { coefficient: Number(event.target.value) })} />
                      <span>×</span>
                      <select value={term.element} onChange={(event) => patchTerm(constraint.id, termIndex, { element: event.target.value })}>
                        {settings.elements.map((element) => <option key={element} value={element}>{element}</option>)}
                      </select>
                      <button type="button" className="secondary compact" disabled={constraint.terms.length <= 1} onClick={() => patchConstraint(constraint.id, { terms: constraint.terms.filter((_item, index) => index !== termIndex) })}>削除</button>
                    </div>
                  ))}
                  <button type="button" className="secondary compact" onClick={() => patchConstraint(constraint.id, { terms: [...constraint.terms, { element: settings.elements[0] ?? "", coefficient: 1 }] })}>項を追加</button>
                </div>
                <div className="composition-constraint-relation">
                  <select value={constraint.operator} onChange={(event) => patchConstraint(constraint.id, { operator: event.target.value as ConstraintOperator })}>
                    <option value="=">=</option><option value="<=">≤</option><option value=">=">≥</option>
                  </select>
                  <input type="number" step="any" value={constraint.rhs} onChange={(event) => patchConstraint(constraint.id, { rhs: Number(event.target.value) })} />
                  <select value={constraint.basis} onChange={(event) => patchConstraint(constraint.id, { basis: event.target.value as ConstraintBasis })}>
                    <option value="atomic_amount">原子比・mol比基準</option>
                    <option value="weight_amount">重量基準</option>
                  </select>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <p className="settings-note">A/Bサイト分離、複数組成式列、組成記述子の候補生成は今回の対象外です。</p>
    </article>
  );
}

function installOptimizePanel(): void {
  const constraints = document.querySelector<HTMLElement>(".feature-constraint-panel");
  if (!constraints) {
    if (optimizeHost && !optimizeHost.isConnected) {
      optimizeRoot?.unmount();
      optimizeRoot = null;
      optimizeHost = null;
    }
    return;
  }
  if (optimizeHost?.isConnected) return;
  optimizeHost = document.createElement("div");
  optimizeHost.className = "composition-settings-host";
  constraints.parentElement?.insertBefore(optimizeHost, constraints);
  optimizeRoot = createRoot(optimizeHost);
  optimizeRoot.render(<CompositionPanel />);
}

function synchronizeExtension(): void {
  installPrepareControls();
  installOptimizePanel();
}

export function installCompositionExtension(): void {
  if (installed) return;
  installed = true;
  installFetchAdapter();
  const observer = new MutationObserver(synchronizeExtension);
  observer.observe(document.documentElement, { subtree: true, childList: true });
  window.addEventListener(CHANGE_EVENT, synchronizeExtension);
  queueMicrotask(synchronizeExtension);
}
