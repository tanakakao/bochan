import { loadCompositionSettings } from "./compositionExtension";

const COMPOSITION_CHANGE_EVENT = "bochan-composition-settings-change";
const PROXY_KEY = "workflowProxyKey";
const OBSERVER_OPTIONS: MutationObserverInit = { subtree: true, childList: true };
const LAYOUT_ANCHOR_SELECTOR = [
  ".model-primary-grid",
  ".feature-missing-panel",
  ".feature-constraint-panel",
  ".composition-model-settings-host",
  ".composition-constraint-settings-host",
  "article.recommended-first",
  ".interactive-visualization-section",
  ".feature-importance-panel",
  "article.panel"
].join(", ");

let installed = false;
let renderScheduled = false;
let layoutObserver: MutationObserver | null = null;

function panelByHeading(title: string): HTMLElement | null {
  return Array.from(document.querySelectorAll<HTMLElement>("article.panel"))
    .find((panel) => panel.querySelector("h3")?.textContent?.trim() === title) ?? null;
}

function placeAfter(node: HTMLElement, reference: Element | null): void {
  if (!reference?.parentElement) return;
  if (reference.nextElementSibling === node) return;
  reference.insertAdjacentElement("afterend", node);
}

function synchronizeSettingsLayout(): void {
  const modelGrid = document.querySelector<HTMLElement>(".model-primary-grid");
  const preprocessingPanel = panelByHeading("説明変数の前処理");
  const missingPanel = document.querySelector<HTMLElement>(".feature-missing-panel");
  const accuracyPanel = panelByHeading("精度評価");
  const importancePanel = panelByHeading("特徴量重要度");
  const compositionHost = document.querySelector<HTMLElement>(
    ".composition-model-settings-host"
  );

  if (preprocessingPanel) {
    preprocessingPanel.classList.add("feature-preprocessing-panel");
    const preprocessingGrid = preprocessingPanel.querySelector<HTMLElement>(
      ".search-transform-grid"
    );
    if (missingPanel && preprocessingGrid && missingPanel.parentElement !== preprocessingGrid) {
      preprocessingGrid.appendChild(missingPanel);
    }
    const missingKicker = missingPanel?.querySelector<HTMLElement>(".panel-kicker");
    if (missingKicker && missingKicker.textContent !== "MISSING VALUES") {
      missingKicker.textContent = "MISSING VALUES";
    }
  }

  if (
    modelGrid
    && preprocessingPanel
    && preprocessingPanel.parentElement !== modelGrid
  ) {
    modelGrid.appendChild(preprocessingPanel);
  }

  const composition = loadCompositionSettings();
  const compositionEnabled = Boolean(composition.enabled && composition.column);
  if (compositionHost) {
    compositionHost.hidden = !compositionEnabled;
  }

  const accuracyAnchor = compositionHost ?? modelGrid;
  if (accuracyPanel) placeAfter(accuracyPanel, accuracyAnchor);
  if (importancePanel) placeAfter(importancePanel, accuracyPanel);
}

function markProxyControls(source: HTMLElement, scope: string): void {
  source.querySelectorAll<HTMLElement>("input, select, button").forEach((control, index) => {
    control.dataset[PROXY_KEY] = `${scope}-${index}`;
  });
}

function cloneProxySource(source: HTMLElement): HTMLElement {
  const clone = source.cloneNode(true) as HTMLElement;
  clone.querySelectorAll<HTMLElement>("[id]").forEach((element) => {
    element.dataset.workflowSourceId = element.id;
    element.removeAttribute("id");
  });
  return clone;
}

function sourceControl(key: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(
    `.composition-constraint-settings-source [data-workflow-proxy-key="${key}"]`
  );
}

function bindCompositionProxy(proxy: HTMLElement): void {
  proxy.addEventListener("click", (event) => {
    const target = (event.target as Element).closest<HTMLElement>(
      "button[data-workflow-proxy-key]"
    );
    if (!target || !proxy.contains(target)) return;
    const key = target.dataset[PROXY_KEY];
    const source = key ? sourceControl(key) : null;
    if (!(source instanceof HTMLButtonElement)) return;
    event.preventDefault();
    source.click();
  });

  proxy.addEventListener("change", (event) => {
    const target = (event.target as Element).closest<HTMLElement>(
      "input[data-workflow-proxy-key], select[data-workflow-proxy-key]"
    );
    if (!target || !proxy.contains(target)) return;
    const key = target.dataset[PROXY_KEY];
    const source = key ? sourceControl(key) : null;

    if (target instanceof HTMLInputElement && source instanceof HTMLInputElement) {
      source.value = target.value;
      source.checked = target.checked;
      source.dispatchEvent(new Event("change", { bubbles: true }));
    } else if (target instanceof HTMLSelectElement && source instanceof HTMLSelectElement) {
      source.value = target.value;
      source.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
}

function replaceProxy(
  parent: HTMLElement,
  className: string,
  signature: string,
  build: () => HTMLElement,
  before: Element | null = null
): void {
  const existing = parent.querySelector<HTMLElement>(`:scope > .${className}`);
  if (existing?.dataset.sourceSignature === signature) return;
  const replacement = build();
  replacement.dataset.sourceSignature = signature;
  if (existing) {
    existing.replaceWith(replacement);
  } else if (before) {
    parent.insertBefore(replacement, before);
  } else {
    parent.appendChild(replacement);
  }
}

function clearCompositionProxies(): void {
  document.querySelector(".composition-search-space-constraints-proxy")?.remove();
  document.querySelector(".composition-linear-constraints-proxy")?.remove();
}

function synchronizeCompositionConstraintLayout(): void {
  const composition = loadCompositionSettings();
  const enabled = Boolean(composition.enabled && composition.column);
  const sourceHost = document.querySelector<HTMLElement>(
    ".composition-constraint-settings-host"
  );
  const searchSpacePanel = panelByHeading("説明変数の探索範囲");
  const featureConstraintPanel = document.querySelector<HTMLElement>(
    ".feature-constraint-panel"
  );

  if (!enabled || !sourceHost || !searchSpacePanel || !featureConstraintPanel) {
    clearCompositionProxies();
    return;
  }

  sourceHost.classList.add("composition-constraint-settings-source");
  sourceHost.hidden = true;
  const sourcePanel = sourceHost.querySelector<HTMLElement>(
    ".composition-constraint-settings-panel"
  );
  const countGrid = sourcePanel?.querySelector<HTMLElement>(".composition-basic-grid");
  const sections = sourcePanel
    ? Array.from(sourcePanel.querySelectorAll<HTMLElement>(
        ":scope > .composition-element-section"
      ))
    : [];
  const ratioSection = sections[0];
  const linearSection = sections[1];
  if (!countGrid || !ratioSection || !linearSection) return;

  markProxyControls(countGrid, "composition-count");
  markProxyControls(ratioSection, "composition-ratio");
  markProxyControls(linearSection, "composition-linear");

  const searchSignature = `${countGrid.outerHTML}\n${ratioSection.outerHTML}`;
  replaceProxy(
    searchSpacePanel,
    "composition-search-space-constraints-proxy",
    searchSignature,
    () => {
      const proxy = document.createElement("section");
      proxy.className = "composition-search-space-constraints-proxy";
      proxy.innerHTML = `
        <div class="composition-inline-heading">
          <div>
            <span class="panel-kicker">COMPOSITION SEARCH SPACE</span>
            <h4>組成候補の元素制約</h4>
            <p>説明変数の探索範囲と同じ場所で、使用元素数と元素ごとの比率範囲を設定します。</p>
          </div>
          <span class="status-chip success">${composition.elements.length} elements</span>
        </div>`;

      const countCard = document.createElement("section");
      countCard.className = "constraint-section composition-count-constraints-proxy";
      countCard.innerHTML = `
        <div class="constraint-section-heading">
          <div>
            <h4>元素数の制約</h4>
            <p>候補組成に含める元素種類数の最小値と最大値を指定します。</p>
          </div>
        </div>`;
      const countClone = cloneProxySource(countGrid);
      countClone.classList.add("composition-count-grid");
      countCard.appendChild(countClone);
      proxy.appendChild(countCard);

      const ratioClone = cloneProxySource(ratioSection);
      ratioClone.classList.add(
        "constraint-section",
        "composition-ratio-constraints-proxy"
      );
      proxy.appendChild(ratioClone);
      bindCompositionProxy(proxy);
      return proxy;
    }
  );

  const linearSignature = linearSection.outerHTML;
  const finalNote = Array.from(
    featureConstraintPanel.querySelectorAll<HTMLElement>(":scope > p.settings-note")
  ).at(-1) ?? null;
  replaceProxy(
    featureConstraintPanel,
    "composition-linear-constraints-proxy",
    linearSignature,
    () => {
      const proxy = cloneProxySource(linearSection);
      proxy.classList.add(
        "constraint-section",
        "composition-linear-constraints-proxy"
      );
      bindCompositionProxy(proxy);
      return proxy;
    },
    finalNote
  );
}

function resultPlaceholder(kind: "accuracy" | "importance"): HTMLElement {
  const article = document.createElement("article");
  article.className = `panel compact-panel results-${kind}-placeholder results-${kind}-slot`;
  article.innerHTML = kind === "accuracy"
    ? `<div class="panel-title"><div><span class="panel-kicker">ACCURACY</span><h3>精度評価の結果</h3><p>交差検証を有効にして再実行すると、Train・Validation・OOF指標を表示します。</p></div><span class="status-chip">Not evaluated</span></div>`
    : `<div class="panel-title"><div><span class="panel-kicker">MODEL INSPECTION</span><h3>特徴量重要度</h3><p>モデル設定で特徴量重要度を有効にして再実行すると表示します。</p></div><span class="status-chip">Not calculated</span></div>`;
  return article;
}

function synchronizeResultsLayout(): void {
  const candidates = document.querySelector<HTMLElement>("article.recommended-first");
  const interactive = document.querySelector<HTMLElement>(
    ".interactive-visualization-section"
  );

  if (!candidates || !interactive) {
    document.querySelectorAll<HTMLElement>(".results-dashboard-layout").forEach((layout) => {
      layout.remove();
    });
    return;
  }

  let layout = candidates.closest<HTMLElement>(".results-dashboard-layout");
  const parent = layout?.parentElement ?? candidates.parentElement;
  if (!parent) return;
  if (!layout) {
    layout = document.createElement("div");
    layout.className = "results-dashboard-layout";
    parent.insertBefore(layout, candidates);
  }

  candidates.classList.add("results-candidates-panel");
  interactive.classList.add("results-interactive-section");
  const plotCards = interactive.querySelectorAll<HTMLElement>(".interactive-plot-card");
  plotCards[0]?.classList.add("results-yy-card");
  plotCards[1]?.classList.add("results-relationship-card");

  const accuracyPanel = panelByHeading("交差検証による精度評価");
  const oldAccuracyPlaceholder = layout.querySelector<HTMLElement>(
    ":scope > .results-accuracy-placeholder"
  );
  if (accuracyPanel) oldAccuracyPlaceholder?.remove();
  const accuracySlot = accuracyPanel
    ?? oldAccuracyPlaceholder
    ?? resultPlaceholder("accuracy");
  accuracySlot.classList.add("results-accuracy-slot");

  const importancePanel = document.querySelector<HTMLElement>(
    ".feature-importance-panel"
  );
  const oldImportancePlaceholder = layout.querySelector<HTMLElement>(
    ":scope > .results-importance-placeholder"
  );
  if (importancePanel) oldImportancePlaceholder?.remove();
  const importanceSlot = importancePanel
    ?? oldImportancePlaceholder
    ?? resultPlaceholder("importance");
  importanceSlot.classList.add("results-importance-slot");

  const desired = [candidates, interactive, accuracySlot, importanceSlot];
  const current = Array.from(layout.children);
  const alreadyOrdered = desired.length === current.length
    && desired.every((node, index) => current[index] === node);
  if (!alreadyOrdered) layout.append(...desired);
}

function nodeContainsLayoutAnchor(node: Node): boolean {
  if (!(node instanceof Element)) return false;
  return node.matches(LAYOUT_ANCHOR_SELECTOR)
    || Boolean(node.querySelector(LAYOUT_ANCHOR_SELECTOR));
}

function mutationAffectsLayout(record: MutationRecord): boolean {
  return [...Array.from(record.addedNodes), ...Array.from(record.removedNodes)]
    .some(nodeContainsLayoutAnchor);
}

function observeLayoutMutations(): void {
  layoutObserver?.observe(document.documentElement, OBSERVER_OPTIONS);
}

function runSynchronization(): void {
  layoutObserver?.disconnect();
  try {
    synchronizeSettingsLayout();
    synchronizeCompositionConstraintLayout();
    synchronizeResultsLayout();
  } finally {
    observeLayoutMutations();
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

export function installWorkflowLayoutExtension(): void {
  if (installed) return;
  installed = true;
  layoutObserver = new MutationObserver((records) => {
    if (records.some(mutationAffectsLayout)) synchronize();
  });
  observeLayoutMutations();
  window.addEventListener(COMPOSITION_CHANGE_EVENT, synchronize);
  window.addEventListener("hashchange", synchronize);
  synchronize();
}
