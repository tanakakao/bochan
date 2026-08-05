const OBSERVER_OPTIONS: MutationObserverInit = { subtree: true, childList: true };
const RESULTS_ANCHOR_SELECTOR = [
  "article.recommended-first",
  ".interactive-visualization-section",
  ".feature-importance-panel"
].join(", ");

let installed = false;
let renderScheduled = false;
let resultsObserver: MutationObserver | null = null;

function panelByHeading(title: string): HTMLElement | null {
  return Array.from(document.querySelectorAll<HTMLElement>("article.panel"))
    .find((panel) => panel.querySelector("h3")?.textContent?.trim() === title) ?? null;
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
  const interactive = document.querySelector<HTMLElement>(".interactive-visualization-section");

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

  const importancePanel = document.querySelector<HTMLElement>(".feature-importance-panel");
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

function nodeContainsResultsAnchor(node: Node): boolean {
  if (!(node instanceof Element)) return false;
  return node.matches(RESULTS_ANCHOR_SELECTOR)
    || Boolean(node.querySelector(RESULTS_ANCHOR_SELECTOR));
}

function observeResultsMutations(): void {
  resultsObserver?.observe(document.documentElement, OBSERVER_OPTIONS);
}

function runSynchronization(): void {
  resultsObserver?.disconnect();
  try {
    synchronizeResultsLayout();
  } finally {
    observeResultsMutations();
  }
}

function scheduleSynchronization(): void {
  if (renderScheduled) return;
  renderScheduled = true;
  queueMicrotask(() => {
    renderScheduled = false;
    runSynchronization();
  });
}

/** Installs layout handling only for the Results page. */
export function installResultsLayoutExtension(): void {
  if (installed) return;
  installed = true;
  resultsObserver = new MutationObserver((records) => {
    const changed = records.some((record) => (
      [...Array.from(record.addedNodes), ...Array.from(record.removedNodes)]
        .some(nodeContainsResultsAnchor)
    ));
    if (changed) scheduleSynchronization();
  });
  observeResultsMutations();
  scheduleSynchronization();
}
