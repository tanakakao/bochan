const progressFallbackStarted = new WeakMap<HTMLElement, number>();
let progressFallbackTimer: number | null = null;

function numericProgress(panel: HTMLElement): number {
  const bar = panel.querySelector<HTMLElement>(".execution-progress-track > span");
  const width = Number.parseFloat(bar?.style.width ?? "");
  return Number.isFinite(width) ? width : 0;
}

function estimatedProgress(elapsedMs: number): { percent: number; stage: number; label: string } {
  const elapsedSeconds = elapsedMs / 1000;
  if (elapsedSeconds < 2) {
    return { percent: 10 + elapsedSeconds * 4, stage: 0, label: "データと探索条件を準備しています" };
  }
  if (elapsedSeconds < 20) {
    const ratio = (elapsedSeconds - 2) / 18;
    return { percent: 18 + ratio * 37, stage: 1, label: "モデル学習を実行しています" };
  }
  if (elapsedSeconds < 50) {
    const ratio = (elapsedSeconds - 20) / 30;
    return { percent: 55 + ratio * 25, stage: 2, label: "候補探索を実行しています" };
  }
  const ratio = Math.min(1, (elapsedSeconds - 50) / 60);
  return { percent: 80 + ratio * 10, stage: 3, label: "候補の予測・可視化を処理しています" };
}

function applyEstimatedProgress(panel: HTMLElement): void {
  const current = numericProgress(panel);
  if (current > 5.5 || panel.classList.contains("failed")) {
    panel.classList.remove("estimated-fallback");
    return;
  }

  const started = progressFallbackStarted.get(panel) ?? Date.now();
  progressFallbackStarted.set(panel, started);
  const estimate = estimatedProgress(Date.now() - started);
  const percent = Math.max(6, Math.min(90, estimate.percent));

  panel.classList.add("estimated-fallback");
  const label = panel.querySelector<HTMLElement>(".execution-progress-label");
  const percentLabel = panel.querySelector<HTMLElement>(".execution-progress-percent");
  const bar = panel.querySelector<HTMLElement>(".execution-progress-track > span");
  if (label) label.textContent = estimate.label;
  if (percentLabel) percentLabel.textContent = `約${Math.round(percent)}%`;
  if (bar) bar.style.width = `${percent}%`;

  panel.querySelectorAll<HTMLElement>(".execution-stage-list li").forEach((item) => {
    const stage = Number(item.dataset.stage ?? 0);
    item.classList.toggle("complete", stage < estimate.stage);
    item.classList.toggle("active", stage === estimate.stage);
    item.classList.remove("failed");
  });
}

function syncProgressFallback(): void {
  const panels = Array.from(
    document.querySelectorAll<HTMLElement>(".execution-progress-panel.workflow")
  );
  panels.forEach(applyEstimatedProgress);
  if (panels.length === 0 && progressFallbackTimer !== null) {
    window.clearInterval(progressFallbackTimer);
    progressFallbackTimer = null;
  }
}

function ensureProgressFallbackTimer(): void {
  if (progressFallbackTimer !== null) return;
  if (!document.querySelector(".execution-progress-panel.workflow")) return;
  progressFallbackTimer = window.setInterval(syncProgressFallback, 650);
  syncProgressFallback();
}

export function installUxCorrectionsRuntime(): void {
  const flaggedWindow = window as typeof window & { __bochanUxCorrectionsInstalled?: boolean };
  if (flaggedWindow.__bochanUxCorrectionsInstalled) return;
  flaggedWindow.__bochanUxCorrectionsInstalled = true;

  const observer = new MutationObserver(() => ensureProgressFallbackTimer());
  observer.observe(document.documentElement, { childList: true, subtree: true });
  ensureProgressFallbackTimer();
}
