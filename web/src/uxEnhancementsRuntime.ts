import { fetchLogs } from "./api";
import type { LogEntry } from "./types";

const CONTEXT_COLLAPSE_KEY = "bochan-context-rail-collapsed";

type DataEntryMode = "new" | "resume";

type ProgressStage = {
  label: string;
  detail: string;
};

const WORKFLOW_STAGES: ProgressStage[] = [
  { label: "データ準備", detail: "入力データと探索条件を準備" },
  { label: "モデル学習", detail: "代理モデルを構築・学習" },
  { label: "候補探索", detail: "獲得関数を最適化" },
  { label: "予測・可視化", detail: "候補の予測値と図を生成" },
  { label: "完了", detail: "Resultsへ反映" }
];

const EVENT_STAGE: Record<string, number> = {
  workflow_started: 0,
  workflow_data_prepared: 0,
  model_fit_started: 1,
  model_fit_completed: 1,
  candidate_generation_started: 2,
  candidate_generation_completed: 2,
  candidate_prediction_completed: 3,
  visualization_created: 3,
  visualization_failed: 3,
  workflow_completed: 4,
  regression_run_completed: 4
};

const EVENT_LABEL: Record<string, string> = {
  workflow_started: "実行を開始しました",
  workflow_data_prepared: "データ準備が完了しました",
  model_fit_started: "モデルを学習しています",
  model_fit_completed: "モデル学習が完了しました",
  candidate_generation_started: "候補を探索しています",
  candidate_generation_completed: "候補探索が完了しました",
  candidate_prediction_completed: "候補の予測値を計算しました",
  visualization_created: "可視化を生成しています",
  visualization_failed: "可視化の一部を生成できませんでした",
  workflow_completed: "ワークフローが完了しました",
  regression_run_completed: "候補提案が完了しました",
  regression_run_failed: "候補提案に失敗しました",
  model_fit_failed: "モデル学習に失敗しました",
  candidate_generation_failed: "候補探索に失敗しました"
};

const FAILURE_EVENTS = new Set([
  "regression_run_failed",
  "model_fit_failed",
  "candidate_generation_failed"
]);

let dataEntryMode: DataEntryMode = "new";
let activeOverlay: HTMLElement | null = null;
let progressPollTimer: number | null = null;
let progressStartedAt = 0;
let progressRequestId: string | undefined;
let syncFrame: number | null = null;

function textElement<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text: string
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function installDataEntrySwitcher(): void {
  const grid = document.querySelector<HTMLElement>(".data-source-grid");
  if (!grid) return;

  let switcher = grid.querySelector<HTMLElement>(":scope > .data-entry-switcher");
  if (!switcher) {
    switcher = document.createElement("section");
    switcher.className = "data-entry-switcher";
    switcher.setAttribute("aria-label", "Data画面の開始方法");

    const heading = document.createElement("div");
    heading.className = "data-entry-switcher-heading";
    heading.append(
      textElement("span", "panel-kicker", "START"),
      textElement("h3", "", "どの方法で始めますか？"),
      textElement("p", "", "新しいデータから解析するか、保存済みの作業を復元します。")
    );

    const choices = document.createElement("div");
    choices.className = "data-entry-choices";
    choices.append(
      createEntryChoice(
        "new",
        "＋",
        "新しい解析",
        "CSV / Excelから新しい最適化を開始"
      ),
      createEntryChoice(
        "resume",
        "↺",
        "作業を再開",
        "保存モデル / プロジェクトを読み込む"
      )
    );
    switcher.append(heading, choices);
    grid.prepend(switcher);

    const restored = Boolean(
      grid.querySelector(".model-artifact-panel .status-chip.success")
    );
    dataEntryMode = restored ? "resume" : "new";
  }

  applyDataEntryMode(grid, dataEntryMode);
}

function createEntryChoice(
  mode: DataEntryMode,
  icon: string,
  title: string,
  description: string
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "data-entry-choice secondary";
  button.dataset.entryTarget = mode;
  button.append(
    textElement("span", "data-entry-choice-icon", icon),
    (() => {
      const copy = document.createElement("span");
      copy.className = "data-entry-choice-copy";
      copy.append(
        textElement("strong", "", title),
        textElement("small", "", description)
      );
      return copy;
    })(),
    textElement("span", "data-entry-choice-arrow", "›")
  );
  button.addEventListener("click", () => {
    const grid = button.closest<HTMLElement>(".data-source-grid");
    if (!grid) return;
    dataEntryMode = mode;
    applyDataEntryMode(grid, mode);
  });
  return button;
}

function applyDataEntryMode(grid: HTMLElement, mode: DataEntryMode): void {
  if (grid.dataset.entryMode !== mode) grid.dataset.entryMode = mode;
  grid.querySelectorAll<HTMLButtonElement>(".data-entry-choice").forEach((button) => {
    const active = button.dataset.entryTarget === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function installContextRailToggle(): void {
  const rail = document.querySelector<HTMLElement>(".right-rail");
  const shell = rail?.closest<HTMLElement>(".app-shell");
  if (!rail || !shell) return;

  let button = rail.querySelector<HTMLButtonElement>(":scope > .context-rail-toggle");
  if (!button) {
    button = document.createElement("button");
    button.type = "button";
    button.className = "context-rail-toggle secondary";
    rail.prepend(button);
    button.addEventListener("click", () => {
      const collapsed = !rail.classList.contains("context-collapsed");
      applyContextRailState(shell, rail, button!, collapsed);
      window.localStorage.setItem(CONTEXT_COLLAPSE_KEY, collapsed ? "1" : "0");
    });
  }

  const collapsed = window.localStorage.getItem(CONTEXT_COLLAPSE_KEY) === "1";
  applyContextRailState(shell, rail, button, collapsed);
}

function applyContextRailState(
  shell: HTMLElement,
  rail: HTMLElement,
  button: HTMLButtonElement,
  collapsed: boolean
): void {
  shell.classList.toggle("context-rail-collapsed", collapsed);
  rail.classList.toggle("context-collapsed", collapsed);
  button.setAttribute("aria-expanded", String(!collapsed));
  button.setAttribute("aria-label", collapsed ? "右サイドバーを開く" : "右サイドバーを折り畳む");
  button.title = collapsed ? "設定サマリーを開く" : "設定サマリーを折り畳む";
  const icon = collapsed ? "‹" : "›";
  if (button.textContent !== icon) button.textContent = icon;
}

function syncBusyProgress(): void {
  const overlay = document.querySelector<HTMLElement>(".overlay");
  if (overlay === activeOverlay) return;
  stopProgressPolling();
  activeOverlay = overlay;
  if (!overlay) return;

  const heading = overlay.querySelector<HTMLElement>(".busy-card h3")?.textContent ?? "処理中";
  const workflowRun = /候補|モデル.*学習|学習済みモデル/.test(heading);
  const card = overlay.querySelector<HTMLElement>(".busy-card");
  if (!card) return;

  const panel = document.createElement("div");
  panel.className = `execution-progress-panel ${workflowRun ? "workflow" : "generic"}`;
  panel.setAttribute("aria-live", "polite");
  panel.innerHTML = workflowRun
    ? workflowProgressMarkup()
    : genericProgressMarkup(heading);
  card.insertBefore(panel, card.querySelector(".busy-state"));

  if (!workflowRun) return;
  progressStartedAt = Date.now();
  progressRequestId = undefined;
  void pollExecutionProgress(panel);
  progressPollTimer = window.setInterval(() => {
    if (!panel.isConnected) {
      stopProgressPolling();
      return;
    }
    void pollExecutionProgress(panel);
  }, 1200);
}

function workflowProgressMarkup(): string {
  return `
    <div class="execution-progress-head">
      <span>RUN PROGRESS</span>
      <strong class="execution-progress-label">実行準備中</strong>
      <em class="execution-progress-percent">5%</em>
    </div>
    <div class="execution-progress-track"><span style="width: 5%"></span></div>
    <ol class="execution-stage-list">
      ${WORKFLOW_STAGES.map((stage, index) => `
        <li data-stage="${index}">
          <span class="execution-stage-marker">${index + 1}</span>
          <span><strong>${stage.label}</strong><small>${stage.detail}</small></span>
        </li>
      `).join("")}
    </ol>
  `;
}

function genericProgressMarkup(heading: string): string {
  return `
    <div class="execution-progress-head">
      <span>PROCESS</span>
      <strong>${escapeHtml(heading)}</strong>
      <em>実行中</em>
    </div>
    <div class="execution-progress-track indeterminate"><span></span></div>
  `;
}

async function pollExecutionProgress(panel: HTMLElement): Promise<void> {
  try {
    const response = await fetchLogs({
      limit: 250,
      requestId: progressRequestId
    });
    const recent = response.entries.filter((entry) => {
      const timestamp = new Date(entry.timestamp).getTime();
      return Number.isFinite(timestamp) && timestamp >= progressStartedAt - 3000;
    });

    if (!progressRequestId) {
      progressRequestId = [...recent]
        .reverse()
        .find((entry) => Boolean(entry.request_id) && isWorkflowEvent(entry.event))
        ?.request_id;
    }

    const entries = progressRequestId
      ? response.entries.filter((entry) => entry.request_id === progressRequestId)
      : recent;
    renderExecutionProgress(panel, entries);
  } catch {
    // Progress is supplemental. Keep the current visual state if logs are temporarily unavailable.
  }
}

function isWorkflowEvent(event: string | undefined): boolean {
  if (!event) return false;
  return event in EVENT_STAGE || FAILURE_EVENTS.has(event);
}

function renderExecutionProgress(panel: HTMLElement, entries: LogEntry[]): void {
  const workflowEntries = entries.filter((entry) => isWorkflowEvent(entry.event));
  if (workflowEntries.length === 0) return;

  let stage = 0;
  let currentEvent = "workflow_started";
  let failed = false;
  let hasModelFit = false;

  workflowEntries.forEach((entry) => {
    const event = entry.event ?? "";
    if (event.startsWith("model_fit_")) hasModelFit = true;
    if (FAILURE_EVENTS.has(event)) failed = true;
    const mapped = EVENT_STAGE[event];
    if (mapped !== undefined && mapped >= stage) {
      stage = mapped;
      currentEvent = event;
    }
  });

  const percentages = [15, 42, 70, 90, 100];
  const percent = failed ? Math.max(percentages[stage] - 5, 10) : percentages[stage];
  const lastEvent = workflowEntries.at(-1)?.event ?? "";
  const label = failed
    ? EVENT_LABEL[lastEvent] ?? "処理に失敗しました"
    : EVENT_LABEL[currentEvent] ?? WORKFLOW_STAGES[stage].label;

  panel.classList.toggle("failed", failed);
  const labelNode = panel.querySelector<HTMLElement>(".execution-progress-label");
  const percentNode = panel.querySelector<HTMLElement>(".execution-progress-percent");
  if (labelNode) labelNode.textContent = label;
  if (percentNode) percentNode.textContent = failed ? "要確認" : `${percent}%`;
  const bar = panel.querySelector<HTMLElement>(".execution-progress-track > span");
  if (bar) bar.style.width = `${percent}%`;

  panel.querySelectorAll<HTMLElement>(".execution-stage-list li").forEach((item) => {
    const itemStage = Number(item.dataset.stage ?? 0);
    const skippedModel = itemStage === 1 && stage >= 2 && !hasModelFit;
    item.classList.toggle("complete", itemStage < stage && !skippedModel);
    item.classList.toggle("active", itemStage === stage && !failed);
    item.classList.toggle("failed", itemStage === stage && failed);
    item.classList.toggle("skipped", skippedModel);
    if (skippedModel) {
      const detail = item.querySelector("small");
      if (detail) detail.textContent = "学習済みモデルを再利用";
    }
  });
}

function stopProgressPolling(): void {
  if (progressPollTimer !== null) window.clearInterval(progressPollTimer);
  progressPollTimer = null;
  progressRequestId = undefined;
}

function installInlineValidation(): void {
  document.querySelectorAll<HTMLInputElement | HTMLSelectElement>("input, select").forEach((control) => {
    if (!control.closest(".search-variable-table")) validateControl(control);
  });
  validateSearchSpaceRows();
  validateSelectionPanels();
}

function validateControl(control: HTMLInputElement | HTMLSelectElement): void {
  if (control.disabled) {
    setFieldError(control, null);
    return;
  }

  if (control instanceof HTMLInputElement && control.type === "number") {
    const value = control.value.trim();
    if (!value) {
      setFieldError(control, control.required ? "値を入力してください。" : null);
      return;
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || control.validity.badInput) {
      setFieldError(control, "数値を入力してください。");
      return;
    }
    if (control.min !== "" && numeric < Number(control.min)) {
      setFieldError(control, `${control.min}以上の値を入力してください。`);
      return;
    }
    if (control.max !== "" && numeric > Number(control.max)) {
      setFieldError(control, `${control.max}以下の値を入力してください。`);
      return;
    }

    const label = control.closest("label")?.textContent ?? "";
    if (/標準偏差|ばらつき/.test(label) && numeric <= 0) {
      setFieldError(control, "0より大きい値を入力してください。");
      return;
    }
  }

  setFieldError(control, null);
}

function validateSearchSpaceRows(): void {
  document.querySelectorAll<HTMLTableRowElement>(".search-variable-table tbody tr").forEach((row) => {
    const cells = row.querySelectorAll<HTMLTableCellElement>("td");
    if (cells.length < 7) return;
    const lower = cells[2].querySelector<HTMLInputElement>('input[type="number"]');
    const upper = cells[3].querySelector<HTMLInputElement>('input[type="number"]');
    const step = cells[4].querySelector<HTMLInputElement>('input[type="number"]');
    const fixedToggle = cells[5].querySelector<HTMLInputElement>('input[type="checkbox"]');
    const fixedNumber = cells[6].querySelector<HTMLInputElement>('input[type="number"]');
    const fixedCategory = cells[6].querySelector<HTMLSelectElement>("select");

    const lowerValue = lower && lower.value !== "" ? Number(lower.value) : null;
    const upperValue = upper && upper.value !== "" ? Number(upper.value) : null;
    const invalidBounds = lowerValue !== null && upperValue !== null && lowerValue >= upperValue;
    if (lower) setFieldError(lower, invalidBounds ? "下限は上限より小さくしてください。" : null);
    if (upper) setFieldError(upper, invalidBounds ? "上限は下限より大きくしてください。" : null);

    if (step) {
      const invalidStep = step.value !== "" && Number(step.value) <= 0;
      setFieldError(step, invalidStep ? "刻みは0より大きくしてください。" : null);
    }

    if (fixedNumber) {
      let fixedError: string | null = null;
      if (fixedToggle?.checked) {
        const fixed = fixedNumber.value !== "" ? Number(fixedNumber.value) : null;
        if (fixed === null || !Number.isFinite(fixed)) {
          fixedError = "固定値を入力してください。";
        } else if (
          lowerValue !== null && upperValue !== null &&
          (fixed < lowerValue || fixed > upperValue)
        ) {
          fixedError = "固定値は探索範囲内にしてください。";
        }
      }
      setFieldError(fixedNumber, fixedError);
    }
    if (fixedCategory) {
      const fixedCategoryError = fixedToggle?.checked && fixedCategory.value === ""
        ? "固定するカテゴリを選択してください。"
        : null;
      setFieldError(fixedCategory, fixedCategoryError);
    }
  });
}

function validateSelectionPanels(): void {
  document.querySelectorAll<HTMLElement>(".selection-panel").forEach((panel) => {
    const heading = panel.querySelector("h3")?.textContent ?? "";
    const required = panel.querySelector(".status-chip.warning")?.textContent?.includes("Required") ?? false;
    let message = panel.querySelector<HTMLElement>(":scope > .panel-inline-validation");
    if (!required) {
      message?.remove();
      return;
    }
    if (!message) {
      message = document.createElement("div");
      message.className = "panel-inline-validation";
      panel.append(message);
    }
    const text = heading.includes("目的")
      ? "目的変数を1つ以上選択してください。"
      : "説明変数を1つ以上選択してください。";
    if (message.textContent !== text) message.textContent = text;
  });
}

function setFieldError(
  control: HTMLInputElement | HTMLSelectElement,
  message: string | null
): void {
  const parent = control.parentElement;
  if (!parent) return;
  let error = control.nextElementSibling instanceof HTMLElement &&
    control.nextElementSibling.classList.contains("field-inline-error")
    ? control.nextElementSibling
    : null;

  if (!message) {
    if (control.getAttribute("aria-invalid") === "true") control.removeAttribute("aria-invalid");
    control.classList.remove("input-invalid");
    error?.remove();
    return;
  }

  control.setAttribute("aria-invalid", "true");
  control.classList.add("input-invalid");
  if (!error) {
    error = document.createElement("small");
    error.className = "field-inline-error";
    control.insertAdjacentElement("afterend", error);
  }
  if (error.textContent !== message) error.textContent = message;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function syncUxEnhancements(): void {
  installDataEntrySwitcher();
  installContextRailToggle();
  syncBusyProgress();
  installInlineValidation();
}

function scheduleSync(): void {
  if (syncFrame !== null) return;
  syncFrame = window.requestAnimationFrame(() => {
    syncFrame = null;
    syncUxEnhancements();
  });
}

export function installUxEnhancementsRuntime(): void {
  const windowWithFlag = window as typeof window & { __bochanUxEnhancementsInstalled?: boolean };
  if (windowWithFlag.__bochanUxEnhancementsInstalled) return;
  windowWithFlag.__bochanUxEnhancementsInstalled = true;

  document.addEventListener("input", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) {
      if (!target.closest(".search-variable-table")) validateControl(target);
      validateSearchSpaceRows();
    }
  }, true);
  document.addEventListener("change", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) {
      if (!target.closest(".search-variable-table")) validateControl(target);
      validateSearchSpaceRows();
      scheduleSync();
    }
  }, true);
  document.addEventListener("blur", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement) {
      if (!target.closest(".search-variable-table")) validateControl(target);
      validateSearchSpaceRows();
    }
  }, true);

  const observer = new MutationObserver(scheduleSync);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  scheduleSync();
}
