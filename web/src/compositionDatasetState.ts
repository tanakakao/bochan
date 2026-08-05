const COMPOSITION_SETTINGS_KEY = "bochan-web-composition-settings";
const ACTIVE_DATASET_KEY = "bochan-web-composition-dataset-id";
const COMPOSITION_CHANGE_EVENT = "bochan-composition-settings-change";

interface DatasetPayload {
  dataset_id?: string;
}

let installed = false;
let originalFetch: typeof window.fetch | null = null;
let visibilityScheduled = false;

function compositionIsSelected(): boolean {
  try {
    const raw = window.localStorage.getItem(COMPOSITION_SETTINGS_KEY);
    if (!raw) return false;
    const settings = JSON.parse(raw) as Record<string, unknown>;
    return Boolean(settings.enabled && String(settings.column ?? "").trim());
  } catch {
    return false;
  }
}

function resetCompositionSelection(): void {
  window.localStorage.setItem(
    COMPOSITION_SETTINGS_KEY,
    JSON.stringify({ enabled: false, column: "", elements: [] })
  );
  window.dispatchEvent(new CustomEvent(COMPOSITION_CHANGE_EVENT));
}

function synchronizeCompositionPanelVisibility(): void {
  const visible = compositionIsSelected();
  document
    .querySelectorAll<HTMLElement>(
      ".composition-model-settings-host, .composition-constraint-settings-host"
    )
    .forEach((host) => {
      host.hidden = !visible;
      host.setAttribute("aria-hidden", String(!visible));
    });
}

function scheduleVisibilitySynchronization(): void {
  if (visibilityScheduled) return;
  visibilityScheduled = true;
  queueMicrotask(() => {
    visibilityScheduled = false;
    synchronizeCompositionPanelVisibility();
  });
}

function isDatasetResponse(url: URL, method: string): boolean {
  if (method === "POST" && url.pathname.endsWith("/datasets")) return true;
  return method === "GET" && /\/datasets\/[^/]+$/.test(url.pathname);
}

function activateDataset(payload: DatasetPayload): void {
  const datasetId = String(payload.dataset_id ?? "").trim();
  if (!datasetId) return;
  const previousDatasetId = window.localStorage.getItem(ACTIVE_DATASET_KEY);
  if (previousDatasetId === datasetId) return;

  window.localStorage.setItem(ACTIVE_DATASET_KEY, datasetId);
  resetCompositionSelection();
}

function installDatasetFetchGuard(): void {
  if (originalFetch) return;
  originalFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const response = await originalFetch!(input, init);
    const rawUrl = typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
    const url = new URL(rawUrl, window.location.href);
    const method = String(init?.method ?? "GET").toUpperCase();

    if (response.ok && isDatasetResponse(url, method)) {
      try {
        activateDataset(await response.clone().json() as DatasetPayload);
      } catch {
        // A malformed dataset response is handled by the normal API error path.
      }
    }
    return response;
  };
}

export function installCompositionDatasetState(): void {
  if (installed) return;
  installed = true;
  installDatasetFetchGuard();

  const observer = new MutationObserver(scheduleVisibilitySynchronization);
  observer.observe(document.documentElement, { subtree: true, childList: true });
  window.addEventListener(COMPOSITION_CHANGE_EVENT, scheduleVisibilitySynchronization);
  scheduleVisibilitySynchronization();
}
