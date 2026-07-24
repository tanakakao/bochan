const RAW_API_BASE = String(import.meta.env.VITE_API_BASE ?? "/api/v1").trim();
const API_BASE = (RAW_API_BASE || "/api/v1").replace(/\/+$/, "");

function responseFilename(response: Response, fallback: string): string {
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) return decodeURIComponent(utf8Match[1]);
  const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
  return plainMatch?.[1] || fallback;
}

async function responseError(response: Response): Promise<Error> {
  const text = await response.text();
  let detail: unknown = text || `HTTP ${response.status}`;
  if (text) {
    try {
      const payload = JSON.parse(text);
      detail = payload?.detail ?? payload;
    } catch {
      detail = text;
    }
  }
  const message = typeof detail === "string" ? detail : JSON.stringify(detail);
  const requestId = response.headers.get("X-Request-ID");
  return new Error(requestId ? `${message} [request_id=${requestId}]` : message);
}

/** Download a Web model artifact using the filename selected in Results. */
export async function downloadNamedModelArtifact(
  runId: string,
  filename: string
): Promise<void> {
  const params = new URLSearchParams({ filename });
  const path = `/runs/${encodeURIComponent(runId)}/model-artifact?${params.toString()}`;
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw await responseError(response);

  const blob = await response.blob();
  const blobUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = blobUrl;
  anchor.download = responseFilename(response, filename);
  anchor.click();
  URL.revokeObjectURL(blobUrl);
}
