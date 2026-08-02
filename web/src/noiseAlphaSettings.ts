export const DEFAULT_NOISE_ALPHA = 1e-4;

const STORAGE_KEY = "bochan-regression-noise-alpha";
const SUPPORTED_MODEL_TYPES = [
  "base",
  "deepgp",
  "deepkernel",
  "pca",
  "rembo",
  "robust"
] as const;

/** Return whether the Web model uses the configurable Gaussian noise floor. */
export function supportsNoiseAlpha(modelType: string): boolean {
  return (SUPPORTED_MODEL_TYPES as readonly string[]).includes(modelType);
}

/** Load the model-scale Gaussian observation-noise variance floor. */
export function loadNoiseAlpha(): number {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === null) return DEFAULT_NOISE_ALPHA;
  const parsed = Number(stored);
  return Number.isFinite(parsed) ? parsed : DEFAULT_NOISE_ALPHA;
}

/** Persist the model-scale Gaussian observation-noise variance floor. */
export function saveNoiseAlpha(alpha: number): void {
  window.localStorage.setItem(STORAGE_KEY, String(alpha));
}
