export const TUTORIAL_VERSION = 1;

const TUTORIAL_STORAGE_KEY = "bochan-web-tutorial";

export type TutorialKind = "overview" | "sample";
export type TutorialStatus = "in_progress" | "completed" | "dismissed";

export interface TutorialProgress {
  version: number;
  kind: TutorialKind;
  status: TutorialStatus;
  stepIndex: number;
  updatedAt: string;
}

function isTutorialKind(value: unknown): value is TutorialKind {
  return value === "overview" || value === "sample";
}

function isTutorialStatus(value: unknown): value is TutorialStatus {
  return value === "in_progress" || value === "completed" || value === "dismissed";
}

export function readTutorialProgress(): TutorialProgress | null {
  if (typeof window === "undefined") return null;

  try {
    const raw = window.localStorage.getItem(TUTORIAL_STORAGE_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw) as Partial<TutorialProgress>;
    if (
      typeof parsed.version !== "number" ||
      !isTutorialStatus(parsed.status) ||
      typeof parsed.stepIndex !== "number" ||
      !Number.isInteger(parsed.stepIndex) ||
      typeof parsed.updatedAt !== "string"
    ) {
      return null;
    }

    return {
      version: parsed.version,
      kind: isTutorialKind(parsed.kind) ? parsed.kind : "overview",
      status: parsed.status,
      stepIndex: Math.max(0, parsed.stepIndex),
      updatedAt: parsed.updatedAt
    };
  } catch {
    return null;
  }
}

export function shouldPromptTutorial(progress: TutorialProgress | null): boolean {
  if (!progress) return true;
  if (progress.version !== TUTORIAL_VERSION) return true;
  return progress.status === "in_progress";
}

export function writeTutorialProgress(
  status: TutorialStatus,
  stepIndex: number,
  kind: TutorialKind = "overview"
): TutorialProgress {
  const progress: TutorialProgress = {
    version: TUTORIAL_VERSION,
    kind,
    status,
    stepIndex: Math.max(0, Math.trunc(stepIndex)),
    updatedAt: new Date().toISOString()
  };

  if (typeof window === "undefined") return progress;

  try {
    window.localStorage.setItem(TUTORIAL_STORAGE_KEY, JSON.stringify(progress));
  } catch {
    // The tutorial remains usable even when browser storage is unavailable.
  }

  return progress;
}
