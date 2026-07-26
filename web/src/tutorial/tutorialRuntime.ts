import { useSyncExternalStore } from "react";
import type { TutorialKind } from "./tutorialStorage";

let activeTutorialKind: TutorialKind | null = null;
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot(): TutorialKind | null {
  return activeTutorialKind;
}

export function setActiveTutorialKind(kind: TutorialKind | null): void {
  if (activeTutorialKind === kind) return;
  activeTutorialKind = kind;
  listeners.forEach((listener) => listener());
}

export function useActiveTutorialKind(): TutorialKind | null {
  return useSyncExternalStore(subscribe, getSnapshot, () => null);
}

export function useSampleTutorialActive(): boolean {
  return useActiveTutorialKind() === "sample";
}
