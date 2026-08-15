import { useState } from "react";
import type { DatasetResponse, SearchVariable, TargetSetting } from "../types";
import { createTargetSetting } from "./workbenchDefaults";

export interface WorkbenchSelectionSnapshot {
  dataset: DatasetResponse;
  featureColumns: string[];
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  variables: Record<string, SearchVariable>;
}

/** Owns the loaded dataset and the user's target/feature selections. */
export function useWorkbenchSelectionState() {
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [targetColumns, setTargetColumns] = useState<string[]>([]);
  const [targetSettings, setTargetSettings] = useState<Record<string, TargetSetting>>({});
  const [variables, setVariables] = useState<Record<string, SearchVariable>>({});

  const columns = dataset?.profile.columns ?? [];
  const preview = dataset?.preview ?? [];

  function replaceSelection(snapshot: WorkbenchSelectionSnapshot) {
    setDataset(snapshot.dataset);
    setFeatureColumns(snapshot.featureColumns);
    setTargetColumns(snapshot.targetColumns);
    setTargetSettings(snapshot.targetSettings);
    setVariables(snapshot.variables);
  }

  function toggleFeature(name: string) {
    if (targetColumns.includes(name)) return;
    setFeatureColumns((current) =>
      current.includes(name)
        ? current.filter((column) => column !== name)
        : [...current, name]
    );
  }

  function toggleTarget(name: string) {
    const profile = columns.find((column) => column.name === name);
    if (!profile) return;
    setTargetColumns((current) => {
      const selected = current.includes(name);
      const next = selected ? current.filter((column) => column !== name) : [...current, name];
      setFeatureColumns((features) => features.filter((column) => !next.includes(column)));
      setTargetSettings((settings) => {
        const updated = { ...settings };
        if (selected) delete updated[name];
        else updated[name] = updated[name] ?? createTargetSetting(profile, preview);
        return updated;
      });
      return next;
    });
  }

  function patchTargetSetting(target: string, patch: Partial<TargetSetting>) {
    setTargetSettings((current) => ({
      ...current,
      [target]: { ...current[target], ...patch, target }
    }));
  }

  function patchVariable(name: string, patch: Partial<SearchVariable>) {
    setVariables((current) => ({
      ...current,
      [name]: { ...current[name], ...patch }
    }));
  }

  return {
    dataset,
    featureColumns,
    targetColumns,
    targetSettings,
    variables,
    replaceSelection,
    toggleFeature,
    toggleTarget,
    patchTargetSetting,
    patchVariable
  };
}
