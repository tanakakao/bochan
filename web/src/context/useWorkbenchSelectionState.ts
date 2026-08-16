import { useRef, useState } from "react";
import type { DatasetResponse, SearchVariable, TargetSetting } from "../types";
import { createTargetSetting } from "./workbenchDefaults";

export interface WorkbenchSelectionSnapshot {
  dataset: DatasetResponse;
  featureColumns: string[];
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  variables: Record<string, SearchVariable>;
}

function cloneTargetSetting(setting: TargetSetting): TargetSetting {
  return {
    ...setting,
    target_classes: setting.target_classes ? [...setting.target_classes] : undefined,
    class_order: setting.class_order ? [...setting.class_order] : undefined,
    target_values: setting.target_values ? [...setting.target_values] : undefined
  };
}

function cloneVariable(variable: SearchVariable): SearchVariable {
  return {
    ...variable,
    categories: variable.categories ? [...variable.categories] : undefined
  };
}

/** Owns the loaded dataset and the user's target/feature selections. */
export function useWorkbenchSelectionState() {
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [targetColumns, setTargetColumns] = useState<string[]>([]);
  const [targetSettings, setTargetSettings] = useState<Record<string, TargetSetting>>({});
  const [variables, setVariables] = useState<Record<string, SearchVariable>>({});

  // Rendering still uses ordinary React state. Candidate-time controls also keep
  // an event-synchronous snapshot so execute() can read a just-edited threshold,
  // fixed value, step, or bound without depending on another render first. All
  // mutations of these values are owned by the helpers below, so the refs remain
  // authoritative until the matching React state update is committed.
  const datasetRef = useRef<DatasetResponse | null>(null);
  const targetSettingsRef = useRef<Record<string, TargetSetting>>({});
  const variablesRef = useRef<Record<string, SearchVariable>>({});

  const columns = dataset?.profile.columns ?? [];
  const preview = dataset?.preview ?? [];

  function replaceSelection(snapshot: WorkbenchSelectionSnapshot) {
    datasetRef.current = snapshot.dataset;
    targetSettingsRef.current = snapshot.targetSettings;
    variablesRef.current = snapshot.variables;
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
        targetSettingsRef.current = updated;
        return updated;
      });
      return next;
    });
  }

  function patchTargetSetting(target: string, patch: Partial<TargetSetting>) {
    const updated = {
      ...targetSettingsRef.current,
      [target]: { ...targetSettingsRef.current[target], ...patch, target }
    };
    targetSettingsRef.current = updated;
    setTargetSettings(updated);
  }

  function patchVariable(name: string, patch: Partial<SearchVariable>) {
    const updated = {
      ...variablesRef.current,
      [name]: { ...variablesRef.current[name], ...patch }
    };
    variablesRef.current = updated;
    setVariables(updated);
  }

  function getCurrentSelection(): WorkbenchSelectionSnapshot | null {
    const currentDataset = datasetRef.current;
    if (!currentDataset) return null;
    return {
      dataset: currentDataset,
      featureColumns: [...featureColumns],
      targetColumns: [...targetColumns],
      targetSettings: Object.fromEntries(
        Object.entries(targetSettingsRef.current).map(([name, setting]) => [
          name,
          cloneTargetSetting(setting)
        ])
      ),
      variables: Object.fromEntries(
        Object.entries(variablesRef.current).map(([name, variable]) => [
          name,
          cloneVariable(variable)
        ])
      )
    };
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
    patchVariable,
    getCurrentSelection
  };
}
