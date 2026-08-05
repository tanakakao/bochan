import type { FeatureImportanceSummaryRecord, RegressionResult, ResultVisualization } from "./types";

type UnknownRecord = Record<string, unknown>;

function asRecord(value: unknown): UnknownRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as UnknownRecord
    : null;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String).filter(Boolean) : [];
}

function firstStringArray(...values: unknown[]): string[] {
  for (const value of values) {
    const resolved = stringArray(value);
    if (resolved.length) return resolved;
  }
  return [];
}

function elementNamesFromFeatures(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = asRecord(item);
    return typeof record?.element === "string" && record.element
      ? [record.element]
      : [];
  });
}

function elementNamesFromFractionFeatures(value: unknown): string[] {
  return stringArray(value).flatMap((feature) => {
    const match = feature.match(/__fraction__([^_]+)$/i);
    return match ? [match[1]] : [];
  });
}

function modelFeatureNames(result: RegressionResult): string[] {
  const internal = stringArray(result.metadata?.tabular_feature_names);
  return internal.length ? internal : result.feature_columns.map(String);
}

function genericFeatureIndex(value: string): number | null {
  const zeroBased = value.match(/^(?:feature|x)(?:[_\s-]|\[)?(\d+)\]?$/i);
  if (zeroBased) return Number(zeroBased[1]);
  const oneBased = value.match(/^(?:特徴量|説明変数)\s*[_-]?\s*(\d+)$/);
  return oneBased ? Math.max(Number(oneBased[1]) - 1, 0) : null;
}

function compositionInfo(result: RegressionResult): {
  column: string;
  elements: string[];
  coordinates: Set<string>;
} {
  const payload = asRecord((result as RegressionResult & {
    composition_feature_importance?: unknown;
  }).composition_feature_importance) ?? {};
  const options = asRecord(result.visualization_options?.composition) ?? {};
  const metadata = asRecord(result.metadata?.composition_feature_importance) ?? {};
  const elements = firstStringArray(
    payload.element_names,
    options.elements,
    metadata.element_names,
    elementNamesFromFeatures(options.features),
    elementNamesFromFractionFeatures(options.fraction_features)
  );
  const coordinates = new Set([
    ...stringArray(payload.coordinate_features),
    ...stringArray(metadata.coordinate_features)
  ]);
  const column = [payload.column, options.column, metadata.column]
    .find((value): value is string => typeof value === "string" && value.length > 0) ?? "";
  return { column, elements, coordinates };
}

function ilrContrastLabel(index: number, elements: string[]): string {
  if (index < 1 || index >= elements.length) return `組成 ILR ${index}`;
  const numerator = elements.slice(0, index);
  const denominator = elements[index];
  return `組成 ILR ${index}: GM(${numerator.join("・")}) / ${denominator}`;
}

function coordinateLabel(value: string, elements: string[]): string | null {
  const coordinate = value.match(/__(fraction|clr|alr)__([^_]+)$/i);
  if (coordinate) {
    const representation = coordinate[1].toUpperCase();
    const element = coordinate[2];
    return representation === "FRACTION"
      ? `${element} 比率`
      : `${element} 比率（${representation}座標）`;
  }
  const ilr = value.match(/__ilr__(\d+)$/i);
  if (!ilr) return null;
  const rawIndex = Number(ilr[1]);
  const displayIndex = rawIndex >= 1 ? rawIndex : rawIndex + 1;
  return ilrContrastLabel(displayIndex, elements);
}

function labelResolver(result: RegressionResult): {
  displayColumns: string[];
  resolve: (value: unknown) => string;
} {
  const internalNames = modelFeatureNames(result);
  const composition = compositionInfo(result);
  const configured = result.visualization_options?.feature_labels ?? {};
  const sourceColumns = new Set(result.feature_columns.map(String));

  function format(value: string): string {
    if (value === "組成全体") return value;
    if (configured[value]) return configured[value];
    if (composition.elements.includes(value)) return `${value} 比率`;
    const coordinate = coordinateLabel(value, composition.elements);
    if (coordinate) return coordinate;
    if (composition.coordinates.has(value)) {
      const suffix = composition.elements.length
        ? `（${composition.elements.join("・")}）`
        : "";
      return `組成座標${suffix}`;
    }
    if (composition.column && value === composition.column) return `${value}（組成式）`;
    if (sourceColumns.has(value)) return value;
    return value;
  }

  const displayColumns = internalNames.map(format);
  function resolve(value: unknown): string {
    const raw = String(value);
    const genericIndex = genericFeatureIndex(raw);
    if (genericIndex !== null && internalNames[genericIndex]) {
      return format(internalNames[genericIndex]);
    }
    const internalIndex = internalNames.indexOf(raw);
    return internalIndex >= 0 ? displayColumns[internalIndex] : format(raw);
  }
  return { displayColumns, resolve };
}

function relabelArray(value: unknown, resolve: (value: unknown) => string): unknown {
  return Array.isArray(value)
    ? value.map((item) => typeof item === "string" ? resolve(item) : item)
    : value;
}

function relabelFigure(
  visualization: ResultVisualization,
  resolve: (value: unknown) => string
): ResultVisualization {
  const data = visualization.figure.data.map((item) => {
    const trace = asRecord(item);
    if (!trace) return item;
    return {
      ...trace,
      name: typeof trace.name === "string" ? resolve(trace.name) : trace.name,
      x: relabelArray(trace.x, resolve),
      y: relabelArray(trace.y, resolve)
    };
  });
  const layout: Record<string, unknown> = { ...visualization.figure.layout };
  for (const axisName of ["xaxis", "yaxis"]) {
    const axis = asRecord(layout[axisName]);
    if (axis?.ticktext) {
      layout[axisName] = { ...axis, ticktext: relabelArray(axis.ticktext, resolve) };
    }
  }
  return { ...visualization, figure: { ...visualization.figure, data, layout } };
}

export function withImportanceFeatureLabels(result: RegressionResult): RegressionResult {
  const { displayColumns, resolve } = labelResolver(result);
  const summary: FeatureImportanceSummaryRecord[] | undefined =
    result.feature_importance_summary?.map((row) => ({
      ...row,
      feature: resolve(row.feature)
    }));
  const visualizations = result.feature_importance_visualizations?.map((figure) =>
    relabelFigure(figure, resolve)
  );
  return {
    ...result,
    feature_columns: displayColumns,
    feature_importance_summary: summary,
    feature_importance_visualizations: visualizations
  };
}
