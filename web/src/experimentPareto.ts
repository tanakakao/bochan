import type { Data } from "plotly.js";
import type {
  ExperimentCycle,
  ExperimentHistoryResponse
} from "./experimentHistory";

interface ParetoRelationPoint {
  x: number;
  y: number;
  displayX: unknown;
  displayY: unknown;
  cycle: ExperimentCycle;
  row: Record<string, unknown>;
  rowIndex: number;
}

type HoverFormatter = (
  cycle: ExperimentCycle,
  row: Record<string, unknown>,
  rowIndex: number
) => string;

function targetSetting(history: ExperimentHistoryResponse | null, target: string) {
  if (!history) return undefined;
  for (const cycle of [...history.cycles].reverse()) {
    const setting = cycle.target_settings.find((value) => value.target === target);
    if (setting) return setting;
  }
  return undefined;
}

export function historyTargetTaskType(
  history: ExperimentHistoryResponse | null,
  target: string
): string {
  if (!history) return "regression";
  for (const cycle of [...history.cycles].reverse()) {
    const setting = cycle.target_settings.find((value) => value.target === target);
    const taskType = setting?.task_type ?? cycle.target_summary[target]?.task_type;
    if (taskType) return String(taskType);
  }
  return "regression";
}

function targetDirection(history: ExperimentHistoryResponse | null, target: string): string {
  const setting = targetSetting(history, target);
  if (setting?.direction) return setting.direction;
  if (!history) return "maximize";
  for (const cycle of [...history.cycles].reverse()) {
    const direction = cycle.target_summary[target]?.direction;
    if (direction) return direction;
  }
  return "maximize";
}

export function historyTargetClassOrder(
  history: ExperimentHistoryResponse | null,
  target: string
): Array<string | number> {
  const configured = targetSetting(history, target)?.class_order ?? [];
  if (configured.length) return [...configured];
  if (!history) return [];

  const values: Array<string | number> = [];
  for (const cycle of history.cycles) {
    for (const row of cycle.rows) {
      const value = row[target];
      if (typeof value !== "string" && typeof value !== "number") continue;
      const exists = values.some((current) => (
        typeof current === typeof value && String(current) === String(value)
      ));
      if (!exists) values.push(value);
    }
  }
  return values;
}

export function isParetoContinuousTask(task: string): boolean {
  return task === "regression" || task === "ordinal";
}

function paretoValue(
  history: ExperimentHistoryResponse | null,
  target: string,
  value: unknown
): number | null {
  const task = historyTargetTaskType(history, target);
  if (task === "regression") {
    const converted = typeof value === "number" ? value : Number(value);
    return Number.isFinite(converted) ? converted : null;
  }
  if (task !== "ordinal") return null;

  const order = historyTargetClassOrder(history, target);
  const index = order.findIndex((candidate) => (
    typeof candidate === typeof value && String(candidate) === String(value)
  ));
  if (index >= 0) return index;

  const converted = typeof value === "number" ? value : Number(value);
  return Number.isFinite(converted) ? converted : null;
}

function paretoRelationPoints(
  history: ExperimentHistoryResponse | null,
  xTarget: string,
  yTarget: string
): ParetoRelationPoint[] {
  if (!history || !xTarget || !yTarget || xTarget === yTarget) return [];

  const xTask = historyTargetTaskType(history, xTarget);
  const yTask = historyTargetTaskType(history, yTarget);
  if (!isParetoContinuousTask(xTask) || !isParetoContinuousTask(yTask)) return [];
  if (
    targetSetting(history, xTarget)?.optimize === false ||
    targetSetting(history, yTarget)?.optimize === false
  ) {
    return [];
  }

  return history.cycles.flatMap((cycle) => cycle.rows.flatMap((row, rowIndex) => {
    const x = paretoValue(history, xTarget, row[xTarget]);
    const y = paretoValue(history, yTarget, row[yTarget]);
    return x === null || y === null ? [] : [{
      x,
      y,
      displayX: row[xTarget],
      displayY: row[yTarget],
      cycle,
      row,
      rowIndex
    }];
  }));
}

function dominates(
  left: ParetoRelationPoint,
  right: ParetoRelationPoint,
  xDirection: string,
  yDirection: string
): boolean {
  const xBetterOrEqual = xDirection === "minimize" ? left.x <= right.x : left.x >= right.x;
  const yBetterOrEqual = yDirection === "minimize" ? left.y <= right.y : left.y >= right.y;
  const xStrict = xDirection === "minimize" ? left.x < right.x : left.x > right.x;
  const yStrict = yDirection === "minimize" ? left.y < right.y : left.y > right.y;
  return xBetterOrEqual && yBetterOrEqual && (xStrict || yStrict);
}

function cumulativeParetoFront(
  points: ParetoRelationPoint[],
  xDirection: string,
  yDirection: string
): ParetoRelationPoint[] {
  const front = points.filter((point, index) => !points.some((candidate, candidateIndex) => (
    index !== candidateIndex && dominates(candidate, point, xDirection, yDirection)
  )));
  const unique = new Map<string, ParetoRelationPoint>();
  for (const point of front) {
    unique.set(`${point.x}:${point.y}`, point);
  }
  return [...unique.values()].sort((left, right) => left.x - right.x);
}

export function historyParetoFrontTraces(
  history: ExperimentHistoryResponse | null,
  xTarget: string,
  yTarget: string,
  hoverFormatter?: HoverFormatter
): Data[] {
  const points = paretoRelationPoints(history, xTarget, yTarget);
  if (!points.length) return [];

  const front = cumulativeParetoFront(
    points,
    targetDirection(history, xTarget),
    targetDirection(history, yTarget)
  );
  if (!front.length) return [];

  return [{
    type: "scatter",
    mode: "lines+markers",
    name: "累積Pareto front",
    x: front.map((point) => point.displayX) as any,
    y: front.map((point) => point.displayY) as any,
    text: front.map((point) => hoverFormatter
      ? hoverFormatter(point.cycle, point.row, point.rowIndex)
      : `Cycle ${point.cycle.cycle_number} · data ${point.rowIndex + 1}`),
    marker: { size: 10, symbol: "diamond-open" },
    line: { dash: "dash", width: 2 },
    hovertemplate: "%{text}<extra></extra>"
  } as Data];
}
