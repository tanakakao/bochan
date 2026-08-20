import type {
  CandidateRow,
  RegressionResult,
  VisualizationOptions
} from "./types";

type FeatureControl = NonNullable<VisualizationOptions["feature_controls"]>[string];

const CACHE = new WeakMap<RegressionResult, RegressionResult>();

function firstCandidate(result: RegressionResult): CandidateRow | undefined {
  const rankOne = result.candidates.find((candidate) => candidate.rank === 1);
  if (rankOne) return rankOne;
  return [...result.candidates].sort((left, right) => left.rank - right.rank)[0];
}

function candidateDefault(
  control: FeatureControl,
  value: string | number | undefined
): string | number {
  if (value === undefined || value === null || value === "") return control.default;
  if (control.kind === "numeric") {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : control.default;
  }
  const matched = (control.values ?? []).find(
    (candidate) => Object.is(candidate, value) || String(candidate) === String(value)
  );
  return matched ?? control.default;
}

/**
 * Return a stable result whose plot slice controls start from the top-ranked candidate.
 *
 * Candidate values are already repaired/decoded by the Web backend. Composition
 * fraction values are therefore handled the same way as ordinary numeric inputs
 * when they are present in the candidate row. Missing or invalid values retain the
 * existing training-data based defaults.
 */
export function withFirstCandidateVisualizationDefaults(
  result: RegressionResult
): RegressionResult {
  const cached = CACHE.get(result);
  if (cached) return cached;

  const options = result.visualization_options;
  const controls = options?.feature_controls;
  const candidate = firstCandidate(result);
  if (!options || !controls || !candidate) {
    CACHE.set(result, result);
    return result;
  }

  let changed = false;
  const nextControls = Object.fromEntries(
    Object.entries(controls).map(([feature, control]) => {
      const nextDefault = candidateDefault(control, candidate.values?.[feature]);
      if (control.kind !== "numeric") {
        if (!Object.is(nextDefault, control.default)) changed = true;
        return [feature, { ...control, default: nextDefault }];
      }

      const numericDefault = Number(nextDefault);
      if (!Number.isFinite(numericDefault)) return [feature, control];
      const nextMin = control.min === undefined
        ? numericDefault
        : Math.min(control.min, numericDefault);
      const nextMax = control.max === undefined
        ? numericDefault
        : Math.max(control.max, numericDefault);
      if (
        !Object.is(numericDefault, control.default)
        || nextMin !== control.min
        || nextMax !== control.max
      ) {
        changed = true;
      }
      return [feature, {
        ...control,
        default: numericDefault,
        min: nextMin,
        max: nextMax
      }];
    })
  );

  const updated = changed
    ? {
        ...result,
        visualization_options: {
          ...options,
          feature_controls: nextControls
        }
      }
    : result;
  CACHE.set(result, updated);
  return updated;
}

/** Stable React key so a newly generated candidate set resets plot control state. */
export function visualizationDefaultsKey(result: RegressionResult): string {
  const candidate = firstCandidate(result);
  const values = candidate
    ? Object.entries(candidate.values ?? {}).sort(([left], [right]) => left.localeCompare(right))
    : [];
  return `${result.visualization_run_id ?? result.dataset_id}:${candidate?.rank ?? "none"}:${JSON.stringify(values)}`;
}
