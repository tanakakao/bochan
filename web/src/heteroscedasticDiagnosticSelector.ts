export const HETEROSCEDASTIC_SCOPE_ALL = "all";

export interface HeteroscedasticScopeOption {
  value: string;
  label: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

/** Return selectable overall / raw feature scopes from a noise profile. */
export function heteroscedasticScopeOptions(
  value: unknown,
  featureColumns: string[]
): HeteroscedasticScopeOption[] {
  const diagnostic = asRecord(value);
  const profile = asRecord(diagnostic?.noise_profile);
  const profileNames = Array.isArray(profile?.feature_names)
    ? profile.feature_names.map(String)
    : [];
  const names = profileNames.length ? profileNames : featureColumns;
  return [
    { value: HETEROSCEDASTIC_SCOPE_ALL, label: "全体" },
    ...names.map((name, index) => ({
      value: String(index),
      label: name
    }))
  ];
}

/** Select the overall row plot or one original-column scatter plot. */
export function filterHeteroscedasticFigures<T extends { id: string }>(
  figures: T[],
  scope: string
): T[] {
  if (scope === HETEROSCEDASTIC_SCOPE_ALL) {
    return figures.filter((figure) => figure.id.endsWith("-heteroscedastic-by-row"));
  }
  return figures.filter((figure) => figure.id.includes(`-heteroscedastic-${scope}-`));
}
