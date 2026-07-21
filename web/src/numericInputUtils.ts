/**
 * Resolve the increment used by an HTML number input from the current magnitude.
 *
 * Rules:
 * - |value| >= 10: 1
 * - 1 <= |value| < 10: 0.1
 * - 0.1 <= |value| < 1: 0.01
 * - 0.01 <= |value| < 0.1: 0.001
 *
 * Zero and empty values use the first finite, non-zero fallback magnitude. This
 * lets a zero lower bound inherit an appropriate increment from its search range.
 */
export function dynamicNumberStep(
  value: unknown,
  fallbackValues: unknown[] = []
): number {
  const direct = Number(value);
  let magnitude = Number.isFinite(direct) ? Math.abs(direct) : 0;

  if (magnitude === 0) {
    const fallbackMagnitudes = fallbackValues
      .map((candidate) => Number(candidate))
      .filter((candidate) => Number.isFinite(candidate) && candidate !== 0)
      .map(Math.abs);
    magnitude = fallbackMagnitudes[0] ?? 1;
  }

  if (magnitude >= 10) return 1;

  const exponent = Math.floor(Math.log10(magnitude));
  return Number((10 ** (exponent - 1)).toPrecision(12));
}
