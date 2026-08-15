/**
 * Resolve the increment used by an HTML number input from the current magnitude.
 *
 * Rules:
 * - |value| >= 10: 1
 * - 1 <= |value| < 10: 0.1
 * - 0.1 <= |value| < 1: 0.01
 * - 0.01 <= |value| < 0.1: 0.001
 *
 * Zero and empty values use the first finite, non-zero fallback magnitude.
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

function isDynamicNumberInput(element: EventTarget | null): element is HTMLInputElement {
  if (!(element instanceof HTMLInputElement) || element.type !== "number") return false;
  if (element.dataset.dynamicNumberStep === "true") return true;

  const rawStep = element.getAttribute("step");
  const explicitStep = rawStep === null || rawStep === "any" ? Number.NaN : Number(rawStep);
  const continuousStep = rawStep === "any" || (
    Number.isFinite(explicitStep) && explicitStep > 0 && explicitStep < 1
  );
  const continuousConstraintValue = Boolean(element.closest(".constraint-relation-row"));

  if (!continuousStep && !continuousConstraintValue) return false;
  element.dataset.dynamicNumberStep = "true";
  return true;
}

/** Update one React-owned number input before the browser applies spinner/arrow increments. */
export function updateDynamicNumberInputStep(element: EventTarget | null): void {
  if (!isDynamicNumberInput(element)) return;
  element.step = String(dynamicNumberStep(element.value, [element.min, element.max]));
}