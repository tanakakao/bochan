const FRACTION_TOKEN = "__fraction__";
let scheduled = false;

function compositionFractionOptions(card: Element): Set<string> {
  return new Set(
    Array.from(card.querySelectorAll<HTMLOptionElement>("option"))
      .map((option) => option.value)
      .filter((value) => value.includes(FRACTION_TOKEN))
  );
}

function guardTernarySelection(): void {
  document.querySelectorAll<HTMLElement>(".interactive-plot-card").forEach((card) => {
    const kindSelect = Array.from(card.querySelectorAll<HTMLSelectElement>("select"))
      .find((select) => select.querySelector('option[value="ternary"]'));
    const ternaryOption = kindSelect?.querySelector<HTMLOptionElement>(
      'option[value="ternary"]'
    );
    if (!kindSelect || !ternaryOption) return;

    const fractionOptions = compositionFractionOptions(card);
    if (fractionOptions.size === 0) return;

    const available = fractionOptions.size >= 3;
    ternaryOption.disabled = !available;
    ternaryOption.title = available
      ? "候補元素から3元素を選択し、組成断面を三角図で表示します。"
      : "組成の三角図には候補元素が3種類以上必要です。";

    if (!available && kindSelect.value === "ternary") {
      kindSelect.value = "1d";
      kindSelect.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
}

function scheduleGuard(): void {
  if (scheduled) return;
  scheduled = true;
  queueMicrotask(() => {
    scheduled = false;
    guardTernarySelection();
  });
}

export function installCompositionVisualizationGuard(): void {
  const observer = new MutationObserver(scheduleGuard);
  observer.observe(document.documentElement, { subtree: true, childList: true });
  document.addEventListener("change", scheduleGuard);
  scheduleGuard();
}
