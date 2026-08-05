const COMPOSITION_CHANGE_EVENT = "bochan-composition-settings-change";
const CONTROL_SELECTOR = ".composition-kind-control";

type InstallExtension = () => void;

let installed = false;

function categoryEnabled(control: HTMLElement): boolean {
  const card = control.closest<HTMLElement>(".feature-variable-choice");
  const checkbox = card?.querySelector<HTMLInputElement>(".feature-type-toggle input");
  return checkbox?.checked === true;
}

function refreshSegmentedControl(control: HTMLElement): void {
  const select = control.querySelector<HTMLSelectElement>("select");
  if (!select) return;

  const visible = categoryEnabled(control);
  control.hidden = !visible;
  control.setAttribute("aria-hidden", String(!visible));

  control.querySelectorAll<HTMLButtonElement>("button[data-composition-kind]").forEach((button) => {
    const active = visible && button.dataset.compositionKind === select.value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function upgradeControl(control: HTMLElement): void {
  const select = control.querySelector<HTMLSelectElement>("select");
  if (!select) return;

  if (control.dataset.segmented !== "true") {
    control.dataset.segmented = "true";
    control.classList.add("composition-kind-control-segmented");
    select.classList.add("composition-kind-native-select");

    const group = document.createElement("div");
    group.className = "composition-kind-segment";
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", select.getAttribute("aria-label") ?? "入力表記");

    ([
      ["normal", "通常"],
      ["composition", "組成式"]
    ] as const).forEach(([value, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "composition-kind-option";
      button.dataset.compositionKind = value;
      button.textContent = label;
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (select.value === value) return;
        select.value = value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        refreshSegmentedControl(control);
      });
      group.appendChild(button);
    });

    control.appendChild(group);
  }

  refreshSegmentedControl(control);
}

function addedCompositionControls(record: MutationRecord): HTMLElement[] {
  const controls = new Set<HTMLElement>();
  record.addedNodes.forEach((node) => {
    if (!(node instanceof Element)) return;
    if (node.matches(CONTROL_SELECTOR)) controls.add(node as HTMLElement);
    node.querySelectorAll<HTMLElement>(CONTROL_SELECTOR).forEach((control) => {
      controls.add(control);
    });
  });
  return [...controls];
}

function refreshSegmentedControls(): void {
  document.querySelectorAll<HTMLElement>(CONTROL_SELECTOR).forEach((control) => {
    if (control.dataset.segmented === "true") refreshSegmentedControl(control);
    else upgradeControl(control);
  });
}

function turnOffCompositionWhenCategoryIsDisabled(input: HTMLInputElement): void {
  if (input.checked) return;
  const card = input.closest<HTMLElement>(".feature-variable-choice");
  const control = card?.querySelector<HTMLElement>(CONTROL_SELECTOR);
  const select = control?.querySelector<HTMLSelectElement>("select");
  if (!select || select.value !== "composition") return;

  select.value = "normal";
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

/**
 * Installs the composition extension and upgrades only newly inserted selector controls.
 *
 * The observer deliberately ignores unrelated page mutations and never replaces the global
 * MutationObserver constructor. This prevents observer feedback loops when composition state
 * changes while preserving the segmented selector UI.
 */
export function installCompositionPrepareControls(installExtension: InstallExtension): void {
  if (installed) return;
  installed = true;

  installExtension();

  const observer = new MutationObserver((records) => {
    const controls = new Set<HTMLElement>();
    records.forEach((record) => {
      addedCompositionControls(record).forEach((control) => controls.add(control));
    });
    controls.forEach(upgradeControl);
  });
  observer.observe(document.documentElement, { subtree: true, childList: true });

  document.addEventListener("change", (event) => {
    const target = event.target as Element | null;
    const categoryInput = target?.closest<HTMLInputElement>(".feature-type-toggle input");
    if (categoryInput) {
      queueMicrotask(() => {
        turnOffCompositionWhenCategoryIsDisabled(categoryInput);
        refreshSegmentedControls();
      });
      return;
    }

    const select = target?.closest<HTMLSelectElement>(`${CONTROL_SELECTOR} select`);
    if (select) {
      const control = select.closest<HTMLElement>(CONTROL_SELECTOR);
      if (control) refreshSegmentedControl(control);
    }
  });

  window.addEventListener(COMPOSITION_CHANGE_EVENT, refreshSegmentedControls);
  queueMicrotask(refreshSegmentedControls);
}
