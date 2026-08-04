const COMPOSITION_CHANGE_EVENT = "bochan-composition-settings-change";
const CONTROL_SELECTOR = ".composition-kind-control";
const OWNED_MUTATION_SELECTOR = [
  CONTROL_SELECTOR,
  ".composition-model-settings-host",
  ".composition-constraint-settings-host",
  ".composition-settings-panel"
].join(", ");

type InstallExtension = () => void;

let synchronizeScheduled = false;

function mutationTargetElement(target: Node): Element | null {
  if (target instanceof Element) return target;
  return target.parentElement;
}

function nodeIsCompositionOwned(node: Node): boolean {
  if (node instanceof Element) {
    return node.matches(OWNED_MUTATION_SELECTOR) || Boolean(node.closest(OWNED_MUTATION_SELECTOR));
  }
  return Boolean(node.parentElement?.closest(OWNED_MUTATION_SELECTOR));
}

function mutationAddsCompositionControl(record: MutationRecord): boolean {
  return Array.from(record.addedNodes).some((node) => {
    if (!(node instanceof Element)) return false;
    return node.matches(CONTROL_SELECTOR) || Boolean(node.querySelector(CONTROL_SELECTOR));
  });
}

function isCompositionOwnedMutation(record: MutationRecord): boolean {
  const target = mutationTargetElement(record.target);
  if (target?.closest(OWNED_MUTATION_SELECTOR)) return true;

  const changedNodes = [
    ...Array.from(record.addedNodes),
    ...Array.from(record.removedNodes)
  ];
  return changedNodes.length > 0 && changedNodes.every(nodeIsCompositionOwned);
}

function withCompositionMutationGuard(installExtension: InstallExtension): void {
  const NativeMutationObserver = window.MutationObserver;

  class GuardedMutationObserver {
    private readonly observer: MutationObserver;

    constructor(callback: MutationCallback) {
      this.observer = new NativeMutationObserver((records, observer) => {
        const externalRecords = records.filter((record) => !isCompositionOwnedMutation(record));
        if (externalRecords.length) callback(externalRecords, observer);
      });
    }

    observe(target: Node, options?: MutationObserverInit): void {
      this.observer.observe(target, options);
    }

    disconnect(): void {
      this.observer.disconnect();
    }

    takeRecords(): MutationRecord[] {
      return this.observer.takeRecords();
    }
  }

  Object.defineProperty(window, "MutationObserver", {
    configurable: true,
    writable: true,
    value: GuardedMutationObserver as unknown as typeof MutationObserver
  });

  try {
    installExtension();
  } finally {
    Object.defineProperty(window, "MutationObserver", {
      configurable: true,
      writable: true,
      value: NativeMutationObserver
    });
  }
}

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

function synchronizeSegmentedControls(): void {
  document.querySelectorAll<HTMLElement>(CONTROL_SELECTOR).forEach(upgradeControl);
}

function scheduleSynchronizeSegmentedControls(): void {
  if (synchronizeScheduled) return;
  synchronizeScheduled = true;
  queueMicrotask(() => {
    synchronizeScheduled = false;
    synchronizeSegmentedControls();
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

function installSegmentedControlObserver(): void {
  const observer = new MutationObserver((records) => {
    const shouldSynchronize = records.some((record) => (
      mutationAddsCompositionControl(record) || !isCompositionOwnedMutation(record)
    ));
    if (shouldSynchronize) scheduleSynchronizeSegmentedControls();
  });
  observer.observe(document.documentElement, { subtree: true, childList: true });

  document.addEventListener("change", (event) => {
    const target = event.target as Element | null;
    const categoryInput = target?.closest<HTMLInputElement>(".feature-type-toggle input");
    if (categoryInput) {
      queueMicrotask(() => {
        turnOffCompositionWhenCategoryIsDisabled(categoryInput);
        scheduleSynchronizeSegmentedControls();
      });
      return;
    }

    const select = target?.closest<HTMLSelectElement>(`${CONTROL_SELECTOR} select`);
    if (select) {
      const control = select.closest<HTMLElement>(CONTROL_SELECTOR);
      if (control) refreshSegmentedControl(control);
    }
  });

  window.addEventListener(COMPOSITION_CHANGE_EVENT, scheduleSynchronizeSegmentedControls);
  scheduleSynchronizeSegmentedControls();
}

export function installCompositionPrepareControls(installExtension: InstallExtension): void {
  withCompositionMutationGuard(installExtension);
  installSegmentedControlObserver();
}
