const COMPOSITION_CHANGE_EVENT = "bochan-composition-settings-change";
const OWNED_MUTATION_SELECTOR = ".composition-settings-host, .composition-kind-control";

type InstallExtension = () => void;

function mutationTargetElement(target: Node): Element | null {
  if (target instanceof Element) return target;
  return target.parentElement;
}

function isCompositionOwnedMutation(record: MutationRecord): boolean {
  const target = mutationTargetElement(record.target);
  return Boolean(target?.closest(OWNED_MUTATION_SELECTOR));
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

function refreshSegmentedControl(control: HTMLElement): void {
  const select = control.querySelector<HTMLSelectElement>("select");
  if (!select) return;
  control.querySelectorAll<HTMLButtonElement>("button[data-composition-kind]").forEach((button) => {
    const active = button.dataset.compositionKind === select.value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function upgradeControl(control: HTMLElement): void {
  if (control.dataset.segmented === "true") {
    refreshSegmentedControl(control);
    return;
  }

  const select = control.querySelector<HTMLSelectElement>("select");
  if (!select) return;

  const replacement = document.createElement("div");
  replacement.className = `${control.className} composition-kind-control-segmented`;
  replacement.dataset.segmented = "true";

  select.classList.add("composition-kind-native-select");
  replacement.appendChild(select);

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
      refreshSegmentedControl(replacement);
    });
    group.appendChild(button);
  });

  replacement.appendChild(group);
  control.replaceWith(replacement);
  refreshSegmentedControl(replacement);
}

function synchronizeSegmentedControls(): void {
  document.querySelectorAll<HTMLElement>(".composition-kind-control").forEach(upgradeControl);
}

function installSegmentedControlObserver(): void {
  const observer = new MutationObserver(() => synchronizeSegmentedControls());
  observer.observe(document.documentElement, { subtree: true, childList: true });
  document.addEventListener("change", (event) => {
    const select = (event.target as Element | null)?.closest<HTMLSelectElement>(
      ".composition-kind-control select"
    );
    if (select) refreshSegmentedControl(select.closest<HTMLElement>(".composition-kind-control")!);
  });
  window.addEventListener(COMPOSITION_CHANGE_EVENT, synchronizeSegmentedControls);
  synchronizeSegmentedControls();
}

export function installCompositionPrepareControls(installExtension: InstallExtension): void {
  withCompositionMutationGuard(installExtension);
  installSegmentedControlObserver();
}
