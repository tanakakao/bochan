const ICON_SELECTOR = ".conversation-launcher-icon";
const ICON_DIRECTORY = `${import.meta.env.BASE_URL}conversation-mode/`;
const ICON_FILENAMES = ["icon.png", "icon.svg", "icon.webp", "icon.jpg", "icon.jpeg"] as const;

let installed = false;
let observer: MutationObserver | null = null;

function loadOptionalIcon(container: HTMLElement): void {
  if (container.dataset.conversationIconBound === "true") return;
  container.dataset.conversationIconBound = "true";

  const fallbackText = container.textContent?.trim() || "✦";

  function tryFilename(index: number): void {
    const filename = ICON_FILENAMES[index];
    if (!filename) {
      container.textContent = fallbackText;
      container.dataset.conversationIconState = "fallback";
      return;
    }

    const image = new Image();
    image.alt = "";
    image.decoding = "async";
    image.addEventListener("load", () => {
      image.className = "conversation-launcher-icon-image";
      image.style.width = "100%";
      image.style.height = "100%";
      image.style.display = "block";
      image.style.objectFit = "contain";
      image.style.borderRadius = "inherit";
      container.replaceChildren(image);
      container.dataset.conversationIconState = "image";
    }, { once: true });
    image.addEventListener("error", () => tryFilename(index + 1), { once: true });
    image.src = `${ICON_DIRECTORY}${filename}`;
  }

  tryFilename(0);
}

function applyConversationIcon(): void {
  document.querySelectorAll<HTMLElement>(ICON_SELECTOR).forEach(loadOptionalIcon);
}

/** Loads an optional public icon while preserving the built-in symbol as fallback. */
export function installConversationIconRuntime(): void {
  if (installed) return;
  installed = true;

  const start = () => {
    applyConversationIcon();
    observer = new MutationObserver(applyConversationIcon);
    observer.observe(document.documentElement, { childList: true, subtree: true });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
}
