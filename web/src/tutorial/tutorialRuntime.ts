import { useEffect, useState } from "react";

function tutorialSampleLoaderVisible(): boolean {
  if (typeof document === "undefined") return false;
  return Boolean(document.querySelector(
    ".tutorial-guide-card.tutorial-kind-sample, .tutorial-guide-card.tutorial-kind-advanced"
  ));
}

/** Returns true only while a practical tutorial that uses the sample dataset is open. */
export function useSampleTutorialActive(): boolean {
  const [active, setActive] = useState(tutorialSampleLoaderVisible);

  useEffect(() => {
    const update = () => setActive(tutorialSampleLoaderVisible());
    update();

    const observer = new MutationObserver(update);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class"]
    });

    return () => observer.disconnect();
  }, []);

  return active;
}
