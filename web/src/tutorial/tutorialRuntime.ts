import { useEffect, useState } from "react";

function sampleTutorialVisible(): boolean {
  if (typeof document === "undefined") return false;
  return Boolean(document.querySelector(".tutorial-guide-card.tutorial-kind-sample"));
}

/** Returns true only while the practical sample tutorial guide is open. */
export function useSampleTutorialActive(): boolean {
  const [active, setActive] = useState(sampleTutorialVisible);

  useEffect(() => {
    const update = () => setActive(sampleTutorialVisible());
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
