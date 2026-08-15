import { useEffect } from "react";
import { updateDynamicNumberInputStep } from "../numericInputUtils";

/** Keeps continuous number-input increments adaptive while tying listeners to React lifecycle. */
export default function DynamicNumberStepBridge() {
  useEffect(() => {
    const root = document.getElementById("root");
    if (!root) return;

    const updateFromEvent = (event: Event) => updateDynamicNumberInputStep(event.target);
    const updateFromKeyboard = (event: KeyboardEvent) => {
      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        updateDynamicNumberInputStep(event.target);
      }
    };

    root.addEventListener("focusin", updateFromEvent, true);
    root.addEventListener("pointerdown", updateFromEvent, true);
    root.addEventListener("input", updateFromEvent, true);
    root.addEventListener("keydown", updateFromKeyboard, true);

    return () => {
      root.removeEventListener("focusin", updateFromEvent, true);
      root.removeEventListener("pointerdown", updateFromEvent, true);
      root.removeEventListener("input", updateFromEvent, true);
      root.removeEventListener("keydown", updateFromKeyboard, true);
    };
  }, []);

  return null;
}