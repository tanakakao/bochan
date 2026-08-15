import { useState } from "react";
import type { RegressionResult } from "../types";

/** Owns the latest optimization result and the signature of its fitted model. */
export function useWorkbenchResultState() {
  const [result, setResult] = useState<RegressionResult | null>(null);
  const [lastModelSignature, setLastModelSignature] = useState<string | null>(null);

  function clearResult() {
    setResult(null);
    setLastModelSignature(null);
  }

  return {
    result,
    setResult,
    lastModelSignature,
    setLastModelSignature,
    clearResult
  };
}
