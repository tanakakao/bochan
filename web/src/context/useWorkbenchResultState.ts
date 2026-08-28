import { useState } from "react";
import type { RegressionResult } from "../types";

/** Owns the latest optimization result and the signatures that produced it. */
export function useWorkbenchResultState() {
  const [result, setResultState] = useState<RegressionResult | null>(null);
  const [resultRevision, setResultRevision] = useState(0);
  const [lastModelSignature, setLastModelSignature] = useState<string | null>(null);
  const [lastSuggestionSignature, setLastSuggestionSignature] = useState<string | null>(null);

  function setResult(nextResult: RegressionResult | null) {
    setResultState(nextResult);
    setResultRevision((current) => current + 1);
  }

  function clearResult() {
    setResultState(null);
    setResultRevision((current) => current + 1);
    setLastModelSignature(null);
    setLastSuggestionSignature(null);
  }

  return {
    result,
    resultRevision,
    setResult,
    lastModelSignature,
    setLastModelSignature,
    lastSuggestionSignature,
    setLastSuggestionSignature,
    clearResult
  };
}
