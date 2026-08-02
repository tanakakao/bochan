"""Explicit heteroscedastic multi-output non-Gaussian AL API."""
from . import multi_output as _multi

def _hetero(name: str, parent: type) -> type:
    """Create a named multi-output heteroscedastic specialization."""
    return type(name, (parent,), {"__doc__": f"Heteroscedastic specialization of ``{parent.__name__}``."})

for _suffix in ["ResponseMeanVariance", "ExpectedObservationVariance", "TotalObservationVariance",
                "ExpectedObservationEntropy", "PredictiveEntropyProxy", "BALDProxy"]:
    globals()["qHeteroMultiOutputNonGaussian" + _suffix] = _hetero("qHeteroMultiOutputNonGaussian" + _suffix, getattr(_multi, "qMultiOutputNonGaussian" + _suffix))
__all__ = [n for n in globals() if n.startswith("qHetero")]
