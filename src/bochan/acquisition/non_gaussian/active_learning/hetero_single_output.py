"""Explicit heteroscedastic non-Gaussian active-learning API."""
from . import single_output as _single

def _hetero(name: str, parent: type) -> type:
    """Create a named thin class sharing the variance-aware core."""
    return type(name, (parent,), {"__doc__": f"Heteroscedastic specialization of ``{parent.__name__}``."})

for _suffix in ["ResponseMeanVariance", "ExpectedObservationVariance", "TotalObservationVariance",
                "ExpectedObservationEntropy", "PredictiveEntropyProxy", "BALDProxy"]:
    globals()["qHeteroNonGaussian" + _suffix] = _hetero("qHeteroNonGaussian" + _suffix, getattr(_single, "qNonGaussian" + _suffix))
__all__ = [n for n in globals() if n.startswith("qHetero")]
