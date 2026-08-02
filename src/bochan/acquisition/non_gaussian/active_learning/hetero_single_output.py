"""Explicit heteroscedastic non-Gaussian active-learning API."""
from . import single_output as _single


def _hetero(name: str, parent: type) -> type:
    """Create a named class sharing the heteroscedastic variance-aware core."""
    return type(
        name,
        (parent,),
        {
            "__doc__": (
                f"Heteroscedastic specialization of ``{parent.__name__}``."
            )
        },
    )


for _suffix in [
    "ResponseMeanVariance",
    "ExpectedObservationVariance",
    "TotalObservationVariance",
    "ExpectedObservationEntropy",
    "PredictiveEntropyProxy",
    "BALDProxy",
    "IntegratedResponseMeanVarianceProxy",
    "NegIntegratedResponseMeanVariance",
    "JointBALDProxy",
    "GreedyJointBALDProxy",
]:
    globals()["qHeteroNonGaussian" + _suffix] = _hetero(
        "qHeteroNonGaussian" + _suffix,
        getattr(_single, "qNonGaussian" + _suffix),
    )

qHeteroNonGaussianNegIntegratedPosteriorVariance = globals()[
    "qHeteroNonGaussianNegIntegratedResponseMeanVariance"
]
qHeteroNonGaussianNIPV = globals()[
    "qHeteroNonGaussianNegIntegratedResponseMeanVariance"
]

__all__ = [name for name in globals() if name.startswith("qHetero")]
