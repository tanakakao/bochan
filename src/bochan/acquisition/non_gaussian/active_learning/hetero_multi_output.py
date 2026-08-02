"""Explicit heteroscedastic multi-output non-Gaussian AL API."""
from . import multi_output as _multi


def _hetero(name: str, parent: type) -> type:
    """Create a named multi-output heteroscedastic specialization."""
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
    globals()["qHeteroMultiOutputNonGaussian" + _suffix] = _hetero(
        "qHeteroMultiOutputNonGaussian" + _suffix,
        getattr(_multi, "qMultiOutputNonGaussian" + _suffix),
    )

qHeteroMultiOutputNonGaussianNegIntegratedPosteriorVariance = globals()[
    "qHeteroMultiOutputNonGaussianNegIntegratedResponseMeanVariance"
]
qHeteroMultiOutputNonGaussianNIPV = globals()[
    "qHeteroMultiOutputNonGaussianNegIntegratedResponseMeanVariance"
]

__all__ = [name for name in globals() if name.startswith("qHetero")]
