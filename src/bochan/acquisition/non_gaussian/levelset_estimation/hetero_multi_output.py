"""Heteroscedastic multi-output non-Gaussian LSE public classes."""
from . import multi_output as _multi

def _hetero(name: str,parent: type)->type:
    """Create an explicitly named heteroscedastic multi-output class."""
    return type(name,(parent,),{"__doc__":f"Heteroscedastic specialization of ``{parent.__name__}``."})
for _suffix in ["Straddle","JointStraddle","BoundaryVariance","ICUProxy","ProbabilityOfExceedanceProxy",
                "ObservationProbabilityOfExceedance","LevelSetUncertainty"]:
    globals()["qHeteroMultiOutputNonGaussian"+_suffix]=_hetero("qHeteroMultiOutputNonGaussian"+_suffix,getattr(_multi,"qMultiOutputNonGaussian"+_suffix))
__all__=[n for n in globals() if n.startswith("qHetero")]
