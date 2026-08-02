"""Heteroscedastic non-Gaussian level-set public classes."""
from . import single_output as _single

def _hetero(name: str,parent: type)->type:
    """Create an explicitly named heteroscedastic specialization."""
    return type(name,(parent,),{"__doc__":f"Heteroscedastic specialization of ``{parent.__name__}``."})
for _suffix in ["Straddle","JointStraddle","BoundaryVariance","ICUProxy","ProbabilityOfExceedanceProxy",
                "ObservationProbabilityOfExceedance","LevelSetUncertainty"]:
    globals()["qHeteroNonGaussian"+_suffix]=_hetero("qHeteroNonGaussian"+_suffix,getattr(_single,"qNonGaussian"+_suffix))
__all__=[n for n in globals() if n.startswith("qHetero")]
