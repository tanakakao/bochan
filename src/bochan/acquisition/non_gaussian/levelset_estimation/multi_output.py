"""Multi-output non-Gaussian level-set acquisitions."""
from . import single_output as _single
from bochan.acquisition.non_gaussian.active_learning.multi_output import _MultiOutputMixin

def _multi(name: str, parent: type) -> type:
    """Construct a named output-reducing LSE specialization."""
    return type(name, (_MultiOutputMixin, parent), {"__doc__": f"Multi-output specialization of ``{parent.__name__}``."})

for _suffix in ["Straddle","JointStraddle","BoundaryVariance","ICUProxy","ProbabilityOfExceedanceProxy",
                "ObservationProbabilityOfExceedance","LevelSetUncertainty"]:
    globals()["qMultiOutputNonGaussian"+_suffix]=_multi("qMultiOutputNonGaussian"+_suffix,getattr(_single,"qNonGaussian"+_suffix))
__all__=[n for n in globals() if n.startswith("qMultiOutput")]
