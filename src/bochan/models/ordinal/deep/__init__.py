from bochan.models.ordinal.robust._num_classes import enable_num_classes_inference

from ._mll_beta import enable_make_mll_beta
from .deepgp import OrdinalDeepGPModel, OrdinalMixedDeepGPModel
from .deepkernel_configurable import DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel
from .deepkerneldeepgp import DeepKernelOrdinalDeepGPModel, DeepKernelOrdinalMixedDeepGPModel


# Match the base / robust ordinal model API: when num_classes is omitted or
# explicitly None, infer it from canonicalized train_Y labels.
OrdinalDeepGPModel = enable_num_classes_inference(OrdinalDeepGPModel)
OrdinalMixedDeepGPModel = enable_num_classes_inference(OrdinalMixedDeepGPModel)
DeepKernelOrdinalGPModel = enable_num_classes_inference(DeepKernelOrdinalGPModel)
DeepKernelOrdinalMixedGPModel = enable_num_classes_inference(DeepKernelOrdinalMixedGPModel)

# DeepKernel ordinal models build either VariationalELBO or
# PredictiveLogLikelihood internally. Allow FitConfig(beta=...) to update the
# selected MLL's KL weight without changing that model-specific choice.
DeepKernelOrdinalGPModel = enable_make_mll_beta(DeepKernelOrdinalGPModel)
DeepKernelOrdinalMixedGPModel = enable_make_mll_beta(DeepKernelOrdinalMixedGPModel)


__all__ = [
    "OrdinalDeepGPModel",
    "OrdinalMixedDeepGPModel",
    "DeepKernelOrdinalGPModel",
    "DeepKernelOrdinalMixedGPModel",
    "DeepKernelOrdinalDeepGPModel",
    "DeepKernelOrdinalMixedDeepGPModel",
]
