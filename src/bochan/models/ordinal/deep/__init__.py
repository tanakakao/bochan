from bochan.models.ordinal.robust._num_classes import enable_num_classes_inference

from .deepgp import OrdinalDeepGPModel, OrdinalMixedDeepGPModel
from .deepkernel_configurable import DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel
from .deepkerneldeepgp import DeepKernelOrdinalDeepGPModel, DeepKernelOrdinalMixedDeepGPModel


# Match the base / robust ordinal model API: when num_classes is omitted or
# explicitly None, infer it from canonicalized train_Y labels.
OrdinalDeepGPModel = enable_num_classes_inference(OrdinalDeepGPModel)
OrdinalMixedDeepGPModel = enable_num_classes_inference(OrdinalMixedDeepGPModel)
DeepKernelOrdinalGPModel = enable_num_classes_inference(DeepKernelOrdinalGPModel)
DeepKernelOrdinalMixedGPModel = enable_num_classes_inference(DeepKernelOrdinalMixedGPModel)


__all__ = [
    "OrdinalDeepGPModel",
    "OrdinalMixedDeepGPModel",
    "DeepKernelOrdinalGPModel",
    "DeepKernelOrdinalMixedGPModel",
    "DeepKernelOrdinalDeepGPModel",
    "DeepKernelOrdinalMixedDeepGPModel",
]
