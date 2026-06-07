from .base import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
    build_mixed_multiclass_kernel,
)
from .base.models import _BaseMulticlassClassificationModel
from .deep import (
    DeepKernelMulticlassClassificationGPModel,
    DeepKernelMulticlassClassificationMixedGPModel,
    MulticlassDeepGPModel,
    MulticlassMixedDeepGPModel,
)
from .high_dim import (
    PCAMulticlassClassificationGPModel,
    PCAMulticlassClassificationMixedGPModel,
    REMBOMulticlassClassificationGPModel,
    REMBOMulticlassClassificationMixedGPModel,
    SaasMulticlassClassificationGPModel,
    SaasMulticlassClassificationMixedGPModel,
)
from .robust import (
    HeteroscedasticMulticlassClassificationGPModel,
    HeteroscedasticMulticlassClassificationMixedGPModel,
    HeteroscedasticMulticlassPosterior,
    OutlierRelevancePursuitMulticlassClassificationGPModel,
    OutlierRelevancePursuitMulticlassClassificationMixedGPModel,
    SparseOutlierSoftmaxLikelihood,
)

__all__ = [
    "_BaseMulticlassClassificationModel",
    "MulticlassClassificationGPModel",
    "MulticlassClassificationMixedGPModel",
    "build_mixed_multiclass_kernel",
    "DeepKernelMulticlassClassificationGPModel",
    "DeepKernelMulticlassClassificationMixedGPModel",
    "MulticlassDeepGPModel",
    "MulticlassMixedDeepGPModel",
    "SaasMulticlassClassificationGPModel",
    "SaasMulticlassClassificationMixedGPModel",
    "PCAMulticlassClassificationGPModel",
    "PCAMulticlassClassificationMixedGPModel",
    "REMBOMulticlassClassificationGPModel",
    "REMBOMulticlassClassificationMixedGPModel",
    "SparseOutlierSoftmaxLikelihood",
    "OutlierRelevancePursuitMulticlassClassificationGPModel",
    "OutlierRelevancePursuitMulticlassClassificationMixedGPModel",
    "HeteroscedasticMulticlassPosterior",
    "HeteroscedasticMulticlassClassificationGPModel",
    "HeteroscedasticMulticlassClassificationMixedGPModel",
]
