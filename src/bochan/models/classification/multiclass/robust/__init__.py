from .heteroscedastic import (
    HeteroscedasticMulticlassClassificationGPModel,
    HeteroscedasticMulticlassClassificationMixedGPModel,
    HeteroscedasticMulticlassPosterior,
)
from .relevance_pursuit import (
    OutlierRelevancePursuitMulticlassClassificationGPModel,
    OutlierRelevancePursuitMulticlassClassificationMixedGPModel,
    SparseOutlierSoftmaxLikelihood,
)

__all__ = [
    "SparseOutlierSoftmaxLikelihood",
    "OutlierRelevancePursuitMulticlassClassificationGPModel",
    "OutlierRelevancePursuitMulticlassClassificationMixedGPModel",
    "HeteroscedasticMulticlassPosterior",
    "HeteroscedasticMulticlassClassificationGPModel",
    "HeteroscedasticMulticlassClassificationMixedGPModel",
]
