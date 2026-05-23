from .multiclass_relevance_pursuit import (
    SparseOutlierSoftmaxLikelihood,
    OutlierRelevancePursuitMulticlassClassificationGPModel,
    OutlierRelevancePursuitMulticlassClassificationMixedGPModel,
)
from .multiclass_heteroscedastic import (
    HeteroscedasticMulticlassPosterior,
    HeteroscedasticMulticlassClassificationGPModel,
    HeteroscedasticMulticlassClassificationMixedGPModel,
)

__all__ = [
    "SparseOutlierSoftmaxLikelihood",
    "OutlierRelevancePursuitMulticlassClassificationGPModel",
    "OutlierRelevancePursuitMulticlassClassificationMixedGPModel",
    "HeteroscedasticMulticlassPosterior",
    "HeteroscedasticMulticlassClassificationGPModel",
    "HeteroscedasticMulticlassClassificationMixedGPModel",
]
