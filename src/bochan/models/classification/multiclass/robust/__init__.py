from .heteroscedastic import (
    HeteroscedasticMulticlassClassificationGPModel,
    HeteroscedasticMulticlassClassificationMixedGPModel,
    HeteroscedasticMulticlassPosterior,
)
from .relevance_pursuit import (
    RobustRelevancePursuitMulticlassClassificationGPModel,
    RobustRelevancePursuitMulticlassClassificationMixedGPModel,
    SparseOutlierSoftmaxLikelihood,
)

# InputPerturbation 等で base posterior と noise posterior の軸順がずれる場合の互換 patch。

__all__ = [
    "SparseOutlierSoftmaxLikelihood",
    "RobustRelevancePursuitMulticlassClassificationGPModel",
    "RobustRelevancePursuitMulticlassClassificationMixedGPModel",
    "HeteroscedasticMulticlassPosterior",
    "HeteroscedasticMulticlassClassificationGPModel",
    "HeteroscedasticMulticlassClassificationMixedGPModel",
]
