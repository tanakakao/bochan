from .heteroscedastic_compat import apply_heteroscedastic_alignment_compat
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

# InputPerturbation 等で base posterior と noise posterior の軸順がずれる場合の互換 patch。
apply_heteroscedastic_alignment_compat()

__all__ = [
    "SparseOutlierSoftmaxLikelihood",
    "OutlierRelevancePursuitMulticlassClassificationGPModel",
    "OutlierRelevancePursuitMulticlassClassificationMixedGPModel",
    "HeteroscedasticMulticlassPosterior",
    "HeteroscedasticMulticlassClassificationGPModel",
    "HeteroscedasticMulticlassClassificationMixedGPModel",
    "apply_heteroscedastic_alignment_compat",
]
