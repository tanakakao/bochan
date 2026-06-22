from . import heteroscedastic as _heteroscedastic
from . import relevance_pursuit as _relevance_pursuit
from ._num_classes import (
    HeteroscedasticOrdinalGPModel,
    HeteroscedasticOrdinalMixedGPModel,
    OutlierRelevancePursuitOrdinalGPModel,
    OutlierRelevancePursuitOrdinalMixedGPModel,
)

# Keep direct imports from the implementation modules aligned with the public API.
_relevance_pursuit.OutlierRelevancePursuitOrdinalGPModel = OutlierRelevancePursuitOrdinalGPModel
_relevance_pursuit.OutlierRelevancePursuitOrdinalMixedGPModel = OutlierRelevancePursuitOrdinalMixedGPModel
_heteroscedastic.HeteroscedasticOrdinalGPModel = HeteroscedasticOrdinalGPModel
_heteroscedastic.HeteroscedasticOrdinalMixedGPModel = HeteroscedasticOrdinalMixedGPModel

__all__ = [
    "OutlierRelevancePursuitOrdinalGPModel",
    "OutlierRelevancePursuitOrdinalMixedGPModel",
    "HeteroscedasticOrdinalGPModel",
    "HeteroscedasticOrdinalMixedGPModel",
]
