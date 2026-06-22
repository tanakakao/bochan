from ._num_classes import enable_num_classes_inference
from .heteroscedastic import HeteroscedasticOrdinalGPModel, HeteroscedasticOrdinalMixedGPModel
from .relevance_pursuit import OutlierRelevancePursuitOrdinalGPModel, OutlierRelevancePursuitOrdinalMixedGPModel


_ROBUST_ORDINAL_MODELS = (
    OutlierRelevancePursuitOrdinalGPModel,
    OutlierRelevancePursuitOrdinalMixedGPModel,
    HeteroscedasticOrdinalGPModel,
    HeteroscedasticOrdinalMixedGPModel,
)

for _model_cls in _ROBUST_ORDINAL_MODELS:
    enable_num_classes_inference(_model_cls)


__all__ = [
    "OutlierRelevancePursuitOrdinalGPModel",
    "OutlierRelevancePursuitOrdinalMixedGPModel",
    "HeteroscedasticOrdinalGPModel",
    "HeteroscedasticOrdinalMixedGPModel",
]
