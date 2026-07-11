from bochan.models.projected_input_perturbation import (
    configure_projected_model_classes,
)

from .decomposition import PCAOrdinalGPModel, PCAOrdinalMixedGPModel, REMBOOrdinalGPModel, REMBOOrdinalMixedGPModel
from .saas_fixed import SaasOrdinalGPModel, SaasOrdinalMixedGPModel

configure_projected_model_classes(
    [
        PCAOrdinalGPModel,
        REMBOOrdinalGPModel,
    ]
)


__all__ = [
    "PCAOrdinalGPModel", "REMBOOrdinalGPModel", "PCAOrdinalMixedGPModel", "REMBOOrdinalMixedGPModel",
    "SaasOrdinalGPModel", "SaasOrdinalMixedGPModel"
]
