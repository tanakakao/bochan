from .decomposition import PCAOrdinalGPModel, REMBOOrdinalGPModel, PCAOrdinalMixedGPModel, REMBOOrdinalMixedGPModel
from .saas_fixed import SaasOrdinalGPModel, SaasOrdinalMixedGPModel
from bochan.models.projected_input_perturbation_compat import (
    patch_projected_model_classes,
)


patch_projected_model_classes(
    [
        PCAOrdinalGPModel,
        REMBOOrdinalGPModel,
    ]
)


__all__ = [
    "PCAOrdinalGPModel", "REMBOOrdinalGPModel", "PCAOrdinalMixedGPModel", "REMBOOrdinalMixedGPModel",
    "SaasOrdinalGPModel", "SaasOrdinalMixedGPModel"
]
