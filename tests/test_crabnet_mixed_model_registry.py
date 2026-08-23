from bochan.api.registry.model import MODEL_REGISTRY
from bochan.models.regression.gaussian.deep import (
    CrabNetMixedDKLModel,
    CrabNetMixedGPModel,
)


def test_mixed_registry_exposes_crabnet_gp_and_dkl_models() -> None:
    registry = MODEL_REGISTRY["mixed"]["regression"]

    assert registry["crabnet_mixed_gp"] is CrabNetMixedGPModel
    assert registry["crabnet_mixed_dkl"] is CrabNetMixedDKLModel
