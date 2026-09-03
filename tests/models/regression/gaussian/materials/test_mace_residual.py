from __future__ import annotations

import torch
from torch import nn

from bochan.models.regression.gaussian.materials import get_material_family
from bochan.models.regression.gaussian.materials.structure import (
    MACEDirectEnergyPredictor,
    MACEResidualGPModel,
)


def test_mace_registry_exposes_residual_gp() -> None:
    registration = get_material_family("mace")
    assert registration.supports("residual_gp")
    assert registration.pretrained.capabilities.direct_prediction is True
    assert registration.pretrained.capabilities.residual_gp is True
    assert registration.resolve_model_class("residual_gp") is MACEResidualGPModel


def test_mace_direct_predictor_is_public() -> None:
    assert issubclass(MACEDirectEnergyPredictor, nn.Module)


def test_mace_residual_model_is_public() -> None:
    assert MACEResidualGPModel.__module__.endswith("materials.structure.mace_residual")
