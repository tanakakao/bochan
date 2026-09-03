from __future__ import annotations

import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor

from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)
from bochan.tabular.structure.material_residual import (
    independent_residual_model_types,
    material_residual_model_types,
)


class _LinearBaseline(DirectMaterialPredictor):
    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: Tensor) -> Tensor:
        return 0.5 * X[..., :1] + 1.0


def test_residual_wrapper_is_model_list_gp_compatible() -> None:
    dtype = torch.double
    train_X = torch.linspace(0.0, 1.0, 6, dtype=dtype).unsqueeze(-1)
    predictor = _LinearBaseline()
    baseline = predictor(train_X)
    train_energy = baseline + 0.1 * torch.sin(train_X)
    train_strength = 2.0 * train_X + 0.2

    residual_model = SingleTaskGP(
        train_X,
        compute_material_residual_targets(train_X, train_energy, predictor),
    )
    residual = ResidualMaterialGPModel(
        predictor=predictor,
        residual_model=residual_model,
    )
    ordinary = SingleTaskGP(train_X, train_strength)

    model = ModelListGP(residual, ordinary)
    test_X = torch.tensor([[0.25], [0.75]], dtype=dtype)
    posterior = model.posterior(test_X)

    assert model.num_outputs == 2
    assert posterior.mean.shape == torch.Size([2, 2])
    assert torch.allclose(residual.baseline(test_X), predictor(test_X))
    assert residual.likelihood is residual_model.likelihood
    assert residual.train_inputs == residual_model.train_inputs


def test_independent_residual_public_model_types_are_distinct_from_multitask() -> None:
    expected = {
        "chgnet_multioutput_residual_gp",
        "chgnet_mixed_multioutput_residual_gp",
        "m3gnet_multioutput_residual_gp",
        "m3gnet_mixed_multioutput_residual_gp",
        "mace_multioutput_residual_gp",
        "mace_mixed_multioutput_residual_gp",
    }

    assert set(independent_residual_model_types()) == expected
    assert expected.issubset(set(material_residual_model_types()))
    assert not any("multitask" in model_type for model_type in expected)
