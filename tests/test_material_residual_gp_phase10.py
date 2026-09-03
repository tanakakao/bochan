from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP
from torch import Tensor, nn

from bochan.models.regression.gaussian.materials.common import (
    DirectMaterialPredictor,
    PretrainedMaterialCapabilities,
    PretrainedMaterialSpec,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
    predict_material_baseline,
    require_residual_gp_capability,
    validate_direct_material_predictions,
)


class _LinearBaseline(DirectMaterialPredictor):
    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        self._output_dim = output_dim
        self.linear = nn.Linear(1, output_dim, bias=False, dtype=torch.double)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, X: Tensor) -> Tensor:
        return self.linear(X)


def _residual_capable_spec() -> PretrainedMaterialSpec:
    return PretrainedMaterialSpec(
        family="dummy",
        domain="structure",
        capabilities=PretrainedMaterialCapabilities(
            representation=True,
            direct_prediction=True,
            residual_gp=True,
        ),
    )


def test_predict_material_baseline_validates_shape_device_and_dtype() -> None:
    predictor = _LinearBaseline()
    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)

    prediction = predict_material_baseline(predictor, X)

    assert prediction.shape == torch.Size([2, 1])
    assert torch.allclose(prediction, 2.0 * X)


def test_validate_direct_predictions_rejects_bad_shape() -> None:
    X = torch.zeros(3, 2, dtype=torch.double)
    prediction = torch.zeros(3, 2, dtype=torch.double)

    with pytest.raises(ValueError, match="output_dim"):
        validate_direct_material_predictions(X, prediction, output_dim=1)


def test_compute_material_residual_targets_preserves_partial_observations() -> None:
    predictor = _LinearBaseline()
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0.1], [float("nan")], [2.4]], dtype=torch.double)

    residual = compute_material_residual_targets(X, Y, predictor)

    assert torch.allclose(residual[[0, 2]], torch.tensor([[0.1], [0.4]], dtype=torch.double))
    assert torch.isnan(residual[1, 0])


def test_residual_model_adds_baseline_to_posterior_mean_only() -> None:
    predictor = _LinearBaseline()
    train_X = torch.tensor([[0.0], [0.3], [0.6], [1.0]], dtype=torch.double)
    train_Y = 2.0 * train_X + torch.tensor(
        [[0.0], [0.1], [-0.05], [0.08]], dtype=torch.double
    )
    residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
    residual_model = SingleTaskGP(train_X, residual_Y)
    model = ResidualMaterialGPModel(
        predictor=predictor,
        residual_model=residual_model,
        pretrained_spec=_residual_capable_spec(),
    )
    test_X = torch.tensor([[0.2], [0.8]], dtype=torch.double)

    residual_posterior = residual_model.posterior(test_X)
    corrected_posterior = model.posterior(test_X)
    baseline = predictor(test_X)

    assert torch.allclose(
        corrected_posterior.mean,
        residual_posterior.mean + baseline,
    )
    assert torch.allclose(corrected_posterior.variance, residual_posterior.variance)


def test_residual_model_requires_matching_output_width() -> None:
    predictor = _LinearBaseline(output_dim=2)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    residual_model = SingleTaskGP(X, torch.zeros(2, 1, dtype=torch.double))

    with pytest.raises(ValueError, match="predictor.output_dim"):
        ResidualMaterialGPModel(predictor=predictor, residual_model=residual_model)


def test_require_residual_gp_capability_rejects_direct_only_backend() -> None:
    spec = PretrainedMaterialSpec(
        family="direct-only",
        domain="structure",
        capabilities=PretrainedMaterialCapabilities(
            representation=True,
            direct_prediction=True,
            residual_gp=False,
        ),
    )

    with pytest.raises(ValueError, match="not residual-GP capable"):
        require_residual_gp_capability(spec)
