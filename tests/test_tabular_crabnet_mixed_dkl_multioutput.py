from __future__ import annotations

import pandas as pd
import pytest
import torch
from botorch.models.model_list_gp_regression import ModelListGP
from torch import nn

from bochan.api import DataContext
from bochan.composition import parse_formula
from bochan.models.regression.gaussian.deep import (
    CrabNetDKLModel,
    CrabNetGPModel,
    CrabNetMixedDKLModel,
    CrabNetMixedGPModel,
)
from bochan.tabular import TabularBayesianOptimizer


class _FakeTransformerEncoder(nn.Module):
    def __init__(self, width: int = 6, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(num_layers)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = torch.tanh(layer(values))
        return values


class _FakeCrabNet(nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        self.d_model = width
        self.embedding = nn.Embedding(119, width)
        self.transformer_encoder = _FakeTransformerEncoder(width)
        self.double()

    def forward(
        self,
        element_ids: torch.Tensor,
        fractions: torch.Tensor,
    ) -> torch.Tensor:
        values = self.embedding(element_ids) * fractions.unsqueeze(-1)
        return self.transformer_encoder(values)


MODEL_CLASSES = {
    "crabnet_gp": CrabNetGPModel,
    "crabnet_dkl": CrabNetDKLModel,
    "crabnet_mixed_gp": CrabNetMixedGPModel,
    "crabnet_mixed_dkl": CrabNetMixedDKLModel,
}
MIXED_MODELS = {"crabnet_mixed_gp", "crabnet_mixed_dkl"}
DKL_MODELS = {"crabnet_dkl", "crabnet_mixed_dkl"}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "formula": [
                "Ba0.45Sr0.15Ti0.10O0.30",
                "Ba0.35Sr0.20Ti0.15O0.30",
                "Ba0.25Sr0.30Ti0.20O0.25",
                "Ba0.20Sr0.25Ti0.25O0.30",
                "Ba0.30Sr0.15Ti0.25O0.30",
                "Ba0.40Sr0.10Ti0.20O0.30",
                "Ba0.15Sr0.35Ti0.20O0.30",
                "Ba0.30Sr0.25Ti0.15O0.30",
            ],
            "temperature": [
                1000.0,
                1050.0,
                1100.0,
                1150.0,
                1200.0,
                1250.0,
                1300.0,
                1350.0,
            ],
            "atmosphere": ["air", "N2", "air", "Ar", "N2", "Ar", "air", "N2"],
            "method": [
                "furnace",
                "SPS",
                "SPS",
                "furnace",
                "SPS",
                "furnace",
                "furnace",
                "SPS",
            ],
            "property_a": [0.4, 0.7, 1.1, 1.4, 1.8, 2.2, 2.5, 1.9],
            "property_b": [2.3, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 1.1],
        }
    )


def _optimizer(
    model_type: str,
    *,
    cross_validation: bool = False,
) -> TabularBayesianOptimizer:
    mixed = model_type in MIXED_MODELS
    model_kwargs: dict[str, object] = {
        "encoder": _FakeCrabNet(),
        "latent_dim": 4,
    }
    if model_type in DKL_MODELS:
        model_kwargs["encoder_training"] = "partial"
    if model_type == "crabnet_mixed_dkl":
        model_kwargs.update(
            {
                "category_embedding_dims": [3, 2],
                "projection_hidden_dim": 8,
            }
        )

    return TabularBayesianOptimizer(
        task_type="multi_objective",
        model_type=model_type,
        input_cols=(
            ["formula", "temperature", "atmosphere", "method"]
            if mixed
            else ["formula", "temperature"]
        ),
        categorical_cols=["atmosphere", "method"] if mixed else [],
        target_cols=["property_a", "property_b"],
        composition_sites={
            "formula": {
                "column": "formula",
                "elements": ["Ba", "Sr", "Ti", "O"],
                "representation": "ilr",
                "coordinate_bounds": (-3.0, 3.0),
                "bounds": {
                    "Ba": [0.05, 0.70],
                    "Sr": [0.05, 0.70],
                    "Ti": [0.05, 0.70],
                    "O": [0.05, 0.80],
                },
            }
        },
        bounds={"temperature": [950.0, 1400.0]},
        model_kwargs=model_kwargs,
        num_epochs=1,
        lr=0.01,
        cross_validation=cross_validation,
        cv_config={"splitter": "kfold", "n_splits": 2, "shuffle": False},
    )


@pytest.mark.parametrize("model_type", list(MODEL_CLASSES))
def test_crabnet_models_build_fully_independent_model_lists(model_type: str) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(model_type).fit(_frame())
    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert bundle.metadata["multi_output"] is True
    assert len(bundle.model.models) == 2

    first, second = bundle.model.models
    expected_cls = MODEL_CLASSES[model_type]
    assert isinstance(first, expected_cls)
    assert isinstance(second, expected_cls)
    assert first is not second
    assert first.material_encoder is not second.material_encoder
    first_encoder_parameter = next(first.material_encoder.parameters())
    second_encoder_parameter = next(second.material_encoder.parameters())
    assert first_encoder_parameter.data_ptr() != second_encoder_parameter.data_ptr()

    if model_type == "crabnet_mixed_dkl":
        assert first.category_embeddings is not second.category_embeddings
        assert (
            first.category_embeddings[0].weight.data_ptr()
            != second.category_embeddings[0].weight.data_ptr()
        )
        assert first.category_cardinalities == (3, 2)
        assert second.category_cardinalities == (3, 2)

    sub_bundles = bundle.metadata["sub_bundles"]
    assert len(sub_bundles) == 2
    assert all(item.metadata["fit_func"] == "fit_deepkernel_mll" for item in sub_bundles)

    raw = optimizer.dataset.X[:2].detach().clone().requires_grad_(True)
    posterior = bundle.model.posterior(raw)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), raw)
    assert posterior.mean.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(gradient).all()
    continuous_dims = sorted(set(range(raw.shape[-1])) - set(bundle.cat_dims))
    assert gradient[:, continuous_dims].abs().sum() > 0


@pytest.mark.parametrize("model_type", list(MODEL_CLASSES))
def test_crabnet_multioutput_optimizes_nehvi(model_type: str) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(model_type).fit(_frame())
    assert optimizer.train_X is not None
    assert optimizer.train_Y is not None

    ref_point = optimizer.train_Y.min(dim=0).values - 0.1
    candidates, acq_value = optimizer.candidate(
        acq_name="nehvi",
        q=1,
        objective_mode="multi_output",
        objective_outputs=[0, 1],
        objective_directions=["maximize", "maximize"],
        data_context=DataContext(
            X_baseline=optimizer.train_X,
            Y_baseline=optimizer.train_Y,
            ref_point=ref_point,
        ),
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    parsed = parse_formula(candidates.loc[0, "formula"])
    assert set(parsed) == {"Ba", "Sr", "Ti", "O"}
    assert 950.0 <= candidates.loc[0, "temperature"] <= 1400.0
    if model_type in MIXED_MODELS:
        assert candidates.loc[0, "atmosphere"] in {"air", "N2", "Ar"}
        assert candidates.loc[0, "method"] in {"furnace", "SPS"}
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


@pytest.mark.parametrize("model_type", list(MODEL_CLASSES))
def test_crabnet_multioutput_cross_validation_keeps_independent_models(
    model_type: str,
) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(model_type, cross_validation=True).fit(_frame())

    result = optimizer.cross_validation_result_
    assert result is not None
    assert set(result.outputs) == {"property_a", "property_b"}
    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    first, second = bundle.model.models
    assert first.material_encoder is not second.material_encoder
