from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from bochan.composition import parse_formula
from bochan.models.regression.gaussian.deep import CrabNetMixedGPModel
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
            "temperature": [1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0],
            "atmosphere": ["air", "N2", "air", "Ar", "N2", "Ar", "air", "N2"],
            "method": ["furnace", "SPS", "SPS", "furnace", "SPS", "furnace", "furnace", "SPS"],
            "property": [0.4, 0.7, 1.1, 1.4, 1.8, 2.2, 2.5, 1.9],
        }
    )


def _optimizer() -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type="crabnet_mixed_gp",
        input_cols=["formula", "temperature", "atmosphere", "method"],
        categorical_cols=["atmosphere", "method"],
        target_cols="property",
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
        model_kwargs={"encoder": _FakeCrabNet(), "latent_dim": 4},
        num_epochs=1,
        lr=0.01,
    )


def test_tabular_crabnet_mixed_gp_fits_predicts_and_optimizes_categories() -> None:
    torch.manual_seed(0)
    optimizer = _optimizer().fit(_frame())
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model

    assert isinstance(model, CrabNetMixedGPModel)
    assert bundle.input_type == "mixed"
    assert len(bundle.cat_dims) == 2
    assert model.cat_dims == bundle.cat_dims
    assert model.process_dim == 1
    assert model.categorical_process_dim == 2
    assert bundle.metadata["fit_func"] == "fit_deepkernel_mll"
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())

    raw = optimizer.dataset.X[:2].detach().clone().requires_grad_(True)
    posterior = model.posterior(raw)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), raw)

    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(gradient).all()
    continuous_dims = sorted(set(range(raw.shape[-1])) - set(bundle.cat_dims))
    assert gradient[:, continuous_dims].abs().sum() > 0

    candidates, acq_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        num_restarts=2,
        raw_samples=16,
        optimizer_kwargs={"options": {"maxiter": 10, "batch_limit": 2}},
    )

    parsed = parse_formula(candidates.loc[0, "formula"])
    assert set(parsed) == {"Ba", "Sr", "Ti", "O"}
    assert 950.0 <= candidates.loc[0, "temperature"] <= 1400.0
    assert candidates.loc[0, "atmosphere"] in {"air", "N2", "Ar"}
    assert candidates.loc[0, "method"] in {"furnace", "SPS"}
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_crabnet_mixed_gp_requires_categorical_process_input() -> None:
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="crabnet_mixed_gp",
        input_cols=["formula", "temperature"],
        target_cols="property",
        composition_sites={
            "formula": {
                "column": "formula",
                "elements": ["Ba", "Sr", "Ti", "O"],
            }
        },
        model_kwargs={"encoder": _FakeCrabNet(), "latent_dim": 4},
        fit_config={"skip_fit": True},
    )

    try:
        optimizer.fit(_frame())
    except ValueError as error:
        assert "requires at least one categorical process column" in str(error)
    else:
        raise AssertionError("crabnet_mixed_gp must reject all-continuous process inputs")
