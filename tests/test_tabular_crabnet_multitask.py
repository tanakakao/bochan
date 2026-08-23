from __future__ import annotations

import pandas as pd
import pytest
import torch
from gpytorch.kernels import MultitaskKernel
from torch import nn

from bochan.api import DataContext
from bochan.composition import parse_formula
from bochan.models.regression.gaussian.deep import CrabNetMultiTaskGPModel
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
            "property_a": [0.4, 0.7, 1.1, 1.4, 1.8, 2.2, 2.5, 1.9],
            "property_b": [0.55, 0.82, 1.15, 1.48, 1.85, 2.15, 2.42, 1.95],
        }
    )


def _optimizer(*, cross_validation: bool = False) -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        task_type="multi_objective",
        model_type="crabnet_multitask",
        input_cols=["formula", "temperature"],
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
        model_kwargs={"encoder": _FakeCrabNet(), "latent_dim": 4},
        num_epochs=1,
        lr=0.01,
        cross_validation=cross_validation,
        cv_config={"splitter": "kfold", "n_splits": 2, "shuffle": False},
    )


def test_crabnet_multitask_builds_one_shared_correlated_model() -> None:
    torch.manual_seed(0)
    optimizer = _optimizer().fit(_frame())
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model

    assert isinstance(model, CrabNetMultiTaskGPModel)
    assert bundle.model_type == "crabnet_multitask"
    assert bundle.task_type == "multi_objective"
    assert bundle.metadata["multi_output"] is False
    assert model.num_outputs == 2
    assert model.num_tasks == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    assert model.task_covar_module is model.deepkernel.covar_module.task_covar_module
    assert model.task_kernel is model.task_covar_module
    assert not any(
        parameter.requires_grad for parameter in model.material_encoder.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.projection.parameters())
    assert any(parameter.requires_grad for parameter in model.task_covar_module.parameters())
    assert bundle.metadata["fit_func"] == "fit_deepkernel_mll"

    raw = optimizer.dataset.X[:2].detach().clone().requires_grad_(True)
    posterior = model.posterior(raw)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), raw)

    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_crabnet_multitask_optimizes_multiobjective_nehvi() -> None:
    torch.manual_seed(0)
    optimizer = _optimizer().fit(_frame())
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
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_crabnet_multitask_cross_validation_preserves_correlated_model() -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(cross_validation=True).fit(_frame())

    result = optimizer.cross_validation_result_
    assert result is not None
    assert set(result.outputs) == {"output_0", "output_1"}
    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, CrabNetMultiTaskGPModel)
    assert isinstance(bundle.model.deepkernel.covar_module, MultitaskKernel)


def test_crabnet_multitask_requires_multiple_targets() -> None:
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="crabnet_multitask",
        input_cols=["formula", "temperature"],
        target_cols="property_a",
        composition_sites={
            "formula": {
                "column": "formula",
                "elements": ["Ba", "Sr", "Ti", "O"],
            }
        },
        bounds={"temperature": [950.0, 1400.0]},
        model_kwargs={"encoder": _FakeCrabNet(), "latent_dim": 4},
        fit_config={"skip_fit": True},
    )

    with pytest.raises(ValueError, match="at least two continuous target columns"):
        optimizer.fit(_frame())


def test_crabnet_multitask_rejects_categorical_process_and_encoder_training() -> None:
    frame = _frame().assign(atmosphere=["air", "N2"] * 4)
    categorical = TabularBayesianOptimizer(
        task_type="multi_objective",
        model_type="crabnet_multitask",
        input_cols=["formula", "temperature", "atmosphere"],
        categorical_cols=["atmosphere"],
        target_cols=["property_a", "property_b"],
        composition_sites={
            "formula": {
                "column": "formula",
                "elements": ["Ba", "Sr", "Ti", "O"],
            }
        },
        bounds={"temperature": [950.0, 1400.0]},
        model_kwargs={"encoder": _FakeCrabNet(), "latent_dim": 4},
        fit_config={"skip_fit": True},
    )
    with pytest.raises(ValueError, match="mixed-multitask"):
        categorical.fit(frame)

    trainable = _optimizer()
    trainable.model_config.model_kwargs["encoder_training"] = "partial"
    with pytest.raises(ValueError, match="always freezes"):
        trainable.fit(_frame())
