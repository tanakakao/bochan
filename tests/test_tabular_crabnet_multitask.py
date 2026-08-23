from __future__ import annotations

import pandas as pd
import pytest
import torch
from gpytorch.kernels import MultitaskKernel
from torch import nn

from bochan.api import DataContext
from bochan.composition import parse_formula
from bochan.models.regression.gaussian.deep import (
    CrabNetMixedMultiTaskDKLModel,
    CrabNetMixedMultiTaskGPModel,
    CrabNetMultiTaskDKLModel,
    CrabNetMultiTaskGPModel,
)
from bochan.serving.fastapi.schemas.tabular import TabularFitModelRequest
from bochan.serving.fastapi.services.tabular import fit_tabular_optimizer
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


MODEL_CASES = [
    ("crabnet_multitask", CrabNetMultiTaskGPModel, False, False),
    ("crabnet_multitask_dkl", CrabNetMultiTaskDKLModel, False, True),
    ("crabnet_mixed_multitask", CrabNetMixedMultiTaskGPModel, True, False),
    (
        "crabnet_mixed_multitask_dkl",
        CrabNetMixedMultiTaskDKLModel,
        True,
        True,
    ),
]


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
            "atmosphere": ["air", "N2", "air", "Ar", "N2", "air", "Ar", "N2"],
            "property_a": [0.4, 0.7, 1.1, 1.4, 1.8, 2.2, 2.5, 1.9],
            "property_b": [0.55, 0.82, 1.15, 1.48, 1.85, 2.15, 2.42, 1.95],
        }
    )


def _optimizer(
    model_type: str,
    *,
    mixed: bool,
    dkl: bool,
    cross_validation: bool = False,
    target_cols: list[str] | str | None = None,
) -> TabularBayesianOptimizer:
    input_cols = ["formula", "temperature"]
    categorical_cols = None
    if mixed:
        input_cols.append("atmosphere")
        categorical_cols = ["atmosphere"]

    model_kwargs: dict[str, object] = {
        "encoder": _FakeCrabNet(),
        "latent_dim": 4,
    }
    if dkl:
        model_kwargs["encoder_training"] = "partial"

    return TabularBayesianOptimizer(
        task_type="multi_objective",
        model_type=model_type,
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        target_cols=target_cols or ["property_a", "property_b"],
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


def _fastapi_payload(model_type: str, *, mixed: bool, dkl: bool) -> dict[str, object]:
    input_cols = ["formula", "temperature"]
    categorical_cols: list[str] = []
    if mixed:
        input_cols.append("atmosphere")
        categorical_cols = ["atmosphere"]

    model_kwargs: dict[str, object] = {
        "encoder": _FakeCrabNet(),
        "latent_dim": 4,
    }
    if dkl:
        model_kwargs["encoder_training"] = "partial"

    return {
        "data": _frame().to_dict(orient="records"),
        "model_config": {
            "task_type": "multi_objective",
            "model_type": model_type,
            "input_type": "mixed" if mixed else "normal",
            "model_kwargs": model_kwargs,
        },
        "fit_config": {"skip_fit": True},
        "input_cols": input_cols,
        "categorical_cols": categorical_cols,
        "target_cols": ["property_a", "property_b"],
        "composition_sites": {
            "formula": {
                "column": "formula",
                "elements": ["Ba", "Sr", "Ti", "O"],
                "representation": "ilr",
                "coordinate_bounds": [-3.0, 3.0],
                "bounds": {
                    "Ba": [0.05, 0.70],
                    "Sr": [0.05, 0.70],
                    "Ti": [0.05, 0.70],
                    "O": [0.05, 0.80],
                },
            }
        },
        "bounds": {"temperature": [950.0, 1400.0]},
    }


@pytest.mark.parametrize(
    ("model_type", "expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_crabnet_multitask_family_builds_one_correlated_model(
    model_type: str,
    expected_cls: type,
    mixed: bool,
    dkl: bool,
) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(model_type, mixed=mixed, dkl=dkl).fit(_frame())
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model

    assert isinstance(model, expected_cls)
    assert bundle.model_type == model_type
    assert bundle.task_type == "multi_objective"
    assert bundle.model_config.multi_output_config is None
    assert bundle.metadata["multi_output"] is False
    assert model.num_outputs == 2
    assert model.num_tasks == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    assert model.task_covar_module is model.deepkernel.covar_module.task_covar_module
    assert model.task_kernel is model.task_covar_module
    assert bundle.metadata["fit_func"] == "fit_deepkernel_mll"

    encoder_parameters = list(model.material_encoder.parameters())
    assert encoder_parameters
    if dkl:
        assert any(parameter.requires_grad for parameter in encoder_parameters)
    else:
        assert not any(parameter.requires_grad for parameter in encoder_parameters)

    if mixed:
        assert list(bundle.cat_dims) == list(optimizer.dataset.cat_dims)
        assert model.categorical_process_dim == 1
    else:
        assert not bundle.cat_dims

    if model_type == "crabnet_mixed_multitask_dkl":
        assert len(model.category_embeddings) == 1
        assert model.category_cardinalities == (3,)

    raw = optimizer.dataset.X[:2].detach().clone().requires_grad_(True)
    posterior = model.posterior(raw)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), raw)

    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(gradient).all()
    continuous_dims = sorted(
        set(range(raw.shape[-1])) - set(optimizer.dataset.cat_dims)
    )
    assert gradient[:, continuous_dims].abs().sum() > 0


@pytest.mark.parametrize(
    ("model_type", "_expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_crabnet_multitask_family_optimizes_multiobjective_nehvi(
    model_type: str,
    _expected_cls: type,
    mixed: bool,
    dkl: bool,
) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(model_type, mixed=mixed, dkl=dkl).fit(_frame())
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
    if mixed:
        assert candidates.loc[0, "atmosphere"] in {"air", "N2", "Ar"}
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


@pytest.mark.parametrize(
    ("model_type", "expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_crabnet_multitask_family_cross_validation_preserves_correlated_model(
    model_type: str,
    expected_cls: type,
    mixed: bool,
    dkl: bool,
) -> None:
    torch.manual_seed(0)
    optimizer = _optimizer(
        model_type,
        mixed=mixed,
        dkl=dkl,
        cross_validation=True,
    ).fit(_frame())

    result = optimizer.cross_validation_result_
    assert result is not None
    assert set(result.outputs) == {"output_0", "output_1"}
    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, expected_cls)
    assert isinstance(bundle.model.deepkernel.covar_module, MultitaskKernel)
    assert bundle.model_config.multi_output_config is None


@pytest.mark.parametrize(
    ("model_type", "expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_fastapi_accepts_and_builds_correlated_crabnet_multitask(
    model_type: str,
    expected_cls: type,
    mixed: bool,
    dkl: bool,
) -> None:
    request = TabularFitModelRequest.model_validate(
        _fastapi_payload(model_type, mixed=mixed, dkl=dkl)
    )
    optimizer = fit_tabular_optimizer(request)
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, expected_cls)
    assert bundle.model_config.multi_output_config is None
    assert bundle.model.num_outputs == 2
    assert isinstance(bundle.model.deepkernel.covar_module, MultitaskKernel)
    encoder_parameters = list(bundle.model.material_encoder.parameters())
    if dkl:
        assert any(parameter.requires_grad for parameter in encoder_parameters)
    else:
        assert not any(parameter.requires_grad for parameter in encoder_parameters)


@pytest.mark.parametrize(
    ("model_type", "_expected_cls", "mixed", "dkl"),
    MODEL_CASES,
)
def test_crabnet_multitask_family_requires_multiple_targets(
    model_type: str,
    _expected_cls: type,
    mixed: bool,
    dkl: bool,
) -> None:
    optimizer = _optimizer(
        model_type,
        mixed=mixed,
        dkl=dkl,
        target_cols="property_a",
    )

    with pytest.raises(ValueError, match="requires at least two continuous target columns"):
        optimizer.fit(_frame())


def test_crabnet_multitask_process_type_errors_point_to_correlated_counterparts() -> None:
    continuous_with_category = _optimizer(
        "crabnet_multitask_dkl",
        mixed=False,
        dkl=True,
    )
    continuous_with_category.source_data_config.categorical_cols = ["atmosphere"]
    continuous_with_category.source_data_config.input_cols = [
        "formula",
        "temperature",
        "atmosphere",
    ]
    with pytest.raises(ValueError, match="crabnet_mixed_multitask_dkl"):
        continuous_with_category.fit(_frame())

    mixed_without_category = _optimizer(
        "crabnet_mixed_multitask",
        mixed=True,
        dkl=False,
    )
    mixed_without_category.source_data_config.categorical_cols = None
    mixed_without_category.source_data_config.input_cols = ["formula", "temperature"]
    with pytest.raises(ValueError, match="crabnet_multitask"):
        mixed_without_category.fit(_frame())
