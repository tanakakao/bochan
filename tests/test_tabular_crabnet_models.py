from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn

from bochan.api import ModelConfig, resolve_model_cls
from bochan.composition import ATOMIC_NUMBERS, CrabNetEncoder, parse_formula
from bochan.models.regression.gaussian.deep import (
    CrabNetDKLModel,
    CrabNetGPModel,
    CrabNetInputTransform,
)
from bochan.tabular import TabularBayesianOptimizer


class FakeTransformerEncoder(nn.Module):
    def __init__(self, width: int = 6, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(num_layers))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = torch.tanh(layer(values))
        return values


class LayeredFakeCrabNet(nn.Module):
    def __init__(self, width: int = 6, num_layers: int = 3) -> None:
        super().__init__()
        self.d_model = width
        self.embedding = nn.Embedding(119, width)
        self.transformer_encoder = FakeTransformerEncoder(width, num_layers)
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
            "pressure": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.0],
            "holding_time": [1.5, 2.0, 3.0, 4.0, 5.5, 6.5, 8.0, 9.0],
            "property": [0.4, 0.7, 1.1, 1.4, 1.8, 2.2, 2.5, 1.9],
        }
    )


def _optimizer(
    model_type: str,
    *,
    encoder: nn.Module,
    model_kwargs: dict[str, object] | None = None,
) -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type=model_type,
        input_cols=["formula", "temperature", "pressure", "holding_time"],
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
        bounds={
            "temperature": [950.0, 1400.0],
            "pressure": [0.5, 2.0],
            "holding_time": [1.0, 10.0],
        },
        model_kwargs={
            "encoder": encoder,
            "latent_dim": 4,
            **(model_kwargs or {}),
        },
        num_epochs=2,
        lr=0.01,
    )


def test_registry_exposes_only_normal_regression_crabnet_selectors() -> None:
    assert resolve_model_cls(ModelConfig(model_type="crabnet_gp")) is CrabNetGPModel
    assert resolve_model_cls(ModelConfig(model_type="crabnet_dkl")) is CrabNetDKLModel

    with pytest.raises(ValueError, match="Unknown model setting"):
        resolve_model_cls(
            ModelConfig(
                task_type="regression",
                model_type="crabnet_gp",
                input_type="mixed",
            )
        )
    with pytest.raises(ValueError, match="Unknown model setting"):
        resolve_model_cls(
            ModelConfig(
                task_type="multi_objective",
                model_type="crabnet_gp",
            )
        )


def test_crabnet_input_transform_reorders_coordinates_and_preserves_gradients() -> None:
    transform = CrabNetInputTransform(
        input_dim=5,
        composition_indices=[1, 3],
        n_components=3,
        method="ilr",
        process_bounds=torch.tensor(
            [[900.0, 0.5, 1.0], [1400.0, 2.0, 10.0]],
            dtype=torch.double,
        ),
    ).double()
    raw = torch.tensor(
        [[1100.0, 0.2, 1.1, -0.3, 4.0]],
        dtype=torch.double,
        requires_grad=True,
    )

    packed = transform(raw)
    loss = (packed * packed.new_tensor([1.0, 2.0, 4.0, 0.5, 0.7, 0.9])).sum()
    (gradient,) = torch.autograd.grad(loss, raw)

    assert packed.shape == torch.Size([1, 6])
    torch.testing.assert_close(
        packed[:, :3].sum(dim=-1),
        torch.ones(1, dtype=torch.double),
    )
    assert torch.isfinite(gradient).all()
    assert gradient[:, [1, 3]].abs().sum() > 0
    assert gradient[:, [0, 2, 4]].abs().sum() > 0


def test_tabular_crabnet_gp_derives_layout_fits_predicts_and_generates_candidate() -> None:
    optimizer = _optimizer("crabnet_gp", encoder=LayeredFakeCrabNet()).fit(_frame())
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model

    assert isinstance(model, CrabNetGPModel)
    assert isinstance(model.input_transform, CrabNetInputTransform)
    assert model.input_transform.composition_indices.tolist() == [0, 1, 2]
    assert model.input_transform.process_indices.tolist() == [3, 4, 5]
    assert model.element_ids.tolist() == [ATOMIC_NUMBERS[element] for element in ("Ba", "Sr", "Ti", "O")]
    assert model.composition_dim == 4
    assert model.process_dim == 3
    assert bundle.metadata["fit_func"] == "fit_deepkernel_mll"
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())

    raw = optimizer.dataset.X[:2].detach().clone().requires_grad_(True)
    posterior = model.posterior(raw)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), raw)
    prediction = optimizer.predict(_frame().iloc[:2], return_type="dataframe")

    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(gradient).all()
    assert gradient[:, :3].abs().sum() > 0
    assert gradient[:, 3:].abs().sum() > 0
    assert len(prediction) == 2

    candidates, acq_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        num_restarts=2,
        raw_samples=16,
        optimizer_kwargs={"options": {"maxiter": 10, "batch_limit": 2}},
    )
    parsed = parse_formula(candidates.loc[0, "formula"])

    assert set(parsed) == {"Ba", "Sr", "Ti", "O"}
    assert torch.isfinite(torch.as_tensor(acq_value)).all()
    assert 950.0 <= candidates.loc[0, "temperature"] <= 1400.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert 1.0 <= candidates.loc[0, "holding_time"] <= 10.0


def test_tabular_crabnet_dkl_maps_training_policy_and_jointly_fits() -> None:
    encoder = LayeredFakeCrabNet()
    frozen_layer = encoder.transformer_encoder.layers[0]
    trainable_layer = encoder.transformer_encoder.layers[-1]
    assert isinstance(frozen_layer, nn.Linear)
    assert isinstance(trainable_layer, nn.Linear)
    frozen_before = frozen_layer.weight.detach().clone()
    trainable_before = trainable_layer.weight.detach().clone()
    optimizer = _optimizer(
        "crabnet_dkl",
        encoder=encoder,
        model_kwargs={"encoder_training": "partial"},
    ).fit(_frame())
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model

    assert isinstance(model, CrabNetDKLModel)
    assert model.trainable_encoder_layers == 1
    assert bundle.metadata["fit_func"] == "fit_deepkernel_mll"
    assert torch.equal(frozen_layer.weight, frozen_before)
    assert not torch.equal(trainable_layer.weight, trainable_before)
    assert not frozen_layer.weight.requires_grad
    assert trainable_layer.weight.requires_grad


def test_tabular_crabnet_dkl_full_training_unfreezes_complete_encoder() -> None:
    optimizer = _optimizer(
        "crabnet_dkl",
        encoder=LayeredFakeCrabNet(),
        model_kwargs={"encoder_training": "full"},
    )
    optimizer.fit_config.num_epochs = 0
    optimizer.fit(_frame())
    model = optimizer.bo.bundle.model

    assert isinstance(model, CrabNetDKLModel)
    assert model.trainable_encoder_layers == "all"
    assert all(parameter.requires_grad for parameter in model.material_encoder.parameters())


@pytest.mark.parametrize(
    ("optimizer", "match"),
    [
        (
            TabularBayesianOptimizer(
                model_type="crabnet_gp",
                input_cols=["temperature"],
                target_cols="property",
                model_kwargs={"encoder": LayeredFakeCrabNet()},
                fit_config={"skip_fit": True},
            ),
            "exactly one composition site",
        ),
        (
            TabularBayesianOptimizer(
                model_type="crabnet_gp",
                input_cols=["formula", "temperature", "furnace"],
                target_cols="property",
                categorical_cols=["furnace"],
                composition_sites={
                    "formula": {
                        "column": "formula",
                        "elements": ["Ba", "Sr", "Ti", "O"],
                    }
                },
                model_kwargs={"encoder": LayeredFakeCrabNet()},
                fit_config={"skip_fit": True},
            ),
            "continuous process columns only",
        ),
    ],
)
def test_tabular_crabnet_rejects_unsupported_domain_shapes(
    optimizer: TabularBayesianOptimizer,
    match: str,
) -> None:
    frame = _frame()
    if "furnace" in optimizer.source_data_config.input_cols:
        frame = frame.assign(furnace=["A", "B"] * 4)

    with pytest.raises(ValueError, match=match):
        optimizer.fit(frame)


def test_tabular_crabnet_rejects_multi_output_and_invalid_encoder_training() -> None:
    frame = _frame().assign(second_property=lambda value: value["property"] * 2)
    multi_output = _optimizer("crabnet_gp", encoder=LayeredFakeCrabNet())
    multi_output.source_data_config.target_cols = ["property", "second_property"]

    with pytest.raises(ValueError, match="single-output"):
        multi_output.fit(frame)

    invalid_training = _optimizer(
        "crabnet_dkl",
        encoder=LayeredFakeCrabNet(),
        model_kwargs={"encoder_training": "frozen"},
    )
    with pytest.raises(ValueError, match="partial.*full"):
        invalid_training.fit(_frame())


def test_tabular_crabnet_rejects_multiple_sites_without_affecting_existing_api() -> None:
    frame = _frame().assign(second_formula=_frame()["formula"])
    optimizer = TabularBayesianOptimizer(
        model_type="crabnet_gp",
        input_cols=["formula", "second_formula", "temperature"],
        target_cols="property",
        composition_sites={
            "first": {
                "column": "formula",
                "elements": ["Ba", "Sr", "Ti", "O"],
            },
            "second": {
                "column": "second_formula",
                "elements": ["Ba", "Sr", "Ti", "O"],
            },
        },
        model_kwargs={"encoder": LayeredFakeCrabNet()},
        fit_config={"skip_fit": True},
    )

    with pytest.raises(ValueError, match="exactly one composition site"):
        optimizer.fit(frame)


def test_real_crabnet_runs_from_formula_rows_to_tabular_candidate() -> None:
    pytest.importorskip("crabnet.kingcrab")
    torch.manual_seed(0)
    encoder = CrabNetEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        pe_resolution=32,
        ple_resolution=32,
    )
    optimizer = _optimizer("crabnet_gp", encoder=encoder)
    optimizer.fit_config.num_epochs = 0
    optimizer.fit(_frame())

    candidates, acquisition_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    parsed = parse_formula(candidates.loc[0, "formula"])
    assert set(parsed) == {"Ba", "Sr", "Ti", "O"}
    assert torch.isfinite(torch.as_tensor(acquisition_value)).all()
    assert 950.0 <= candidates.loc[0, "temperature"] <= 1400.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert 1.0 <= candidates.loc[0, "holding_time"] <= 10.0
