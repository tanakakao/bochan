from __future__ import annotations

import io

import pandas as pd
import pytest
import torch
from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor, nn

from bochan.api import ModelConfig, resolve_model_cls
from bochan.composition import ATOMIC_NUMBERS, RoostEncoder, parse_formula
from bochan.models.regression.gaussian.deep import (
    CompositionMaterialInputTransform,
    RoostDKLModel,
    RoostGPModel,
)
from bochan.tabular import TabularBayesianOptimizer


class FakeRoostDescriptor(nn.Module):
    """Small Roost descriptor exposing the canonical fine-tuning modules."""

    def __init__(self, width: int = 6, num_layers: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Linear(width, width)
        self.graphs = nn.ModuleList(nn.Linear(width, width) for _ in range(num_layers))
        self.cry_pool = nn.ModuleList([nn.Linear(width, width)])

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        del self_idx, nbr_idx
        features = torch.tanh(self.embedding(elem_fea))
        for graph in self.graphs:
            features = features + torch.tanh(graph(features))

        messages = torch.tanh(self.cry_pool[0](features)) * elem_weights
        num_materials = int(cry_elem_idx[-1].item()) + 1
        pooled = messages.new_zeros((num_materials, messages.shape[-1]))
        normalizer = elem_weights.new_zeros((num_materials, 1))
        pooled.index_add_(0, cry_elem_idx, messages)
        normalizer.index_add_(0, cry_elem_idx, elem_weights)
        return pooled / normalizer.clamp_min(torch.finfo(messages.dtype).eps)


class LayeredFakeRoost(nn.Module):
    """Five-tensor Roost backbone used to exercise the Tabular boundary."""

    def __init__(self, width: int = 6, num_layers: int = 3) -> None:
        super().__init__()
        self.output_dim = width
        self.elem_embedding = nn.Embedding(119, width)
        self.material_nn = FakeRoostDescriptor(width, num_layers)
        self.double()

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        return self.material_nn(
            elem_weights,
            self.elem_embedding(elem_fea),
            self_idx,
            nbr_idx,
            cry_elem_idx,
        )


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
            "pressure": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.0],
            "holding_time": [1.5, 2.0, 3.0, 4.0, 5.5, 6.5, 8.0, 9.0],
            "property": [0.4, 0.7, 1.1, 1.4, 1.8, 2.2, 2.5, 1.9],
        }
    )


def _composition_site() -> dict[str, object]:
    return {
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
        composition_sites={"formula": _composition_site()},
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


def _small_candidate(optimizer: TabularBayesianOptimizer, *, q: int = 1):
    return optimizer.candidate(
        acq_name="logei",
        q=q,
        num_restarts=2,
        raw_samples=16,
        optimizer_kwargs={"options": {"maxiter": 10, "batch_limit": 2}},
    )


def test_registry_exposes_only_normal_regression_roost_selectors() -> None:
    assert resolve_model_cls(ModelConfig(model_type="roost_gp")) is RoostGPModel
    assert resolve_model_cls(ModelConfig(model_type="roost_dkl")) is RoostDKLModel

    with pytest.raises(ValueError, match="Unknown model setting"):
        resolve_model_cls(
            ModelConfig(
                task_type="regression",
                model_type="roost_gp",
                input_type="mixed",
            )
        )
    with pytest.raises(ValueError, match="Unknown model setting"):
        resolve_model_cls(
            ModelConfig(
                task_type="multi_objective",
                model_type="roost_gp",
            )
        )


def test_tabular_roost_gp_fits_predicts_and_generates_q_batch_formulas() -> None:
    torch.manual_seed(0)
    optimizer = _optimizer("roost_gp", encoder=LayeredFakeRoost()).fit(_frame())
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model

    assert isinstance(model, RoostGPModel)
    assert isinstance(model.input_transform, CompositionMaterialInputTransform)
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

    candidates, acq_value = _small_candidate(optimizer, q=2)

    assert len(candidates) == 2
    assert all(set(parse_formula(formula)) == {"Ba", "Sr", "Ti", "O"} for formula in candidates["formula"])
    assert torch.isfinite(torch.as_tensor(acq_value)).all()
    assert candidates["temperature"].between(950.0, 1400.0).all()
    assert candidates["pressure"].between(0.5, 2.0).all()
    assert candidates["holding_time"].between(1.0, 10.0).all()


def test_tabular_roost_dkl_maps_partial_policy_and_jointly_fits() -> None:
    torch.manual_seed(0)
    encoder = LayeredFakeRoost()
    frozen_layer = encoder.material_nn.graphs[0]
    trainable_layer = encoder.material_nn.graphs[-1]
    pool = encoder.material_nn.cry_pool[0]
    frozen_before = frozen_layer.weight.detach().clone()
    trainable_before = trainable_layer.weight.detach().clone()
    pool_before = pool.weight.detach().clone()

    optimizer = _optimizer(
        "roost_dkl",
        encoder=encoder,
        model_kwargs={
            "encoder_training": "PARTIAL",
            "trainable_encoder_layers": 1,
        },
    ).fit(_frame())
    bundle = optimizer.bo.bundle
    assert bundle is not None
    model = bundle.model

    assert isinstance(model, RoostDKLModel)
    assert model.encoder_training == "partial"
    assert model.trainable_encoder_layers == 1
    assert bundle.metadata["fit_func"] == "fit_deepkernel_mll"
    assert torch.equal(frozen_layer.weight, frozen_before)
    assert not torch.equal(trainable_layer.weight, trainable_before)
    assert not torch.equal(pool.weight, pool_before)
    assert not frozen_layer.weight.requires_grad
    assert trainable_layer.weight.requires_grad
    assert pool.weight.requires_grad


def test_tabular_roost_dkl_defaults_to_partial_and_supports_full_training() -> None:
    partial = _optimizer("roost_dkl", encoder=LayeredFakeRoost())
    partial.fit_config.num_epochs = 0
    partial.fit(_frame())
    partial_model = partial.bo.bundle.model

    assert isinstance(partial_model, RoostDKLModel)
    assert partial_model.encoder_training == "partial"
    assert partial_model.trainable_encoder_layers == 1

    full = _optimizer(
        "roost_dkl",
        encoder=LayeredFakeRoost(),
        model_kwargs={"encoder_training": "full"},
    )
    full.fit_config.num_epochs = 0
    full.fit(_frame())
    full_model = full.bo.bundle.model

    assert isinstance(full_model, RoostDKLModel)
    assert full_model.encoder_training == "full"
    assert all(parameter.requires_grad for parameter in full_model.material_encoder.encoder.elem_embedding.parameters())
    assert all(parameter.requires_grad for parameter in full_model.material_encoder.encoder.material_nn.parameters())


def test_tabular_roost_multi_output_uses_independent_encoders() -> None:
    frame = _frame().assign(second_property=lambda value: value["property"] * 2)
    optimizer = _optimizer("roost_gp", encoder=LayeredFakeRoost())
    optimizer.source_data_config.target_cols = ["property", "second_property"]
    optimizer.fit_config.num_epochs = 0
    optimizer.fit(frame)

    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, ModelListGP)
    assert len(bundle.model.models) == 2
    first, second = bundle.model.models
    assert isinstance(first, RoostGPModel)
    assert isinstance(second, RoostGPModel)
    assert first.material_encoder is not second.material_encoder
    assert first.material_encoder.encoder is not second.material_encoder.encoder


@pytest.mark.parametrize(
    ("optimizer", "match"),
    [
        (
            TabularBayesianOptimizer(
                model_type="roost_gp",
                input_cols=["temperature"],
                target_cols="property",
                model_kwargs={"encoder": LayeredFakeRoost()},
                fit_config={"skip_fit": True},
            ),
            "exactly one composition site",
        ),
        (
            TabularBayesianOptimizer(
                model_type="roost_gp",
                input_cols=["formula", "temperature", "furnace"],
                target_cols="property",
                categorical_cols=["furnace"],
                composition_sites={"formula": _composition_site()},
                model_kwargs={"encoder": LayeredFakeRoost()},
                fit_config={"skip_fit": True},
            ),
            "continuous process columns only",
        ),
        (
            TabularBayesianOptimizer(
                model_type="roost_gp",
                input_cols=["formula", "temperature"],
                target_cols="property",
                composition_sites={"formula": _composition_site()},
                model_kwargs={
                    "encoder": LayeredFakeRoost(),
                    "encoder_training": "partial",
                },
                fit_config={"skip_fit": True},
            ),
            "always freezes the encoder",
        ),
        (
            TabularBayesianOptimizer(
                model_type="roost_dkl",
                input_cols=["formula", "temperature"],
                target_cols="property",
                composition_sites={"formula": _composition_site()},
                model_kwargs={
                    "encoder": LayeredFakeRoost(),
                    "encoder_training": "frozen",
                },
                fit_config={"skip_fit": True},
            ),
            "partial.*full",
        ),
    ],
)
def test_tabular_roost_rejects_unsupported_domain_or_training_modes(
    optimizer: TabularBayesianOptimizer,
    match: str,
) -> None:
    frame = _frame()
    if "furnace" in optimizer.source_data_config.input_cols:
        frame = frame.assign(furnace=["A", "B"] * 4)

    with pytest.raises(ValueError, match=match):
        optimizer.fit(frame)


def test_tabular_roost_rejects_multiple_sites_and_independent_descriptors() -> None:
    frame = _frame().assign(second_formula=_frame()["formula"])
    multiple = TabularBayesianOptimizer(
        model_type="roost_gp",
        input_cols=["formula", "second_formula", "temperature"],
        target_cols="property",
        composition_sites={
            "first": _composition_site(),
            "second": {
                **_composition_site(),
                "column": "second_formula",
            },
        },
        model_kwargs={"encoder": LayeredFakeRoost()},
        fit_config={"skip_fit": True},
    )
    with pytest.raises(ValueError, match="exactly one composition site"):
        multiple.fit(frame)

    descriptors = _optimizer("roost_gp", encoder=LayeredFakeRoost())
    descriptors.composition.sites["formula"]["include_descriptors"] = True
    with pytest.raises(ValueError, match="do not accept independent.*descriptor"):
        descriptors.fit(_frame())


def test_tabular_roost_optimizer_roundtrips_with_composition_state() -> None:
    optimizer = _optimizer("roost_gp", encoder=LayeredFakeRoost())
    optimizer.fit_config.num_epochs = 0
    optimizer.fit(_frame())
    reference = optimizer.predict(_frame().iloc[:2], return_type="dataframe")

    buffer = io.BytesIO()
    torch.save(optimizer, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    restored_prediction = restored.predict(
        _frame().iloc[:2],
        return_type="dataframe",
    )
    candidates, acq_value = _small_candidate(restored)

    pd.testing.assert_frame_equal(restored_prediction, reference)
    assert isinstance(restored.bo.bundle.model, RoostGPModel)
    assert set(parse_formula(candidates.loc[0, "formula"])) == {
        "Ba",
        "Sr",
        "Ti",
        "O",
    }
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_real_aviary_roost_runs_from_formula_rows_to_tabular_candidate() -> None:
    pytest.importorskip("aviary.roost.model")
    torch.manual_seed(0)
    encoder = RoostEncoder(
        elem_fea_len=8,
        n_graph=1,
        elem_heads=1,
        elem_gate=(8,),
        elem_msg=(8,),
        cry_heads=1,
        cry_gate=(8,),
        cry_msg=(8,),
    ).double()
    optimizer = _optimizer("roost_gp", encoder=encoder)
    optimizer.fit_config.num_epochs = 0
    optimizer.fit(_frame())

    candidates, acquisition_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    assert set(parse_formula(candidates.loc[0, "formula"])) == {
        "Ba",
        "Sr",
        "Ti",
        "O",
    }
    assert torch.isfinite(torch.as_tensor(acquisition_value)).all()
    assert 950.0 <= candidates.loc[0, "temperature"] <= 1400.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert 1.0 <= candidates.loc[0, "holding_time"] <= 10.0
