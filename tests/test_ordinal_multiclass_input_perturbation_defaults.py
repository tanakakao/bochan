from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

import bochan.api  # noqa: F401 - installs high-level compatibility routes
import bochan.api.engine as engine_module
import bochan.api.factory as factory_module
from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation_compat import (
    InputPerturbationMultiOutputObjectiveAdapter,
)
from bochan.acquisition.ordinal.bayesian_optimization import (
    qMultiOutputOrdinalUtilityObjective,
)
from bochan.acquisition.ordinal.bayesian_optimization import (
    hetero_multi_output as hetero_ordinal_module,
)
from bochan.api import (
    AcquisitionConfig,
    InputTransformConfig,
    ModelConfig,
    ObjectiveConfig,
)


class _DummyOrdinalLikelihood(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def class_probs_from_f(self, latent: torch.Tensor) -> torch.Tensor:
        value = latent.squeeze(-1) if latent.shape[-1] == 1 else latent
        if self.num_classes != 3:
            raise NotImplementedError("The test likelihood currently uses 3 classes.")
        logits = torch.stack(
            (-value, torch.zeros_like(value), value),
            dim=-1,
        )
        return torch.softmax(logits, dim=-1)


class _DummyOrdinalSubmodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ordinal_likelihood = _DummyOrdinalLikelihood()
        self.num_classes = 3


class _DummyOrdinalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.models = nn.ModuleList(
            [_DummyOrdinalSubmodel(), _DummyOrdinalSubmodel()]
        )

    @property
    def num_outputs(self) -> int:
        return len(self.models)


class _DummyMulticlassModel(nn.Module):
    @property
    def num_outputs(self) -> int:
        return 2


def _make_bundle(task_type: str, model, *, n_w: int = 16):
    model_config = ModelConfig(
        task_type=task_type,
        model_type="base",
        input_transform_config=InputTransformConfig(
            normalize=True,
            perturbation=True,
            n_w=n_w,
        ),
        outcome_transform=False,
    )
    return SimpleNamespace(
        task_type=task_type,
        model_type="base",
        model=model,
        model_config=model_config,
        train_X=torch.rand(8, 5, dtype=torch.double),
        train_Y=torch.zeros(8, 2, dtype=torch.long),
        metadata={"multi_output": True},
    )


def _resolve(bundle, name: str, **kwargs):
    return engine_module._resolve_objective_config_n_w_from_input_transform(
        acq_config=AcquisitionConfig(name=name, **kwargs),
        bundle=bundle,
    )


def test_ordinal_vector_strategies_infer_multi_output_objective() -> None:
    bundle = _make_bundle("ordinal", _DummyOrdinalModel())

    for name in ("ehvi", "nehvi", "nparego", "nsgaii"):
        resolved = _resolve(bundle, name)
        assert resolved.objective_config is not None
        assert resolved.objective_config.mode == "multi_output"
        assert resolved.objective_config.n_w == 16
        assert resolved.objective_config.risk_type is None


def test_multiclass_vector_strategies_infer_multi_output_objective() -> None:
    bundle = _make_bundle("multiclass", _DummyMulticlassModel())

    for name in ("ehvi", "nehvi", "nparego", "nsgaii"):
        resolved = _resolve(bundle, name)
        assert resolved.objective_config is not None
        assert resolved.objective_config.mode == "multi_output"
        assert resolved.objective_config.n_w == 16
        assert resolved.objective_config.risk_type is None


def test_score_based_strategies_remain_without_mc_objective() -> None:
    bundles = (
        _make_bundle("ordinal", _DummyOrdinalModel()),
        _make_bundle("multiclass", _DummyMulticlassModel()),
    )

    for bundle in bundles:
        for name in ("bald", "entropy", "variance", "straddle", "icu"):
            resolved = _resolve(bundle, name)
            assert resolved.objective_config is None


def test_explicit_ordinal_and_multiclass_objective_configs_are_preserved() -> None:
    bundles = (
        _make_bundle("ordinal", _DummyOrdinalModel()),
        _make_bundle("multiclass", _DummyMulticlassModel()),
    )

    for bundle in bundles:
        explicit = ObjectiveConfig(
            mode="multi_output",
            n_w=4,
            risk_type="cvar",
            alpha=0.75,
        )
        resolved = _resolve(
            bundle,
            "ehvi",
            objective_config=explicit,
        )
        assert resolved.objective_config is explicit


def test_ordinal_objective_aggregates_48_rows_to_q3() -> None:
    model = _DummyOrdinalModel()
    bundle = _make_bundle("ordinal", model)
    resolved = _resolve(bundle, "ehvi")
    objective = factory_module.build_objective(bundle=bundle, config=resolved)

    assert isinstance(objective, qMultiOutputOrdinalUtilityObjective)
    assert objective.input_perturbation_n_w == 16

    samples = torch.randn(5, 2, 48, 2, dtype=torch.double)
    X = torch.rand(2, 3, 5, dtype=torch.double)

    values = objective(samples=samples, X=X)
    unaggregated = qMultiOutputOrdinalUtilityObjective(
        model=model,
        utility_values=[
            torch.arange(3, dtype=torch.double),
            torch.arange(3, dtype=torch.double),
        ],
        input_perturbation_n_w=None,
    )(samples=samples, X=None)
    expected = unaggregated.reshape(5, 2, 3, 16, 2).mean(dim=-2)

    assert values.shape == torch.Size([5, 2, 3, 2])
    assert torch.allclose(values, expected)


def test_multiclass_objective_aggregates_48_rows_to_q3() -> None:
    bundle = _make_bundle("multiclass", _DummyMulticlassModel())
    resolved = _resolve(
        bundle,
        "ehvi",
        acqf_kwargs={"output_target_classes": [1, 2]},
    )
    objective = factory_module.build_objective(bundle=bundle, config=resolved)

    assert isinstance(objective, InputPerturbationMultiOutputObjectiveAdapter)
    assert objective.n_w == 16

    logits = torch.randn(5, 2, 48, 2, 3, dtype=torch.double)
    probabilities = torch.softmax(logits, dim=-1)
    X = torch.rand(2, 3, 5, dtype=torch.double)

    values = objective(samples=probabilities, X=X)
    selected = torch.stack(
        (probabilities[..., 0, 1], probabilities[..., 1, 2]),
        dim=-1,
    )
    expected = selected.reshape(5, 2, 3, 16, 2).mean(dim=-2)

    assert values.shape == torch.Size([5, 2, 3, 2])
    assert torch.allclose(values, expected)
    assert torch.allclose(objective(samples=probabilities, X=None), expected)


def test_hetero_ordinal_aggregates_after_utility_adjustment(monkeypatch) -> None:
    model = _DummyOrdinalModel()
    base_objective = qMultiOutputOrdinalUtilityObjective(
        model=model,
        utility_values=[
            torch.arange(3, dtype=torch.double),
            torch.arange(3, dtype=torch.double),
        ],
        input_perturbation_n_w=16,
        risk_type=None,
    )
    objective = hetero_ordinal_module.qHeteroMultiOutputOrdinalUtilityObjective(
        model=model,
        base_objective=base_objective,
        beta=1.0,
        noise_penalty=0.0,
    )

    def fake_stack_multi_summaries(model, X, **kwargs):
        del model, kwargs
        batch_shape = X.shape[:-2]
        q_expanded = int(X.shape[-2]) * 16
        robust_mean = torch.zeros(
            *batch_shape,
            q_expanded,
            2,
            dtype=X.dtype,
            device=X.device,
        )
        return {
            "robust_mean": robust_mean,
            "sigma": torch.zeros_like(robust_mean),
        }

    monkeypatch.setattr(
        hetero_ordinal_module,
        "stack_multi_summaries",
        fake_stack_multi_summaries,
    )

    samples = torch.randn(5, 2, 48, 2, dtype=torch.double)
    X = torch.rand(2, 3, 5, dtype=torch.double)

    values = objective(samples=samples, X=X)
    expected = base_objective(samples=samples, X=X)

    assert getattr(
        objective.__class__,
        "_bochan_input_perturbation_patched",
        False,
    )
    assert values.shape == torch.Size([5, 2, 3, 2])
    assert torch.allclose(values, expected)
