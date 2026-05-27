from __future__ import annotations

"""Ordinal DeepGP single-output smoke tests.

This mirrors ``test_ordinal_base_single_output.py`` while swapping only the model
family, fit helper, and DeepGP-specific train-data assertions.
"""

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from bochan.models.ordinal.deep.deepgp import (
    OrdinalDeepGPModel,
    OrdinalMixedDeepGPModel,
    fit_true_deep_ordinal_gp,
)
from tests.test_binary_classification_base_single_output import (
    DTYPE,
    DEVICE,
    assert_candidates_in_bounds,
    assert_optimizer_compatibility_result,
    make_random_batch,
    maybe_suppress_botorch_initial_warnings,
    optimize_mixed_with_case,
    optimize_with_case,
)
from tests.test_ordinal_base_single_output import (
    NUM_CLASSES,
    UTILITY_VALUES,
    _build_input_transform,
    _ordinal_optimizer_constraint_scenarios,
    _representative_ordinal_acquisition_cases,
    make_ordinal_toy_data,
    ordinal_acquisition_cases,
)


def make_ordinal_deepgp_toy_data(
    n: int = 16,
    d: int = 5,
    cat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepGP ordinal 用 toy data。mixed のカテゴリは 0..2 にする。"""
    if not cat:
        return make_ordinal_toy_data(n=n, d=d, cat=False, num_classes=NUM_CLASSES)

    torch.manual_seed(0)
    cont_x = torch.rand(n, d, device=DEVICE, dtype=DTYPE)
    cat_x = torch.randint(0, 3, (n, 1), device=DEVICE).to(dtype=DTYPE)
    train_x = torch.cat([cont_x, cat_x], dim=-1)

    score = (
        0.9 * cont_x[..., 0]
        - 0.6 * cont_x[..., 1]
        + 0.5 * cont_x[..., 2 % d]
        + torch.sin(2.0 * cont_x[..., 3 % d])
        + 0.15 * cat_x.squeeze(-1)
    )
    q1, q2 = torch.quantile(
        score,
        torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=score.dtype, device=score.device),
    )
    train_y = torch.zeros_like(score, dtype=torch.long)
    train_y = train_y + (score > q1).long() + (score > q2).long()
    train_y[0] = 0
    train_y[1] = 1
    train_y[2] = 2

    bounds = torch.zeros(2, train_x.shape[-1], device=DEVICE, dtype=DTYPE)
    bounds[1, :d] = 1.0
    bounds[1, -1] = 2.0
    return train_x, train_y, bounds


def make_random_ordinal_mixed_batch(
    bounds: torch.Tensor,
    cat_dims: list[int],
    *,
    batch_size: int = 4,
    q: int = 2,
) -> torch.Tensor:
    """ordinal deep mixed 用 candidate。カテゴリ値は 0, 1, 2。"""
    X = make_random_batch(bounds=bounds, batch_size=batch_size, q=q)
    cat_values = torch.tensor([0.0, 1.0, 2.0], dtype=X.dtype, device=X.device)
    for cat_id in cat_dims:
        random_idx = torch.randint(0, len(cat_values), X.shape[:-1], device=X.device)
        X[..., cat_id] = cat_values[random_idx]
    return X


def _fit_deepgp_ordinal_model(model: Any, *, num_epochs: int, lr: float = 0.01) -> None:
    fit_true_deep_ordinal_gp(model, num_epochs=num_epochs, lr=lr)


def _expected_transformed_x(model: Any, train_x: torch.Tensor) -> torch.Tensor:
    expected = model.input_transform(train_x) if getattr(model, "input_transform", None) is not None else train_x
    if isinstance(expected, tuple):
        expected = expected[0]
    return expected


def _assert_deepgp_ordinal_model_training(
    model: Any,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
) -> None:
    model.eval()
    expected_x = _expected_transformed_x(model, train_x)

    assert model.num_outputs == 1
    assert model.num_classes == NUM_CLASSES
    assert model.train_inputs[0].shape == train_x.shape
    assert model.train_inputs_raw[0].shape == train_x.shape
    assert torch.allclose(model.train_inputs[0], train_x)
    assert torch.allclose(model.train_inputs_raw[0], train_x)
    assert model.train_inputs_raw[0].data_ptr() != train_x.data_ptr()
    assert torch.equal(model.train_targets, train_y.long())

    with torch.no_grad():
        transformed_x = model.transform_inputs(train_x)
        posterior = model.posterior(train_x)
        probs = model.class_probs(train_x)
        pred_class = model.predict_class(train_x)
        expected_u = model.expected_utility(
            train_x,
            UTILITY_VALUES.to(device=train_x.device, dtype=train_x.dtype),
        )

    assert transformed_x.shape == expected_x.shape
    assert torch.allclose(transformed_x, expected_x)
    assert posterior.mean.shape[-2] == train_x.shape[-2]
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert probs.shape == torch.Size([train_x.shape[-2], NUM_CLASSES])
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(train_x.shape[-2], dtype=probs.dtype, device=probs.device))
    assert pred_class.shape == torch.Size([train_x.shape[-2]])
    assert torch.isin(pred_class, torch.arange(NUM_CLASSES, device=pred_class.device)).all()
    assert expected_u.shape == torch.Size([train_x.shape[-2]])
    assert torch.isfinite(expected_u).all()

    cutpoints = model.ordinal_likelihood.cutpoints
    assert cutpoints.shape == torch.Size([NUM_CLASSES - 1])
    assert torch.isfinite(cutpoints).all()
    assert torch.all(cutpoints[1:] > cutpoints[:-1])

    mll = model.make_mll()
    assert hasattr(mll, "mll") or hasattr(mll, "base_mll") or hasattr(mll, "model")

    if cat_dims:
        assert isinstance(model, OrdinalMixedDeepGPModel)
        assert list(model.cat_dims) == cat_dims
        cat_values = torch.tensor([0.0, 1.0, 2.0], dtype=train_x.dtype, device=train_x.device)
        for cat_id in cat_dims:
            assert torch.isin(model.train_inputs[0][:, cat_id], cat_values).all()
            assert torch.isin(model.train_inputs_raw[0][:, cat_id], cat_values).all()
            assert torch.isin(transformed_x[:, cat_id], cat_values).all()


def create_ordinal_deepgp_model_bundle(
    *,
    cat: bool = False,
    n: int = 16,
    d: int = 5,
    num_epochs: int = 4,
) -> dict[str, Any]:
    train_x, train_y, bounds = make_ordinal_deepgp_toy_data(n=n, d=d, cat=cat)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    input_transform = _build_input_transform(train_x, bounds, cat_dims)

    torch.manual_seed(0)
    if cat:
        model = OrdinalMixedDeepGPModel(
            train_X=train_x,
            train_Y=train_y,
            num_classes=NUM_CLASSES,
            cat_dims=cat_dims,
            input_transform=input_transform,
            list_hidden_dims=[4],
            num_inducing=8,
            lr=0.01,
            num_epochs=num_epochs,
            conditioning_steps=4,
        )
    else:
        model = OrdinalDeepGPModel(
            train_X=train_x,
            train_Y=train_y,
            num_classes=NUM_CLASSES,
            input_transform=input_transform,
            list_hidden_dims=[4],
            num_inducing=8,
            lr=0.01,
            num_epochs=num_epochs,
            conditioning_steps=4,
        )

    _fit_deepgp_ordinal_model(model, num_epochs=num_epochs, lr=0.01)
    _assert_deepgp_ordinal_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def ordinal_deepgp_model_bundle() -> dict[str, Any]:
    return create_ordinal_deepgp_model_bundle(cat=False)


@pytest.fixture(scope="module")
def ordinal_deepgp_mixed_model_bundle() -> dict[str, Any]:
    return create_ordinal_deepgp_model_bundle(cat=True)


def test_ordinal_deepgp_model_basic_behavior(ordinal_deepgp_model_bundle: dict[str, Any]) -> None:
    _assert_deepgp_ordinal_model_training(
        model=ordinal_deepgp_model_bundle["model"],
        train_x=ordinal_deepgp_model_bundle["train_x"],
        train_y=ordinal_deepgp_model_bundle["train_y"],
        cat_dims=ordinal_deepgp_model_bundle["cat_dims"],
    )


def test_ordinal_deepgp_mixed_model_basic_behavior(ordinal_deepgp_mixed_model_bundle: dict[str, Any]) -> None:
    _assert_deepgp_ordinal_model_training(
        model=ordinal_deepgp_mixed_model_bundle["model"],
        train_x=ordinal_deepgp_mixed_model_bundle["train_x"],
        train_y=ordinal_deepgp_mixed_model_bundle["train_y"],
        cat_dims=ordinal_deepgp_mixed_model_bundle["cat_dims"],
    )


def test_ordinal_deepgp_acquisition_forward_shapes(ordinal_deepgp_model_bundle: dict[str, Any]) -> None:
    model = ordinal_deepgp_model_bundle["model"]
    train_x = ordinal_deepgp_model_bundle["train_x"]
    X = make_random_batch(ordinal_deepgp_model_bundle["bounds"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in ordinal_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_ordinal_deepgp_mixed_acquisition_forward_shapes(ordinal_deepgp_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_deepgp_mixed_model_bundle["model"]
    train_x = ordinal_deepgp_mixed_model_bundle["train_x"]
    X = make_random_ordinal_mixed_batch(ordinal_deepgp_mixed_model_bundle["bounds"], ordinal_deepgp_mixed_model_bundle["cat_dims"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in ordinal_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


@pytest.mark.slow
def test_ordinal_deepgp_optimize_acqf_representative_smoke(ordinal_deepgp_model_bundle: dict[str, Any]) -> None:
    model = ordinal_deepgp_model_bundle["model"]
    train_x = ordinal_deepgp_model_bundle["train_x"]
    bounds = ordinal_deepgp_model_bundle["bounds"]
    for acq_cls, kwargs, case_id in _representative_ordinal_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf(acq_function=acq_cls(model=model, **kwargs), bounds=bounds, q=2, sequential=True, num_restarts=2, raw_samples=16, options={"maxiter": 10})
        assert cands.shape == torch.Size([2, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)


@pytest.mark.slow
def test_ordinal_deepgp_mixed_optimize_acqf_mixed_representative_smoke(ordinal_deepgp_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_deepgp_mixed_model_bundle["model"]
    train_x = ordinal_deepgp_mixed_model_bundle["train_x"]
    bounds = ordinal_deepgp_mixed_model_bundle["bounds"]
    cat_id = ordinal_deepgp_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 0.0}, {cat_id: 1.0}, {cat_id: 2.0}]
    cat_values = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)
    for acq_cls, kwargs, case_id in _representative_ordinal_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf_mixed(acq_function=acq_cls(model=model, **kwargs), bounds=bounds, q=2, fixed_features_list=fixed_features_list, num_restarts=2, raw_samples=16, options={"maxiter": 10})
        assert cands.shape == torch.Size([2, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id


@pytest.mark.slow
def test_ordinal_deepgp_optimizer_constraint_case_smoke(ordinal_deepgp_model_bundle: dict[str, Any]) -> None:
    model = ordinal_deepgp_model_bundle["model"]
    train_x = ordinal_deepgp_model_bundle["train_x"]
    bounds = ordinal_deepgp_model_bundle["bounds"]
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _ordinal_optimizer_constraint_scenarios(model, train_x, bounds):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_with_case(acqf=acq_cls(model=model, **kwargs), bounds=bounds, q=2, optimize_func=optimize_func, optimize_method=optimize_method, constraint_case=constraint_case, num_restarts=2, raw_samples=16, maxiter=10)
        assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=2, d=train_x.shape[-1], constraint_case=constraint_case, case_id=case_id)


@pytest.mark.slow
def test_ordinal_deepgp_mixed_optimizer_constraint_case_smoke(ordinal_deepgp_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_deepgp_mixed_model_bundle["model"]
    train_x = ordinal_deepgp_mixed_model_bundle["train_x"]
    bounds = ordinal_deepgp_mixed_model_bundle["bounds"]
    cat_id = ordinal_deepgp_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 0.0}, {cat_id: 1.0}, {cat_id: 2.0}]
    cat_values = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _ordinal_optimizer_constraint_scenarios(model, train_x, bounds, mixed=True):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_mixed_with_case(acqf=acq_cls(model=model, **kwargs), bounds=bounds, q=2, fixed_features_list=fixed_features_list, optimize_func=optimize_func, optimize_method=optimize_method, constraint_case=constraint_case, num_restarts=2, raw_samples=16, maxiter=10)
        assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=2, d=train_x.shape[-1], constraint_case=constraint_case, case_id=case_id)
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id
