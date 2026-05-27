from __future__ import annotations

"""Binary classification SAAS multi-output smoke tests."""

from typing import Any

import pytest
import torch
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from bochan.models.classification.binary.base import MultiOutputBinaryClassificationModel
from bochan.models.classification.binary.high_dim import (
    SaasBinaryClassificationGPModel,
    SaasBinaryClassificationMixedGPModel,
)
from tests._binary_classification_multi_output_variant_utils import assert_multi_output_wrapper_training
from tests.test_binary_classification_base_multi_output import (
    N_OUTPUTS,
    _build_input_transform,
    _optimizer_constraint_scenarios,
    _representative_multi_output_acquisition_cases,
    make_multi_output_binary_toy_data,
    multi_output_acquisition_cases,
)
from tests.test_binary_classification_base_single_output import (
    DTYPE,
    DEVICE,
    assert_candidates_in_bounds,
    assert_optimizer_compatibility_result,
    make_random_batch,
    make_random_mixed_batch,
    maybe_suppress_botorch_initial_warnings,
    optimize_mixed_with_case,
    optimize_with_case,
)
from tests.test_binary_classification_saas_single_output import (
    _assert_saas_model_training,
    _fit_saas_binary_model,
)


def create_saas_multi_output_binary_model_bundle(
    *,
    cat: bool = False,
    n: int = 16,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
) -> dict[str, Any]:
    train_x, train_y, bounds = make_multi_output_binary_toy_data(n=n, d=d, cat=cat, m=m)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    model_cls = SaasBinaryClassificationMixedGPModel if cat else SaasBinaryClassificationGPModel

    models: list[Any] = []
    for j in range(train_y.shape[-1]):
        kwargs: dict[str, Any] = {
            "train_X": train_x,
            "train_Y": train_y[:, [j]],
            "input_transform": _build_input_transform(train_x, bounds, cat_dims),
            "num_inducing_points": 8,
        }
        if cat:
            kwargs["cat_dims"] = cat_dims
        torch.manual_seed(j)
        submodel = model_cls(**kwargs)
        _fit_saas_binary_model(submodel, num_epochs=num_epochs, lr=0.01)
        models.append(submodel)

    model = MultiOutputBinaryClassificationModel(*models)
    assert_multi_output_wrapper_training(
        model=model,
        train_x=train_x,
        train_y=train_y,
        cat_dims=cat_dims,
        submodel_assert_fn=_assert_saas_model_training,
    )
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def saas_multi_output_binary_model_bundle() -> dict[str, Any]:
    return create_saas_multi_output_binary_model_bundle(cat=False)


@pytest.fixture(scope="module")
def saas_multi_output_binary_mixed_model_bundle() -> dict[str, Any]:
    return create_saas_multi_output_binary_model_bundle(cat=True)


def test_saas_multi_output_binary_model_basic_behavior(saas_multi_output_binary_model_bundle: dict[str, Any]) -> None:
    assert_multi_output_wrapper_training(
        model=saas_multi_output_binary_model_bundle["model"],
        train_x=saas_multi_output_binary_model_bundle["train_x"],
        train_y=saas_multi_output_binary_model_bundle["train_y"],
        cat_dims=saas_multi_output_binary_model_bundle["cat_dims"],
        submodel_assert_fn=_assert_saas_model_training,
    )


def test_saas_multi_output_binary_mixed_model_basic_behavior(saas_multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    assert_multi_output_wrapper_training(
        model=saas_multi_output_binary_mixed_model_bundle["model"],
        train_x=saas_multi_output_binary_mixed_model_bundle["train_x"],
        train_y=saas_multi_output_binary_mixed_model_bundle["train_y"],
        cat_dims=saas_multi_output_binary_mixed_model_bundle["cat_dims"],
        submodel_assert_fn=_assert_saas_model_training,
    )


def test_saas_multi_output_binary_acquisition_forward_shapes(saas_multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = saas_multi_output_binary_model_bundle["model"]
    train_x = saas_multi_output_binary_model_bundle["train_x"]
    X = make_random_batch(saas_multi_output_binary_model_bundle["bounds"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in multi_output_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_saas_multi_output_binary_mixed_acquisition_forward_shapes(saas_multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    model = saas_multi_output_binary_mixed_model_bundle["model"]
    train_x = saas_multi_output_binary_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(saas_multi_output_binary_mixed_model_bundle["bounds"], saas_multi_output_binary_mixed_model_bundle["cat_dims"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in multi_output_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


@pytest.mark.slow
def test_saas_multi_output_binary_optimize_acqf_representative_smoke(saas_multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = saas_multi_output_binary_model_bundle["model"]
    train_x = saas_multi_output_binary_model_bundle["train_x"]
    bounds = saas_multi_output_binary_model_bundle["bounds"]
    for acq_cls, kwargs, case_id in _representative_multi_output_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf(acq_function=acq_cls(model=model, **kwargs), bounds=bounds, q=2, sequential=True, num_restarts=2, raw_samples=16, options={"maxiter": 10})
        assert cands.shape == torch.Size([2, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)


@pytest.mark.slow
def test_saas_multi_output_binary_mixed_optimize_acqf_mixed_representative_smoke(saas_multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    model = saas_multi_output_binary_mixed_model_bundle["model"]
    train_x = saas_multi_output_binary_mixed_model_bundle["train_x"]
    bounds = saas_multi_output_binary_mixed_model_bundle["bounds"]
    cat_id = saas_multi_output_binary_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    for acq_cls, kwargs, case_id in _representative_multi_output_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf_mixed(acq_function=acq_cls(model=model, **kwargs), bounds=bounds, q=2, fixed_features_list=fixed_features_list, num_restarts=2, raw_samples=16, options={"maxiter": 10})
        assert cands.shape == torch.Size([2, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id


@pytest.mark.slow
def test_saas_multi_output_binary_optimizer_constraint_case_smoke(saas_multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = saas_multi_output_binary_model_bundle["model"]
    train_x = saas_multi_output_binary_model_bundle["train_x"]
    bounds = saas_multi_output_binary_model_bundle["bounds"]
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _optimizer_constraint_scenarios(model, train_x, bounds):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_with_case(acqf=acq_cls(model=model, **kwargs), bounds=bounds, q=2, optimize_func=optimize_func, optimize_method=optimize_method, constraint_case=constraint_case, num_restarts=2, raw_samples=16, maxiter=10)
        assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=2, d=train_x.shape[-1], constraint_case=constraint_case, case_id=case_id)


@pytest.mark.slow
def test_saas_multi_output_binary_mixed_optimizer_constraint_case_smoke(saas_multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    model = saas_multi_output_binary_mixed_model_bundle["model"]
    train_x = saas_multi_output_binary_mixed_model_bundle["train_x"]
    bounds = saas_multi_output_binary_mixed_model_bundle["bounds"]
    cat_id = saas_multi_output_binary_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _optimizer_constraint_scenarios(model, train_x, bounds, mixed=True):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_mixed_with_case(acqf=acq_cls(model=model, **kwargs), bounds=bounds, q=2, fixed_features_list=fixed_features_list, optimize_func=optimize_func, optimize_method=optimize_method, constraint_case=constraint_case, num_restarts=2, raw_samples=16, maxiter=10)
        assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=2, d=train_x.shape[-1], constraint_case=constraint_case, case_id=case_id)
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id
