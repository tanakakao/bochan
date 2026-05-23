from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from tests.test_binary_classification_base_single_output import (
    maybe_suppress_botorch_initial_warnings,
    optimizer_constraint_smoke_scenarios,
    optimize_with_case,
    optimize_mixed_with_case,
    assert_optimizer_compatibility_result,
)

from bochan.models.classification.binary.high_dim import (
    SaasBinaryClassificationGPModel,
    SaasBinaryClassificationMixedGPModel,
)
from tests.test_binary_classification_base_single_output import (
    make_binary_toy_data,
    assert_model_training,
    acquisition_cases,
    make_random_batch,
    make_random_mixed_batch,
)


def _build_input_transform(train_x: torch.Tensor, bounds: torch.Tensor, cat_dims: list[int]) -> Normalize:
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(d=train_x.shape[-1], bounds=bounds, indices=cont_indices)


def create_binary_saas_model_bundle(*, cat: bool = False) -> dict[str, Any]:
    train_x, train_y, bounds = make_binary_toy_data(cat=cat)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    kwargs: dict[str, Any] = {
        "train_X": train_x,
        "train_Y": train_y,
        "input_transform": _build_input_transform(train_x, bounds, cat_dims),
    }

    if cat:
        kwargs["cat_dims"] = cat_dims
        model = SaasBinaryClassificationMixedGPModel(**kwargs)
    else:
        model = SaasBinaryClassificationGPModel(**kwargs)

    assert_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def binary_saas_model_bundle() -> dict[str, Any]:
    return create_binary_saas_model_bundle(cat=False)


@pytest.fixture(scope="module")
def binary_saas_mixed_model_bundle() -> dict[str, Any]:
    return create_binary_saas_model_bundle(cat=True)


def test_binary_saas_acquisition_forward_shapes(binary_saas_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_model_bundle["model"]
    train_x = binary_saas_model_bundle["train_x"]
    X = make_random_batch(binary_saas_model_bundle["bounds"], batch_size=4, q=2)

    for acq_cls, kwargs, _ in acquisition_cases(model=model, train_x=train_x):
        acq = acq_cls(model=model, **kwargs)
        out = acq(X)
        assert out.shape == torch.Size([4])
        assert torch.isfinite(out).all()


def test_binary_saas_mixed_acquisition_forward_shapes(binary_saas_mixed_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_mixed_model_bundle["model"]
    train_x = binary_saas_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(binary_saas_mixed_model_bundle["bounds"], binary_saas_mixed_model_bundle["cat_dims"], batch_size=4, q=2)

    for acq_cls, kwargs, _ in acquisition_cases(model=model, train_x=train_x):
        acq = acq_cls(model=model, **kwargs)
        out = acq(X)
        assert out.shape == torch.Size([4])
        assert torch.isfinite(out).all()


def run_jupyter_all_checks(*, num_epochs: int = 0) -> dict[str, Any]:
    """Jupyter 向け: single / mixed の主要 forward check をまとめて実行する。"""
    _ = num_epochs  # API 互換用（本テストでは学習ループ未使用）

    single_bundle = create_binary_saas_model_bundle(cat=False)
    mixed_bundle = create_binary_saas_model_bundle(cat=True)

    test_binary_saas_acquisition_forward_shapes(single_bundle)
    test_binary_saas_mixed_acquisition_forward_shapes(mixed_bundle)

    return {"single": single_bundle, "mixed": mixed_bundle}


@pytest.mark.slow
def test_binary_saas_acquisition_optimize_acqf_smoke(binary_saas_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_model_bundle["model"]
    train_x = binary_saas_model_bundle["train_x"]
    bounds = binary_saas_model_bundle["bounds"]
    q = 2
    d = train_x.shape[-1]
    with maybe_suppress_botorch_initial_warnings():
        for acq_cls, kwargs, _ in acquisition_cases(model=model, train_x=train_x):
            cands, acq_value = optimize_acqf(acq_function=acq_cls(model=model, **kwargs), bounds=bounds, q=q, sequential=True, num_restarts=3, raw_samples=16, options={"maxiter": 10})
            assert cands.shape == torch.Size([q, d])
            assert torch.isfinite(cands).all()
            assert torch.isfinite(acq_value).all()


@pytest.mark.slow
def test_binary_saas_mixed_acquisition_optimize_acqf_smoke(binary_saas_mixed_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_mixed_model_bundle["model"]
    train_x = binary_saas_mixed_model_bundle["train_x"]
    bounds = binary_saas_mixed_model_bundle["bounds"]
    cat_id = binary_saas_mixed_model_bundle["cat_dims"][0]
    q = 2
    d = train_x.shape[-1]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    with maybe_suppress_botorch_initial_warnings():
        for acq_cls, kwargs, _ in acquisition_cases(model=model, train_x=train_x):
            cands, acq_value = optimize_acqf_mixed(acq_function=acq_cls(model=model, **kwargs), bounds=bounds, q=q, fixed_features_list=fixed_features_list, num_restarts=3, raw_samples=16, options={"maxiter": 10})
            assert cands.shape == torch.Size([q, d])
            assert torch.isfinite(cands).all()
            assert torch.isfinite(acq_value).all()


@pytest.mark.slow
def test_binary_saas_optimizer_constraint_compatibility_smoke() -> None:
    bundle = create_binary_saas_model_bundle(cat=False)
    model, train_x, bounds = bundle["model"], bundle["train_x"], bundle["bounds"]
    q, d = 2, train_x.shape[-1]
    for acq_cls, acq_kwargs, _, opt_func, opt_method, ccase, cid in optimizer_constraint_smoke_scenarios(model, train_x, bounds):
        cands, acq_value = optimize_with_case(acqf=acq_cls(model=model, **acq_kwargs), bounds=bounds, q=q, optimize_func=opt_func, optimize_method=opt_method, constraint_case=ccase, num_restarts=3, raw_samples=16, maxiter=10)
        assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=q, d=d, constraint_case=ccase, case_id=cid)


@pytest.mark.slow
def test_binary_saas_mixed_optimizer_constraint_compatibility_smoke() -> None:
    bundle = create_binary_saas_model_bundle(cat=True)
    model, train_x, bounds = bundle["model"], bundle["train_x"], bundle["bounds"]
    q, d = 2, train_x.shape[-1]
    cat_id = bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    for acq_cls, acq_kwargs, _, opt_func, opt_method, ccase, cid in optimizer_constraint_smoke_scenarios(model, train_x, bounds):
        cands, acq_value = optimize_mixed_with_case(acqf=acq_cls(model=model, **acq_kwargs), bounds=bounds, fixed_features_list=fixed_features_list, q=q, optimize_func=opt_func, optimize_method=opt_method, constraint_case=ccase, num_restarts=3, raw_samples=16, maxiter=10)
        assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=q, d=d, constraint_case=ccase, case_id=cid)
