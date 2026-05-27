from __future__ import annotations

"""Ordinal MAP-SAAS single-output smoke tests.

This mirrors ``test_ordinal_base_single_output.py`` while adding SAAS-specific
checks for the latent kernel prior and, for mixed models, raw-space public inputs
plus one-hot encoded internal training inputs.
"""

from typing import Any

import pytest
import torch
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from bochan.fit.ordinal import fit_ordinal_mll, make_ordinal_mll
from bochan.models.ordinal.high_dim import SaasOrdinalGPModel, SaasOrdinalMixedGPModel
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
from tests.test_ordinal_base_single_output import (
    NUM_CLASSES,
    UTILITY_VALUES,
    _build_input_transform,
    _ordinal_optimizer_constraint_scenarios,
    _representative_ordinal_acquisition_cases,
    make_ordinal_toy_data,
    ordinal_acquisition_cases,
)


def _fit_saas_ordinal_model(model: Any, *, num_epochs: int, lr: float = 0.03) -> None:
    """MAP-SAAS ordinal model を ordinal MLL で軽量 fit する。"""
    mll = make_ordinal_mll(model)
    fit_ordinal_mll(mll, fit_model=model, num_epochs=num_epochs, lr=lr)


def _base_kernel(model: Any) -> Any:
    covar = getattr(model.model, "covar_module", None)
    return getattr(covar, "base_kernel", covar)


def _has_saas_prior(model: Any) -> bool:
    base_kernel = _base_kernel(model)
    try:
        prior_names = [name for name, *_ in base_kernel.named_priors()]
    except Exception:
        prior_names = []
    return any("saas" in str(name).lower() or "tau" in str(name).lower() for name in prior_names)


def _expected_transformed_x(model: Any, train_x: torch.Tensor) -> torch.Tensor:
    expected = model.input_transform(train_x) if getattr(model, "input_transform", None) is not None else train_x
    if isinstance(expected, tuple):
        expected = expected[0]
    return expected


def _assert_saas_ordinal_model_training(
    model: Any,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
) -> None:
    """Ordinal SAAS wrapper の raw / transformed / encoded training contract を確認する。"""
    model.eval()

    assert model.num_outputs == 1
    assert model.num_classes == NUM_CLASSES
    assert model.train_inputs[0].shape == train_x.shape
    assert model.train_inputs_raw[0].shape == train_x.shape
    assert model.train_inputs[0].dtype == train_x.dtype
    assert model.train_inputs_raw[0].dtype == train_x.dtype
    assert model.train_inputs[0].device == train_x.device
    assert model.train_inputs_raw[0].device == train_x.device
    assert torch.allclose(model.train_inputs[0], train_x)
    assert torch.allclose(model.train_inputs_raw[0], train_x)
    assert model.train_inputs_raw[0].data_ptr() != train_x.data_ptr()
    assert torch.equal(model.train_targets, train_y.long())

    mll = make_ordinal_mll(model)
    assert mll.model is model.model
    assert mll.likelihood is model.likelihood
    assert mll.num_data == model.model.train_inputs[0].shape[-2]
    assert _has_saas_prior(model), "SAAS prior が latent kernel に付与されていません。"

    with torch.no_grad():
        transformed_x = model.transform_inputs(train_x)
        posterior = model.posterior(train_x)
        probs = model.class_probs(train_x)
        pred_class = model.predict_class(train_x)
        expected_u = model.expected_utility(
            train_x,
            UTILITY_VALUES.to(device=train_x.device, dtype=train_x.dtype),
        )

    assert posterior.mean.shape[-2] == train_x.shape[-2]
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert probs.shape == torch.Size([train_x.shape[-2], NUM_CLASSES])
    assert torch.isfinite(probs).all()
    assert torch.allclose(
        probs.sum(dim=-1),
        torch.ones(train_x.shape[-2], dtype=probs.dtype, device=probs.device),
    )
    assert pred_class.shape == torch.Size([train_x.shape[-2]])
    assert torch.isin(pred_class, torch.arange(NUM_CLASSES, device=pred_class.device)).all()
    assert expected_u.shape == torch.Size([train_x.shape[-2]])
    assert torch.isfinite(expected_u).all()

    cutpoints = model.ordinal_likelihood.cutpoints
    assert cutpoints.shape == torch.Size([NUM_CLASSES - 1])
    assert torch.isfinite(cutpoints).all()
    assert torch.all(cutpoints[1:] > cutpoints[:-1])

    if cat_dims:
        assert isinstance(model, SaasOrdinalMixedGPModel)
        assert list(model.cat_dims) == cat_dims
        assert model.raw_dim == train_x.shape[-1]
        assert model.encoded_dim > model.raw_dim
        assert model.encoded_train_input_raw.shape == torch.Size([train_x.shape[0], model.encoded_dim])
        assert transformed_x.shape == torch.Size([train_x.shape[0], model.encoded_dim])
        assert model.model.train_inputs[0].shape == torch.Size([train_x.shape[0], model.encoded_dim])
        assert torch.allclose(model.model.train_inputs[0], model.transform_inputs(train_x))
        assert torch.equal(model.model.train_targets, train_y.long())

        decoded_train_x = model.decode_inputs(model.encoded_train_input_raw)
        assert decoded_train_x.shape == train_x.shape
        assert torch.allclose(decoded_train_x, train_x)

        cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=train_x.dtype, device=train_x.device)
        for cat_id in cat_dims:
            assert torch.isin(model.train_inputs[0][:, cat_id], cat_values).all()
            assert torch.isin(model.train_inputs_raw[0][:, cat_id], cat_values).all()
            encoded_block = model.encoded_cat_dims[cat_id]
            one_hot = model.encoded_train_input_raw[:, encoded_block]
            assert torch.allclose(
                one_hot.sum(dim=-1),
                torch.ones(train_x.shape[0], dtype=train_x.dtype, device=train_x.device),
            )
    else:
        expected_inner_x = _expected_transformed_x(model, train_x)
        assert transformed_x.shape == train_x.shape
        assert torch.allclose(transformed_x, expected_inner_x)
        assert model.model.train_inputs[0].shape == expected_inner_x.shape
        assert torch.allclose(model.model.train_inputs[0], expected_inner_x)
        assert torch.equal(model.model.train_targets, train_y.long())


def create_ordinal_saas_model_bundle(
    *,
    cat: bool = False,
    n: int = 16,
    d: int = 5,
    num_epochs: int = 4,
) -> dict[str, Any]:
    train_x, train_y, bounds = make_ordinal_toy_data(n=n, d=d, cat=cat, num_classes=NUM_CLASSES)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    input_transform = _build_input_transform(train_x, bounds, cat_dims)

    torch.manual_seed(0)
    if cat:
        model = SaasOrdinalMixedGPModel(
            train_X=train_x,
            train_Y=train_y,
            num_classes=NUM_CLASSES,
            cat_dims=cat_dims,
            input_transform=input_transform,
            num_inducing_points=8,
        )
    else:
        model = SaasOrdinalGPModel(
            train_X=train_x,
            train_Y=train_y,
            num_classes=NUM_CLASSES,
            input_transform=input_transform,
            num_inducing_points=8,
        )

    _fit_saas_ordinal_model(model, num_epochs=num_epochs, lr=0.03)
    _assert_saas_ordinal_model_training(
        model=model,
        train_x=train_x,
        train_y=train_y,
        cat_dims=cat_dims,
    )
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def ordinal_saas_model_bundle() -> dict[str, Any]:
    return create_ordinal_saas_model_bundle(cat=False)


@pytest.fixture(scope="module")
def ordinal_saas_mixed_model_bundle() -> dict[str, Any]:
    return create_ordinal_saas_model_bundle(cat=True)


def test_ordinal_saas_model_basic_behavior(ordinal_saas_model_bundle: dict[str, Any]) -> None:
    _assert_saas_ordinal_model_training(
        model=ordinal_saas_model_bundle["model"],
        train_x=ordinal_saas_model_bundle["train_x"],
        train_y=ordinal_saas_model_bundle["train_y"],
        cat_dims=ordinal_saas_model_bundle["cat_dims"],
    )


def test_ordinal_saas_mixed_model_basic_behavior(ordinal_saas_mixed_model_bundle: dict[str, Any]) -> None:
    _assert_saas_ordinal_model_training(
        model=ordinal_saas_mixed_model_bundle["model"],
        train_x=ordinal_saas_mixed_model_bundle["train_x"],
        train_y=ordinal_saas_mixed_model_bundle["train_y"],
        cat_dims=ordinal_saas_mixed_model_bundle["cat_dims"],
    )


def test_ordinal_saas_acquisition_forward_shapes(ordinal_saas_model_bundle: dict[str, Any]) -> None:
    model = ordinal_saas_model_bundle["model"]
    train_x = ordinal_saas_model_bundle["train_x"]
    X = make_random_batch(ordinal_saas_model_bundle["bounds"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in ordinal_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_ordinal_saas_mixed_acquisition_forward_shapes(ordinal_saas_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_saas_mixed_model_bundle["model"]
    train_x = ordinal_saas_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(
        ordinal_saas_mixed_model_bundle["bounds"],
        ordinal_saas_mixed_model_bundle["cat_dims"],
        batch_size=4,
        q=2,
    )
    for acq_cls, kwargs, case_id in ordinal_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


@pytest.mark.slow
def test_ordinal_saas_optimize_acqf_representative_smoke(ordinal_saas_model_bundle: dict[str, Any]) -> None:
    model = ordinal_saas_model_bundle["model"]
    train_x = ordinal_saas_model_bundle["train_x"]
    bounds = ordinal_saas_model_bundle["bounds"]
    for acq_cls, kwargs, case_id in _representative_ordinal_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf(
                acq_function=acq_cls(model=model, **kwargs),
                bounds=bounds,
                q=2,
                sequential=True,
                num_restarts=2,
                raw_samples=16,
                options={"maxiter": 10},
            )
        assert cands.shape == torch.Size([2, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)


@pytest.mark.slow
def test_ordinal_saas_mixed_optimize_acqf_mixed_representative_smoke(ordinal_saas_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_saas_mixed_model_bundle["model"]
    train_x = ordinal_saas_mixed_model_bundle["train_x"]
    bounds = ordinal_saas_mixed_model_bundle["bounds"]
    cat_id = ordinal_saas_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    for acq_cls, kwargs, case_id in _representative_ordinal_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf_mixed(
                acq_function=acq_cls(model=model, **kwargs),
                bounds=bounds,
                q=2,
                fixed_features_list=fixed_features_list,
                num_restarts=2,
                raw_samples=16,
                options={"maxiter": 10},
            )
        assert cands.shape == torch.Size([2, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id


@pytest.mark.slow
def test_ordinal_saas_optimizer_constraint_case_smoke(ordinal_saas_model_bundle: dict[str, Any]) -> None:
    model = ordinal_saas_model_bundle["model"]
    train_x = ordinal_saas_model_bundle["train_x"]
    bounds = ordinal_saas_model_bundle["bounds"]
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _ordinal_optimizer_constraint_scenarios(model, train_x, bounds):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_with_case(
                acqf=acq_cls(model=model, **kwargs),
                bounds=bounds,
                q=2,
                optimize_func=optimize_func,
                optimize_method=optimize_method,
                constraint_case=constraint_case,
                num_restarts=2,
                raw_samples=16,
                maxiter=10,
            )
        assert_optimizer_compatibility_result(
            cands=cands,
            acq_value=acq_value,
            bounds=bounds,
            q=2,
            d=train_x.shape[-1],
            constraint_case=constraint_case,
            case_id=case_id,
        )


@pytest.mark.slow
def test_ordinal_saas_mixed_optimizer_constraint_case_smoke(ordinal_saas_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_saas_mixed_model_bundle["model"]
    train_x = ordinal_saas_mixed_model_bundle["train_x"]
    bounds = ordinal_saas_mixed_model_bundle["bounds"]
    cat_id = ordinal_saas_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _ordinal_optimizer_constraint_scenarios(model, train_x, bounds, mixed=True):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_mixed_with_case(
                acqf=acq_cls(model=model, **kwargs),
                bounds=bounds,
                q=2,
                fixed_features_list=fixed_features_list,
                optimize_func=optimize_func,
                optimize_method=optimize_method,
                constraint_case=constraint_case,
                num_restarts=2,
                raw_samples=16,
                maxiter=10,
            )
        assert_optimizer_compatibility_result(
            cands=cands,
            acq_value=acq_value,
            bounds=bounds,
            q=2,
            d=train_x.shape[-1],
            constraint_case=constraint_case,
            case_id=case_id,
        )
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id


def run_jupyter_forward_check(
    *,
    cat: bool = False,
    n: int = 16,
    d: int = 5,
    num_epochs: int = 4,
    batch_size: int = 4,
    q: int = 2,
    verbose_forward_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_ordinal_saas_model_bundle(cat=cat, n=n, d=d, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]
    X = make_random_mixed_batch(bounds, cat_dims, batch_size=batch_size, q=q) if cat else make_random_batch(bounds, batch_size=batch_size, q=q)

    print("=" * 80)
    print(f"Jupyter ordinal SAAS forward check cat={cat}")
    if verbose_forward_detail:
        print(f"train_inputs_raw[0].shape={model.train_inputs_raw[0].shape}")
        print(f"train_inputs[0].shape={model.train_inputs[0].shape}")
        print(f"train_targets.shape={model.train_targets.shape}")
        print(f"cutpoints={model.ordinal_likelihood.cutpoints.detach().cpu().tolist()}")
        print(f"bounds.shape={bounds.shape}")
        print(f"X.shape={X.shape}")
        if cat:
            print(f"raw_dim={model.raw_dim}, encoded_dim={model.encoded_dim}")
            print(f"encoded_train_input_raw.shape={model.encoded_train_input_raw.shape}")
    print("=" * 80)

    for acq_cls, kwargs, case_id in ordinal_acquisition_cases(model, train_x):
        values = acq_cls(model=model, **kwargs)(X)
        assert values.shape == torch.Size([batch_size]), case_id
        assert torch.isfinite(values).all(), case_id
        if verbose_forward_detail:
            print(f"[OK] {case_id} shape={tuple(values.shape)} min={values.min().item():.6g} max={values.max().item():.6g}")

    print("forward check passed.")
    return bundle


def run_jupyter_all_forward_checks(*, num_epochs: int = 4, verbose_forward_detail: bool = False) -> None:
    run_jupyter_forward_check(cat=False, num_epochs=num_epochs, verbose_forward_detail=verbose_forward_detail)
    run_jupyter_forward_check(cat=True, num_epochs=num_epochs, verbose_forward_detail=verbose_forward_detail)
    print("all ordinal SAAS forward checks passed.")
