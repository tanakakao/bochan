from __future__ import annotations

"""Binary classification MAP-SAAS single-output smoke tests.

The common toy data, acquisition cases, optimizer wrappers, and constraint
assertions are imported from ``test_binary_classification_base_single_output``.
SAAS-specific checks verify raw-space public inputs, transformed / encoded inner
training inputs, and the presence of a SAAS prior on the latent kernel.
"""

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.high_dim import (
    SaasBinaryClassificationGPModel,
    SaasBinaryClassificationMixedGPModel,
)
from tests.test_binary_classification_base_single_output import (
    DTYPE,
    DEVICE,
    acquisition_cases,
    assert_candidates_in_bounds,
    assert_optimizer_compatibility_result,
    make_binary_toy_data,
    make_constraint_cases,
    make_random_batch,
    make_random_mixed_batch,
    maybe_suppress_botorch_initial_warnings,
    optimize_mixed_with_case,
    optimize_with_case,
    optimizer_cases,
)


def _build_input_transform(train_x: torch.Tensor, bounds: torch.Tensor, cat_dims: list[int]) -> Normalize:
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(d=train_x.shape[-1], bounds=bounds, indices=cont_indices)


def _fit_saas_binary_model(model: Any, *, num_epochs: int, lr: float = 0.01) -> None:
    fit_binary_classifier_mll(model.make_mll(), num_epochs=num_epochs, lr=lr)


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


def _assert_saas_model_training(
    model: Any,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
) -> None:
    model.eval()

    assert model.train_inputs[0].shape == train_x.shape
    assert model.train_inputs_raw[0].shape == train_x.shape
    assert model.train_inputs[0].dtype == train_x.dtype
    assert model.train_inputs_raw[0].dtype == train_x.dtype
    assert model.train_inputs[0].device == train_x.device
    assert model.train_inputs_raw[0].device == train_x.device
    assert torch.allclose(model.train_inputs[0], train_x)
    assert torch.allclose(model.train_inputs_raw[0], train_x)
    assert model.train_inputs_raw[0].data_ptr() != train_x.data_ptr()
    assert torch.allclose(model.train_targets, train_y.reshape(-1))

    mll = model.make_mll()
    assert mll.model is model.model
    assert mll.num_data == model.model.train_inputs[0].shape[-2]
    assert _has_saas_prior(model), "SAAS prior が latent kernel に付与されていません。"

    with torch.no_grad():
        transformed_x = model.transform_inputs(train_x)
        posterior = model.posterior(train_x)
        latent_posterior = model.latent_posterior(train_x)

    if cat_dims:
        assert isinstance(model, SaasBinaryClassificationMixedGPModel)
        assert list(model.cat_dims) == cat_dims
        assert model.raw_dim == train_x.shape[-1]
        assert model.encoded_dim > model.raw_dim
        assert model.encoded_train_input_raw.shape == torch.Size([train_x.shape[0], model.encoded_dim])
        assert transformed_x.shape == torch.Size([train_x.shape[0], model.encoded_dim])
        assert model.model.train_inputs[0].shape == torch.Size([train_x.shape[0], model.encoded_dim])
        assert torch.allclose(model.model.train_inputs[0], model.transform_inputs(train_x))

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
        expected_inner_x = model.input_transform(train_x) if model.input_transform is not None else train_x
        if isinstance(expected_inner_x, tuple):
            expected_inner_x = expected_inner_x[0]
        assert transformed_x.shape == train_x.shape
        assert torch.allclose(transformed_x, expected_inner_x)
        assert model.model.train_inputs[0].shape == expected_inner_x.shape
        assert torch.allclose(model.model.train_inputs[0], expected_inner_x)

    assert torch.allclose(model.model.train_targets, train_y.reshape(-1))
    assert posterior.mean.shape == train_y.shape
    assert posterior.variance.shape == train_y.shape
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert (posterior.mean >= 0.0).all() and (posterior.mean <= 1.0).all()
    assert torch.isfinite(latent_posterior.mean).all()


def create_binary_saas_model_bundle(
    *,
    cat: bool = False,
    n: int = 16,
    d: int = 5,
    num_epochs: int = 8,
) -> dict[str, Any]:
    train_x, train_y, bounds = make_binary_toy_data(n=n, d=d, cat=cat)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    input_transform = _build_input_transform(train_x, bounds, cat_dims)

    torch.manual_seed(0)
    if cat:
        model = SaasBinaryClassificationMixedGPModel(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
            cat_dims=cat_dims,
            num_inducing_points=8,
        )
    else:
        model = SaasBinaryClassificationGPModel(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
            num_inducing_points=8,
        )

    _fit_saas_binary_model(model, num_epochs=num_epochs, lr=0.01)
    _assert_saas_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def binary_saas_model_bundle() -> dict[str, Any]:
    return create_binary_saas_model_bundle(cat=False)


@pytest.fixture(scope="module")
def binary_saas_mixed_model_bundle() -> dict[str, Any]:
    return create_binary_saas_model_bundle(cat=True)


def _representative_acquisition_cases(model: Any, train_x: torch.Tensor):
    names = {"predictive_entropy", "latent_straddle", "pof", "binary_ei", "binary_pi", "binary_ucb"}
    return [case for case in acquisition_cases(model=model, train_x=train_x) if case[2] in names]


def _representative_constraint_cases(bounds: torch.Tensor) -> list[dict[str, Any]]:
    names = {"step_only", "step_k_sparse_constraints"}
    return [case for case in make_constraint_cases(bounds) if case["case_id"] in names]


def _get_acquisition_case(model: Any, train_x: torch.Tensor, case_id: str):
    for acq_cls, kwargs, current_case_id in acquisition_cases(model=model, train_x=train_x):
        if current_case_id == case_id:
            return acq_cls, kwargs
    raise AssertionError(f"acquisition case not found: {case_id}")


def _optimizer_constraint_scenarios(
    model: Any,
    train_x: torch.Tensor,
    bounds: torch.Tensor,
    *,
    mixed: bool = False,
    full_matrix: bool = False,
):
    if full_matrix:
        scenarios = []
        for acq_cls, kwargs, acq_id in _representative_acquisition_cases(model, train_x):
            for optimize_func, optimize_method, optimizer_id in optimizer_cases():
                for constraint_case in make_constraint_cases(bounds):
                    case_id = f"{acq_id}__{optimizer_id}__{constraint_case['case_id']}"
                    if mixed:
                        case_id = f"mixed__{case_id}"
                    scenarios.append((acq_cls, kwargs, acq_id, optimize_func, optimize_method, constraint_case, case_id))
        return scenarios

    acq_cls, kwargs = _get_acquisition_case(model, train_x, "binary_ucb")
    prefix = "saas_mixed" if mixed else "saas"
    return [
        (acq_cls, kwargs, "binary_ucb", "torch", "adam", constraint_case, f"{prefix}__binary_ucb__torch_adam__{constraint_case['case_id']}")
        for constraint_case in _representative_constraint_cases(bounds)
    ]


def test_binary_saas_model_basic_behavior(binary_saas_model_bundle: dict[str, Any]) -> None:
    _assert_saas_model_training(
        binary_saas_model_bundle["model"],
        binary_saas_model_bundle["train_x"],
        binary_saas_model_bundle["train_y"],
        cat_dims=binary_saas_model_bundle["cat_dims"],
    )


def test_binary_saas_mixed_model_basic_behavior(binary_saas_mixed_model_bundle: dict[str, Any]) -> None:
    _assert_saas_model_training(
        binary_saas_mixed_model_bundle["model"],
        binary_saas_mixed_model_bundle["train_x"],
        binary_saas_mixed_model_bundle["train_y"],
        cat_dims=binary_saas_mixed_model_bundle["cat_dims"],
    )


def test_binary_saas_acquisition_forward_shapes(binary_saas_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_model_bundle["model"]
    train_x = binary_saas_model_bundle["train_x"]
    X = make_random_batch(binary_saas_model_bundle["bounds"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in acquisition_cases(model=model, train_x=train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_binary_saas_mixed_acquisition_forward_shapes(binary_saas_mixed_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_mixed_model_bundle["model"]
    train_x = binary_saas_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(binary_saas_mixed_model_bundle["bounds"], binary_saas_mixed_model_bundle["cat_dims"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in acquisition_cases(model=model, train_x=train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


@pytest.mark.slow
def test_binary_saas_optimize_acqf_representative_smoke(binary_saas_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_model_bundle["model"]
    train_x = binary_saas_model_bundle["train_x"]
    bounds = binary_saas_model_bundle["bounds"]
    q = 2
    for acq_cls, kwargs, case_id in _representative_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf(
                acq_function=acq_cls(model=model, **kwargs),
                bounds=bounds,
                q=q,
                sequential=True,
                num_restarts=2,
                raw_samples=16,
                options={"maxiter": 10},
            )
        assert cands.shape == torch.Size([q, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)


@pytest.mark.slow
def test_binary_saas_mixed_optimize_acqf_mixed_representative_smoke(binary_saas_mixed_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_mixed_model_bundle["model"]
    train_x = binary_saas_mixed_model_bundle["train_x"]
    bounds = binary_saas_mixed_model_bundle["bounds"]
    cat_id = binary_saas_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    q = 2
    for acq_cls, kwargs, case_id in _representative_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf_mixed(
                acq_function=acq_cls(model=model, **kwargs),
                bounds=bounds,
                q=q,
                fixed_features_list=fixed_features_list,
                num_restarts=2,
                raw_samples=16,
                options={"maxiter": 10},
            )
        assert cands.shape == torch.Size([q, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id


@pytest.mark.slow
def test_binary_saas_optimizer_constraint_case_smoke(binary_saas_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_model_bundle["model"]
    train_x = binary_saas_model_bundle["train_x"]
    bounds = binary_saas_model_bundle["bounds"]
    q = 2
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _optimizer_constraint_scenarios(model, train_x, bounds):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_with_case(
                acqf=acq_cls(model=model, **kwargs),
                bounds=bounds,
                q=q,
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
            q=q,
            d=train_x.shape[-1],
            constraint_case=constraint_case,
            case_id=case_id,
        )


@pytest.mark.slow
def test_binary_saas_mixed_optimizer_constraint_case_smoke(binary_saas_mixed_model_bundle: dict[str, Any]) -> None:
    model = binary_saas_mixed_model_bundle["model"]
    train_x = binary_saas_mixed_model_bundle["train_x"]
    bounds = binary_saas_mixed_model_bundle["bounds"]
    cat_id = binary_saas_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    q = 2
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _optimizer_constraint_scenarios(model, train_x, bounds, mixed=True):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_mixed_with_case(
                acqf=acq_cls(model=model, **kwargs),
                bounds=bounds,
                q=q,
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
            q=q,
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
    num_epochs: int = 8,
    batch_size: int = 4,
    q: int = 2,
    verbose_forward_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_binary_saas_model_bundle(cat=cat, n=n, d=d, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]
    X = make_random_mixed_batch(bounds, cat_dims, batch_size=batch_size, q=q) if cat else make_random_batch(bounds, batch_size=batch_size, q=q)

    print("=" * 80)
    print(f"Jupyter SAAS forward check cat={cat}")
    if verbose_forward_detail:
        print(f"train_x.shape={train_x.shape}")
        print(f"train_inputs[0].shape={model.train_inputs[0].shape}")
        print(f"model.train_inputs[0].shape={model.model.train_inputs[0].shape}")
        if cat:
            print(f"raw_dim={model.raw_dim}, encoded_dim={model.encoded_dim}")
            print(f"encoded_train_input_raw.shape={model.encoded_train_input_raw.shape}")
        print(f"bounds.shape={bounds.shape}")
        print(f"X.shape={X.shape}")
    print("=" * 80)

    for acq_cls, kwargs, case_id in acquisition_cases(model=model, train_x=train_x):
        values = acq_cls(model=model, **kwargs)(X)
        assert values.shape == torch.Size([batch_size]), case_id
        assert torch.isfinite(values).all(), case_id
        if verbose_forward_detail:
            print(f"[OK] {case_id} shape={tuple(values.shape)} min={values.min().item():.6g} max={values.max().item():.6g}")

    print("forward check passed.")
    return bundle
