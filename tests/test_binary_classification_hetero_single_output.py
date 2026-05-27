from __future__ import annotations

"""Binary classification heteroscedastic single-output smoke tests.

This module follows ``test_binary_classification_base_single_output.py`` and
reuses its toy data, optimizer wrappers, and constraint assertions.  Unlike the
standard binary tests, acquisition checks here intentionally use only the
heteroscedastic-specific acquisition functions.
"""

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls.variational_elbo import VariationalELBO

from bochan.acquisition.binary.active_learning import (
    qHeteroBinaryBALD,
    qHeteroBinaryIntegratedPosteriorVariance,
    qHeteroBinaryMarginUncertainty,
    qHeteroBinaryPredictiveEntropy,
    qHeteroBinaryProbabilityVariance,
)
from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.robust import (
    HeteroscedasticBinaryClassificationGPModel,
    HeteroscedasticBinaryClassificationMixedGPModel,
)
from tests.test_binary_classification_base_single_output import (
    DTYPE,
    DEVICE,
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
    print_linear_constraint_diagnostics,
)


def _build_input_transform(
    train_x: torch.Tensor,
    bounds: torch.Tensor,
    cat_dims: list[int],
) -> Normalize:
    """Hetero / mixed hetero 共通の raw-space input_transform を作る。"""
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(
        d=train_x.shape[-1],
        bounds=bounds,
        indices=cont_indices,
    )


def _make_classifier_mll(model: Any) -> VariationalELBO:
    """Hetero wrapper の final classifier 用 MLL を作る。"""
    return VariationalELBO(
        likelihood=model.likelihood,
        model=model.model,
        num_data=model.model.train_inputs[0].shape[-2],
    )


def _fit_hetero_binary_model(
    model: Any,
    *,
    num_epochs: int,
    lr: float = 0.01,
) -> None:
    """Hetero binary classification の final classifier を軽量設定で fit する。"""
    fit_binary_classifier_mll(
        _make_classifier_mll(model),
        num_epochs=num_epochs,
        lr=lr,
    )


def hetero_acquisition_cases(model: Any) -> list[tuple[type, dict[str, Any], str]]:
    """Hetero 専用 acquisition のみを返す。"""
    common_kwargs: dict[str, Any] = {
        "reduction": "mean",
        "pending_penalty_weight": 0.01,
        "pending_penalty_beta": 5.0,
        "noise_mode": "inverse_linear",
        "noise_combine": "multiply",
        "noise_penalty_lambda": 0.1,
        "noise_min_weight": 0.05,
        "noise_weight_scale": 1.0,
        "noise_model_outputs_log_var": True,
    }
    return [
        (
            qHeteroBinaryPredictiveEntropy,
            dict(common_kwargs),
            "hetero_predictive_entropy",
        ),
        (
            qHeteroBinaryBALD,
            {**common_kwargs, "num_samples": 8},
            "hetero_bald",
        ),
        (
            qHeteroBinaryProbabilityVariance,
            dict(common_kwargs),
            "hetero_probability_variance",
        ),
        (
            qHeteroBinaryMarginUncertainty,
            dict(common_kwargs),
            "hetero_margin_uncertainty",
        ),
        (
            qHeteroBinaryIntegratedPosteriorVariance,
            dict(common_kwargs),
            "hetero_integrated_posterior_variance",
        ),
    ]


def _assert_hetero_model_training(
    model: Any,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
) -> None:
    """Heteroscedastic binary classification wrapper の基本状態を確認する。"""
    model.eval()

    assert model.train_inputs_raw[0].shape == train_x.shape
    assert model.train_inputs_raw[0].dtype == train_x.dtype
    assert model.train_inputs_raw[0].device == train_x.device
    assert torch.allclose(model.train_inputs_raw[0], train_x)
    assert model.train_inputs_raw[0].data_ptr() != train_x.data_ptr()
    assert torch.allclose(model.train_targets, train_y.reshape(-1))

    with torch.no_grad():
        expected_train_inputs = (
            model.input_transform(train_x)
            if getattr(model, "input_transform", None) is not None
            else train_x
        )
        if isinstance(expected_train_inputs, tuple):
            expected_train_inputs = expected_train_inputs[0]

        transformed_x = model.transform_inputs(train_x)
        posterior_without_noise = model.posterior(train_x, observation_noise=False)
        posterior_with_noise = model.posterior(train_x, observation_noise=True)
        latent_posterior = model.latent_posterior(train_x)
        noise_var = model.predict_noise_var(train_x, ref_like=train_y)
        noise_logvar = model.predict_noise_logvar(train_x, ref_like=train_y)

    assert model.train_inputs[0].shape == expected_train_inputs.shape
    assert model.model.train_inputs[0].shape == expected_train_inputs.shape
    assert transformed_x.shape == expected_train_inputs.shape
    assert torch.allclose(model.train_inputs[0], expected_train_inputs)
    assert torch.allclose(model.model.train_inputs[0], expected_train_inputs)
    assert torch.allclose(transformed_x, expected_train_inputs)
    assert torch.allclose(model.model.train_targets, train_y.reshape(-1))

    assert hasattr(model, "noise_model")
    assert hasattr(model, "noise_input_transform")
    assert model.train_Yvar.shape == train_y.shape
    assert torch.isfinite(model.train_Yvar).all()
    assert (model.train_Yvar >= model.min_noise).all()

    assert noise_var.shape == train_y.shape
    assert noise_logvar.shape == train_y.shape
    assert torch.isfinite(noise_var).all()
    assert torch.isfinite(noise_logvar).all()
    assert (noise_var >= 0.0).all()

    assert posterior_without_noise.mean.shape == train_y.shape
    assert posterior_without_noise.variance.shape == train_y.shape
    assert posterior_with_noise.mean.shape == train_y.shape
    assert posterior_with_noise.variance.shape == train_y.shape
    assert torch.isfinite(posterior_without_noise.mean).all()
    assert torch.isfinite(posterior_without_noise.variance).all()
    assert torch.isfinite(posterior_with_noise.mean).all()
    assert torch.isfinite(posterior_with_noise.variance).all()
    assert (posterior_without_noise.mean >= 0.0).all() and (posterior_without_noise.mean <= 1.0).all()
    assert (posterior_with_noise.mean >= 0.0).all() and (posterior_with_noise.mean <= 1.0).all()
    assert torch.all(posterior_with_noise.variance >= posterior_without_noise.variance)
    assert torch.isfinite(latent_posterior.mean).all()

    mll = _make_classifier_mll(model)
    assert mll.model is model.model
    assert mll.likelihood is model.likelihood
    assert mll.num_data == model.model.train_inputs[0].shape[-2]

    if cat_dims:
        assert list(model.cat_dims) == cat_dims
        assert list(model.model.cat_dims) == cat_dims
        cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=train_x.dtype, device=train_x.device)
        for cat_id in cat_dims:
            assert torch.isin(model.train_inputs_raw[0][:, cat_id], cat_values).all()
            assert torch.isin(model.train_inputs[0][:, cat_id], cat_values).all()
            assert torch.isin(model.model.train_inputs[0][:, cat_id], cat_values).all()
            assert torch.isin(transformed_x[:, cat_id], cat_values).all()


def create_binary_hetero_model_bundle(
    *,
    cat: bool = False,
    n: int = 12,
    d: int = 5,
    num_epochs: int = 8,
    aux_num_epochs: int = 4,
) -> dict[str, Any]:
    """Jupyter / pytest 共通で使う binary hetero model 作成関数。"""
    train_x, train_y, bounds = make_binary_toy_data(
        n=n,
        d=d,
        cat=cat,
    )
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    input_transform = _build_input_transform(
        train_x=train_x,
        bounds=bounds,
        cat_dims=cat_dims,
    )

    torch.manual_seed(0)
    if cat:
        model = HeteroscedasticBinaryClassificationMixedGPModel(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
            cat_dims=cat_dims,
            num_inducing_points=8,
            aux_num_epochs=aux_num_epochs,
            aux_lr=0.01,
            min_noise=1e-6,
        )
    else:
        model = HeteroscedasticBinaryClassificationGPModel(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
            num_inducing_points=8,
            aux_num_epochs=aux_num_epochs,
            aux_lr=0.01,
            min_noise=1e-6,
        )

    _fit_hetero_binary_model(
        model,
        num_epochs=num_epochs,
        lr=0.01,
    )
    _assert_hetero_model_training(
        model=model,
        train_x=train_x,
        train_y=train_y,
        cat_dims=cat_dims,
    )

    return {
        "model": model,
        "train_x": train_x,
        "train_y": train_y,
        "bounds": bounds,
        "cat_dims": cat_dims,
    }


@pytest.fixture(scope="module")
def binary_hetero_model_bundle() -> dict[str, Any]:
    """pytest 用: 通常 binary hetero model。"""
    return create_binary_hetero_model_bundle(
        cat=False,
        n=12,
        d=5,
        num_epochs=8,
        aux_num_epochs=4,
    )


@pytest.fixture(scope="module")
def binary_hetero_mixed_model_bundle() -> dict[str, Any]:
    """pytest 用: mixed binary hetero model。"""
    return create_binary_hetero_model_bundle(
        cat=True,
        n=12,
        d=5,
        num_epochs=8,
        aux_num_epochs=4,
    )


def _representative_hetero_acquisition_cases(model: Any) -> list[tuple[type, dict[str, Any], str]]:
    names = {
        "hetero_predictive_entropy",
        "hetero_bald",
        "hetero_probability_variance",
        "hetero_margin_uncertainty",
    }
    return [case for case in hetero_acquisition_cases(model) if case[2] in names]


def _representative_constraint_cases(bounds: torch.Tensor) -> list[dict[str, Any]]:
    names = {"step_only", "step_k_sparse_constraints"}
    return [
        case
        for case in make_constraint_cases(bounds)
        if case["case_id"] in names
    ]


def _get_acquisition_case(model: Any, case_id: str):
    for acq_cls, kwargs, current_case_id in hetero_acquisition_cases(model):
        if current_case_id == case_id:
            return acq_cls, kwargs
    raise AssertionError(f"hetero acquisition case not found: {case_id}")


def _optimizer_constraint_scenarios(
    model: Any,
    bounds: torch.Tensor,
    *,
    mixed: bool = False,
    full_matrix: bool = False,
):
    if full_matrix:
        scenarios = []
        for acq_cls, kwargs, acq_id in _representative_hetero_acquisition_cases(model):
            for optimize_func, optimize_method, optimizer_id in optimizer_cases():
                for constraint_case in make_constraint_cases(bounds):
                    case_id = (
                        f"{acq_id}__{optimizer_id}__"
                        f"{constraint_case['case_id']}"
                    )
                    if mixed:
                        case_id = f"mixed__{case_id}"
                    scenarios.append(
                        (
                            acq_cls,
                            kwargs,
                            acq_id,
                            optimize_func,
                            optimize_method,
                            constraint_case,
                            case_id,
                        )
                    )
        return scenarios

    acq_cls, kwargs = _get_acquisition_case(
        model=model,
        case_id="hetero_probability_variance",
    )
    prefix = "hetero_mixed" if mixed else "hetero"
    return [
        (
            acq_cls,
            kwargs,
            "hetero_probability_variance",
            "torch",
            "adam",
            constraint_case,
            f"{prefix}__probability_variance__torch_adam__{constraint_case['case_id']}",
        )
        for constraint_case in _representative_constraint_cases(bounds)
    ]


def test_binary_hetero_model_basic_behavior(
    binary_hetero_model_bundle: dict[str, Any],
) -> None:
    _assert_hetero_model_training(
        model=binary_hetero_model_bundle["model"],
        train_x=binary_hetero_model_bundle["train_x"],
        train_y=binary_hetero_model_bundle["train_y"],
        cat_dims=binary_hetero_model_bundle["cat_dims"],
    )


def test_binary_hetero_mixed_model_basic_behavior(
    binary_hetero_mixed_model_bundle: dict[str, Any],
) -> None:
    _assert_hetero_model_training(
        model=binary_hetero_mixed_model_bundle["model"],
        train_x=binary_hetero_mixed_model_bundle["train_x"],
        train_y=binary_hetero_mixed_model_bundle["train_y"],
        cat_dims=binary_hetero_mixed_model_bundle["cat_dims"],
    )


def test_binary_hetero_acquisition_forward_shapes(
    binary_hetero_model_bundle: dict[str, Any],
) -> None:
    model = binary_hetero_model_bundle["model"]
    X = make_random_batch(
        binary_hetero_model_bundle["bounds"],
        batch_size=4,
        q=2,
    )

    for acq_cls, kwargs, case_id in hetero_acquisition_cases(model):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_binary_hetero_mixed_acquisition_forward_shapes(
    binary_hetero_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_hetero_mixed_model_bundle["model"]
    X = make_random_mixed_batch(
        binary_hetero_mixed_model_bundle["bounds"],
        binary_hetero_mixed_model_bundle["cat_dims"],
        batch_size=4,
        q=2,
    )

    for acq_cls, kwargs, case_id in hetero_acquisition_cases(model):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


@pytest.mark.slow
def test_binary_hetero_optimize_acqf_representative_smoke(
    binary_hetero_model_bundle: dict[str, Any],
) -> None:
    model = binary_hetero_model_bundle["model"]
    train_x = binary_hetero_model_bundle["train_x"]
    bounds = binary_hetero_model_bundle["bounds"]
    q = 2

    for acq_cls, kwargs, case_id in _representative_hetero_acquisition_cases(model):
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
        assert_candidates_in_bounds(
            cands=cands,
            bounds=bounds,
        )


@pytest.mark.slow
def test_binary_hetero_mixed_optimize_acqf_mixed_representative_smoke(
    binary_hetero_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_hetero_mixed_model_bundle["model"]
    train_x = binary_hetero_mixed_model_bundle["train_x"]
    bounds = binary_hetero_mixed_model_bundle["bounds"]
    cat_id = binary_hetero_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    q = 2

    for acq_cls, kwargs, case_id in _representative_hetero_acquisition_cases(model):
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
        assert_candidates_in_bounds(
            cands=cands,
            bounds=bounds,
        )
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id


@pytest.mark.slow
def test_binary_hetero_optimizer_constraint_case_smoke(
    binary_hetero_model_bundle: dict[str, Any],
) -> None:
    model = binary_hetero_model_bundle["model"]
    train_x = binary_hetero_model_bundle["train_x"]
    bounds = binary_hetero_model_bundle["bounds"]
    q = 2

    for (
        acq_cls,
        kwargs,
        _,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in _optimizer_constraint_scenarios(model, bounds):
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
def test_binary_hetero_mixed_optimizer_constraint_case_smoke(
    binary_hetero_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_hetero_mixed_model_bundle["model"]
    train_x = binary_hetero_mixed_model_bundle["train_x"]
    bounds = binary_hetero_mixed_model_bundle["bounds"]
    cat_id = binary_hetero_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    q = 2

    for (
        acq_cls,
        kwargs,
        _,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in _optimizer_constraint_scenarios(
        model,
        bounds,
        mixed=True,
    ):
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


# ============================================================
# Jupyter helpers
# ============================================================


def run_jupyter_forward_check(
    *,
    cat: bool = False,
    n: int = 12,
    d: int = 5,
    num_epochs: int = 8,
    aux_num_epochs: int = 4,
    batch_size: int = 4,
    q: int = 2,
    verbose_forward_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_binary_hetero_model_bundle(
        cat=cat,
        n=n,
        d=d,
        num_epochs=num_epochs,
        aux_num_epochs=aux_num_epochs,
    )
    model = bundle["model"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]

    X = (
        make_random_mixed_batch(
            bounds,
            cat_dims,
            batch_size=batch_size,
            q=q,
        )
        if cat
        else make_random_batch(
            bounds,
            batch_size=batch_size,
            q=q,
        )
    )

    print("=" * 80)
    print(f"Jupyter hetero forward check cat={cat}")
    if verbose_forward_detail:
        print(f"train_inputs_raw[0].shape={model.train_inputs_raw[0].shape}")
        print(f"train_inputs[0].shape={model.train_inputs[0].shape}")
        print(f"model.train_inputs[0].shape={model.model.train_inputs[0].shape}")
        print(f"train_Yvar.shape={model.train_Yvar.shape}")
        print(f"noise_model={type(model.noise_model).__name__}")
        print(f"bounds.shape={bounds.shape}")
        print(f"X.shape={X.shape}")
    print("=" * 80)

    for acq_cls, kwargs, case_id in hetero_acquisition_cases(model):
        try:
            values = acq_cls(model=model, **kwargs)(X)
            assert values.shape == torch.Size([batch_size]), case_id
            assert torch.isfinite(values).all(), case_id
            if verbose_forward_detail:
                print(
                    f"[OK] {case_id} shape={tuple(values.shape)} "
                    f"min={values.min().item():.6g} "
                    f"max={values.max().item():.6g}"
                )
        except Exception as exc:
            print(f"[NG] {case_id} {type(exc).__name__}")
            print(str(exc))
            raise

    print("forward check passed.")
    return bundle


def run_jupyter_all_forward_checks(
    *,
    num_epochs: int = 8,
    aux_num_epochs: int = 4,
    verbose_forward_detail: bool = False,
) -> None:
    run_jupyter_forward_check(
        cat=False,
        num_epochs=num_epochs,
        aux_num_epochs=aux_num_epochs,
        verbose_forward_detail=verbose_forward_detail,
    )
    run_jupyter_forward_check(
        cat=True,
        num_epochs=num_epochs,
        aux_num_epochs=aux_num_epochs,
        verbose_forward_detail=verbose_forward_detail,
    )
    print("all hetero forward checks passed.")
