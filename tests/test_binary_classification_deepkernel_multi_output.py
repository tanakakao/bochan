from __future__ import annotations

"""Binary classification DeepKernel multi-output smoke tests.

Each output is modeled by an independent single-output DeepKernel binary
classifier and wrapped by ``MultiOutputBinaryClassificationModel``. Acquisition
cases and optimizer / constraint scenarios are reused from the base multi-output
test.
"""

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from bochan.fit import fit_deepkernel_mll
from bochan.models.classification.binary.base import MultiOutputBinaryClassificationModel
from bochan.models.classification.binary.deep import (
    DeepKernelBinaryClassificationGPModel,
    DeepKernelBinaryClassificationMixedGPModel,
)
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


def _fit_deepkernel_single_output_model(
    model: Any,
    *,
    num_epochs: int,
    lr: float = 0.01,
) -> None:
    """DeepKernel single-output binary classifier を軽量 fit する。"""
    fit_deepkernel_mll(
        model.make_mll(),
        num_epochs=num_epochs,
        lr=lr,
    )


def _expected_transformed_x(submodel: Any, train_x: torch.Tensor) -> torch.Tensor:
    expected = (
        submodel.input_transform(train_x)
        if getattr(submodel, "input_transform", None) is not None
        else train_x
    )
    if isinstance(expected, tuple):
        expected = expected[0]
    return expected


def _assert_deepkernel_single_submodel_training(
    submodel: Any,
    train_x: torch.Tensor,
    train_y_j: torch.Tensor,
    *,
    cat_dims: list[int],
    output_index: int,
) -> None:
    """MultiOutput wrapper 内の DeepKernel single-output submodel を確認する。"""
    submodel.eval()

    assert submodel.num_outputs == 1
    assert submodel.train_inputs[0].shape == train_x.shape, output_index
    assert submodel.train_inputs_raw[0].shape == train_x.shape, output_index
    assert submodel.train_inputs[0].dtype == train_x.dtype, output_index
    assert submodel.train_inputs_raw[0].dtype == train_x.dtype, output_index
    assert torch.allclose(submodel.train_inputs[0], train_x), output_index
    assert torch.allclose(submodel.train_inputs_raw[0], train_x), output_index
    assert submodel.train_inputs_raw[0].data_ptr() != train_x.data_ptr(), output_index
    assert torch.allclose(submodel.train_targets, train_y_j.reshape(-1)), output_index

    with torch.no_grad():
        expected_x = _expected_transformed_x(submodel, train_x)
        transformed_x = submodel.transform_inputs(train_x)
        posterior = submodel.posterior(train_x)
        latent_posterior = submodel.latent_posterior(train_x)

    assert transformed_x.shape == expected_x.shape, output_index
    assert torch.allclose(transformed_x, expected_x), output_index
    assert submodel.model.train_inputs[0].shape == expected_x.shape, output_index
    assert torch.allclose(submodel.model.train_inputs[0], expected_x), output_index
    assert torch.allclose(submodel.model.train_targets, train_y_j.reshape(-1)), output_index
    assert submodel.make_mll().model is submodel.model, output_index

    assert posterior.mean.shape == train_y_j.shape, output_index
    assert posterior.variance.shape == train_y_j.shape, output_index
    assert torch.isfinite(posterior.mean).all(), output_index
    assert torch.isfinite(posterior.variance).all(), output_index
    assert (posterior.mean >= 0.0).all() and (posterior.mean <= 1.0).all(), output_index
    assert torch.isfinite(latent_posterior.mean).all(), output_index

    if cat_dims:
        assert hasattr(submodel, "cat_dims"), output_index
        assert hasattr(submodel.model, "cat_dims"), output_index
        assert list(submodel.cat_dims) == cat_dims, output_index
        assert list(submodel.model.cat_dims) == cat_dims, output_index
        cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=train_x.dtype, device=train_x.device)
        for cat_id in cat_dims:
            assert torch.isin(submodel.train_inputs[0][:, cat_id], cat_values).all(), output_index
            assert torch.isin(submodel.train_inputs_raw[0][:, cat_id], cat_values).all(), output_index
            assert torch.isin(transformed_x[:, cat_id], cat_values).all(), output_index
            assert torch.isin(submodel.model.train_inputs[0][:, cat_id], cat_values).all(), output_index


def _assert_deepkernel_multi_output_model_training(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
) -> None:
    """DeepKernel single-output model list から作った multi-output wrapper を確認する。"""
    model.eval()
    n, m = train_y.shape

    assert model.num_outputs == m
    assert len(model.models) == m
    assert model.batch_shape == torch.Size([])
    assert model.num_classes_list == [2 for _ in range(m)]
    assert list(getattr(model, "cat_dims", [])) == cat_dims
    assert model.train_inputs[0].shape == train_x.shape
    assert torch.allclose(model.train_inputs[0], train_x)
    assert torch.allclose(model.train_targets, train_y)
    assert torch.allclose(model.train_Y, train_y)

    for j, submodel in enumerate(model.models):
        _assert_deepkernel_single_submodel_training(
            submodel=submodel,
            train_x=train_x,
            train_y_j=train_y[:, [j]],
            cat_dims=cat_dims,
            output_index=j,
        )

    with torch.no_grad():
        posterior = model.posterior(train_x)
        prob_post = model.probability_posterior(train_x)
        latent_post = model.latent_posterior(train_x)
        mean_probability = model.mean_probability(train_x)
        probability_variance = model.probability_variance(train_x)
        class_probs = model.class_probs(train_x)
        pred_class = model.predict_class(train_x)
        subset_post = model.posterior(train_x, output_indices=[0, m - 1])
        subset_latent = model.latent_posterior(train_x, output_indices=[0])

    assert posterior.mean.shape == torch.Size([n, m])
    assert posterior.variance.shape == torch.Size([n, m])
    assert prob_post.mean.shape == torch.Size([n, m])
    assert mean_probability.shape == torch.Size([n, m])
    assert probability_variance.shape == torch.Size([n, m])
    assert class_probs.shape == torch.Size([n, m, 2])
    assert pred_class.shape == torch.Size([n, m])
    assert subset_post.mean.shape == torch.Size([n, 2])
    assert subset_latent.mean.shape == torch.Size([n, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(latent_post.mean).all()
    assert (posterior.mean >= 0.0).all() and (posterior.mean <= 1.0).all()
    assert torch.allclose(class_probs.sum(dim=-1), torch.ones_like(posterior.mean))
    assert torch.isin(pred_class, torch.tensor([0, 1], device=pred_class.device)).all()


def create_deepkernel_multi_output_binary_model_bundle(
    *,
    cat: bool = False,
    n: int = 16,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    input_transform: Normalize | None = None,
) -> dict[str, Any]:
    """出力ごとに DeepKernel single-output classifier を fit して multi-output wrapper を作る。"""
    train_x, train_y, bounds = make_multi_output_binary_toy_data(n=n, d=d, cat=cat, m=m)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    model_cls = DeepKernelBinaryClassificationMixedGPModel if cat else DeepKernelBinaryClassificationGPModel

    models: list[Any] = []
    for j in range(train_y.shape[-1]):
        sub_input_transform = _build_input_transform(train_x, bounds, cat_dims) if input_transform is None else input_transform
        kwargs: dict[str, Any] = {
            "train_X": train_x,
            "train_Y": train_y[:, [j]],
            "input_transform": sub_input_transform,
            "num_inducing_points": 8,
        }
        if cat:
            kwargs["cat_dims"] = cat_dims

        torch.manual_seed(j)
        submodel = model_cls(**kwargs)
        _fit_deepkernel_single_output_model(submodel, num_epochs=num_epochs, lr=0.01)
        models.append(submodel)

    model = MultiOutputBinaryClassificationModel(*models)
    _assert_deepkernel_multi_output_model_training(
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
def deepkernel_multi_output_binary_model_bundle() -> dict[str, Any]:
    """pytest 用: 通常 DeepKernel multi-output binary classification model。"""
    return create_deepkernel_multi_output_binary_model_bundle(
        cat=False,
        n=16,
        d=5,
        m=N_OUTPUTS,
        num_epochs=4,
    )


@pytest.fixture(scope="module")
def deepkernel_multi_output_binary_mixed_model_bundle() -> dict[str, Any]:
    """pytest 用: mixed DeepKernel multi-output binary classification model。"""
    return create_deepkernel_multi_output_binary_model_bundle(
        cat=True,
        n=16,
        d=5,
        m=N_OUTPUTS,
        num_epochs=4,
    )


def test_deepkernel_multi_output_binary_model_basic_behavior(
    deepkernel_multi_output_binary_model_bundle: dict[str, Any],
) -> None:
    _assert_deepkernel_multi_output_model_training(
        model=deepkernel_multi_output_binary_model_bundle["model"],
        train_x=deepkernel_multi_output_binary_model_bundle["train_x"],
        train_y=deepkernel_multi_output_binary_model_bundle["train_y"],
        cat_dims=deepkernel_multi_output_binary_model_bundle["cat_dims"],
    )


def test_deepkernel_multi_output_binary_mixed_model_basic_behavior(
    deepkernel_multi_output_binary_mixed_model_bundle: dict[str, Any],
) -> None:
    _assert_deepkernel_multi_output_model_training(
        model=deepkernel_multi_output_binary_mixed_model_bundle["model"],
        train_x=deepkernel_multi_output_binary_mixed_model_bundle["train_x"],
        train_y=deepkernel_multi_output_binary_mixed_model_bundle["train_y"],
        cat_dims=deepkernel_multi_output_binary_mixed_model_bundle["cat_dims"],
    )


def test_deepkernel_multi_output_binary_acquisition_forward_shapes(
    deepkernel_multi_output_binary_model_bundle: dict[str, Any],
) -> None:
    model = deepkernel_multi_output_binary_model_bundle["model"]
    train_x = deepkernel_multi_output_binary_model_bundle["train_x"]
    X = make_random_batch(deepkernel_multi_output_binary_model_bundle["bounds"], batch_size=4, q=2)

    for acq_cls, kwargs, case_id in multi_output_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_deepkernel_multi_output_binary_mixed_acquisition_forward_shapes(
    deepkernel_multi_output_binary_mixed_model_bundle: dict[str, Any],
) -> None:
    model = deepkernel_multi_output_binary_mixed_model_bundle["model"]
    train_x = deepkernel_multi_output_binary_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(
        deepkernel_multi_output_binary_mixed_model_bundle["bounds"],
        deepkernel_multi_output_binary_mixed_model_bundle["cat_dims"],
        batch_size=4,
        q=2,
    )

    for acq_cls, kwargs, case_id in multi_output_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_deepkernel_multi_output_binary_family_case_coverage(
    deepkernel_multi_output_binary_model_bundle: dict[str, Any],
) -> None:
    model = deepkernel_multi_output_binary_model_bundle["model"]
    train_x = deepkernel_multi_output_binary_model_bundle["train_x"]
    case_ids = {case_id for _, _, case_id in multi_output_acquisition_cases(model, train_x)}
    assert any(case_id.startswith("al_") for case_id in case_ids)
    assert any(case_id.startswith("lse_") for case_id in case_ids)
    assert any(case_id.startswith("bo_") for case_id in case_ids)
    assert "bo_qehvi" in case_ids
    assert "bo_qnehvi" in case_ids
    assert "bo_nparego" in case_ids


def test_deepkernel_multi_output_binary_constraint_scenario_coverage(
    deepkernel_multi_output_binary_model_bundle: dict[str, Any],
) -> None:
    model = deepkernel_multi_output_binary_model_bundle["model"]
    train_x = deepkernel_multi_output_binary_model_bundle["train_x"]
    bounds = deepkernel_multi_output_binary_model_bundle["bounds"]
    scenarios = _optimizer_constraint_scenarios(model, train_x, bounds)
    case_ids = {scenario[-1] for scenario in scenarios}

    assert any(case_id.startswith("al_") for case_id in case_ids)
    assert any(case_id.startswith("lse_") for case_id in case_ids)
    assert any(case_id.startswith("bo_") for case_id in case_ids)
    assert any("evo_cmaes" in case_id for case_id in case_ids)
    assert any("evo_pso" in case_id for case_id in case_ids)
    assert any("evo_ga" in case_id for case_id in case_ids)
    assert any("constraints_only" in case_id for case_id in case_ids)
    assert any("step_k_sparse_constraints" in case_id for case_id in case_ids)


@pytest.mark.slow
def test_deepkernel_multi_output_binary_optimize_acqf_representative_smoke(
    deepkernel_multi_output_binary_model_bundle: dict[str, Any],
) -> None:
    model = deepkernel_multi_output_binary_model_bundle["model"]
    train_x = deepkernel_multi_output_binary_model_bundle["train_x"]
    bounds = deepkernel_multi_output_binary_model_bundle["bounds"]
    q = 2

    for acq_cls, kwargs, case_id in _representative_multi_output_acquisition_cases(model, train_x):
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
def test_deepkernel_multi_output_binary_mixed_optimize_acqf_mixed_representative_smoke(
    deepkernel_multi_output_binary_mixed_model_bundle: dict[str, Any],
) -> None:
    model = deepkernel_multi_output_binary_mixed_model_bundle["model"]
    train_x = deepkernel_multi_output_binary_mixed_model_bundle["train_x"]
    bounds = deepkernel_multi_output_binary_mixed_model_bundle["bounds"]
    cat_id = deepkernel_multi_output_binary_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    q = 2

    for acq_cls, kwargs, case_id in _representative_multi_output_acquisition_cases(model, train_x):
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
def test_deepkernel_multi_output_binary_optimizer_constraint_case_smoke(
    deepkernel_multi_output_binary_model_bundle: dict[str, Any],
) -> None:
    model = deepkernel_multi_output_binary_model_bundle["model"]
    train_x = deepkernel_multi_output_binary_model_bundle["train_x"]
    bounds = deepkernel_multi_output_binary_model_bundle["bounds"]
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
def test_deepkernel_multi_output_binary_mixed_optimizer_constraint_case_smoke(
    deepkernel_multi_output_binary_mixed_model_bundle: dict[str, Any],
) -> None:
    model = deepkernel_multi_output_binary_mixed_model_bundle["model"]
    train_x = deepkernel_multi_output_binary_mixed_model_bundle["train_x"]
    bounds = deepkernel_multi_output_binary_mixed_model_bundle["bounds"]
    cat_id = deepkernel_multi_output_binary_mixed_model_bundle["cat_dims"][0]
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
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    batch_size: int = 4,
    q: int = 2,
    verbose_forward_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_deepkernel_multi_output_binary_model_bundle(cat=cat, n=n, d=d, m=m, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]
    X = make_random_mixed_batch(bounds, cat_dims, batch_size=batch_size, q=q) if cat else make_random_batch(bounds, batch_size=batch_size, q=q)

    print("=" * 80)
    print(f"Jupyter DeepKernel multi-output binary forward check cat={cat}")
    if verbose_forward_detail:
        print(f"num_outputs={model.num_outputs}")
        print(f"train_inputs[0].shape={model.train_inputs[0].shape}")
        print(f"train_targets.shape={model.train_targets.shape}")
        print(f"bounds.shape={bounds.shape}")
        print(f"X.shape={X.shape}")
    print("=" * 80)

    for acq_cls, kwargs, case_id in multi_output_acquisition_cases(model, train_x):
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
    print("all DeepKernel multi-output binary forward checks passed.")
