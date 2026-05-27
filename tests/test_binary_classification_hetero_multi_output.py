from __future__ import annotations

"""Binary classification heteroscedastic multi-output smoke tests."""

from typing import Any

import pytest
import torch
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from bochan.acquisition.binary.active_learning import (
    qHeteroMultiOutputBinaryBALD,
    qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy,
    qHeteroMultiOutputBinaryMarginUncertainty,
    qHeteroMultiOutputBinaryPredictiveEntropy,
    qHeteroMultiOutputBinaryProbabilityVariance,
)
from bochan.acquisition.binary.bayesian_optimization import (
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement,
    qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputBinaryNParEGO,
)
from bochan.acquisition.binary.levelset_estimation import (
    qHeteroMultiOutputBinaryBoundaryVarianceAcquisition,
    qHeteroMultiOutputBinaryClassEntropyAcquisition,
    qHeteroMultiOutputBinaryICUAcquisition,
    qHeteroMultiOutputBinaryJointLatentStraddleAcquisition,
    qHeteroMultiOutputBinaryLatentStraddleAcquisition,
)
from bochan.models.classification.binary.base import MultiOutputBinaryClassificationModel
from bochan.models.classification.binary.robust import (
    HeteroscedasticBinaryClassificationGPModel,
    HeteroscedasticBinaryClassificationMixedGPModel,
)
from tests._binary_classification_multi_output_variant_utils import assert_multi_output_wrapper_training
from tests.test_binary_classification_base_multi_output import (
    N_OUTPUTS,
    _build_input_transform,
    make_multi_output_binary_toy_data,
)
from tests.test_binary_classification_base_single_output import (
    DTYPE,
    DEVICE,
    assert_candidates_in_bounds,
    assert_optimizer_compatibility_result,
    make_constraint_cases,
    make_random_batch,
    make_random_mixed_batch,
    maybe_suppress_botorch_initial_warnings,
    optimize_mixed_with_case,
    optimize_with_case,
    optimizer_cases,
)
from tests.test_binary_classification_hetero_single_output import (
    _assert_hetero_model_training,
    _fit_hetero_binary_model,
)


def _hetero_bo_reference_objects(model: MultiOutputBinaryClassificationModel, train_x: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        y_baseline = model.probability_posterior(train_x).mean.reshape(-1, model.num_outputs).clamp(1e-6, 1.0 - 1e-6)
    ref_point = torch.full((model.num_outputs,), -0.05, dtype=train_x.dtype, device=train_x.device)
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=y_baseline)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
    weights = torch.ones(model.num_outputs, dtype=train_x.dtype, device=train_x.device)
    weights = weights / weights.sum()
    return {"ref_point": ref_point, "ref_point_list": ref_point.detach().cpu().tolist(), "partitioning": partitioning, "sampler": sampler, "weights": weights}


def hetero_multi_output_active_learning_acquisition_cases(model: MultiOutputBinaryClassificationModel) -> list[tuple[type, dict[str, Any], str]]:
    common_kwargs: dict[str, Any] = {
        "reduction": "mean",
        "output_mode": "mean",
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
        (qHeteroMultiOutputBinaryPredictiveEntropy, dict(common_kwargs), "al_hetero_predictive_entropy"),
        (qHeteroMultiOutputBinaryProbabilityVariance, dict(common_kwargs), "al_hetero_probability_variance"),
        (qHeteroMultiOutputBinaryMarginUncertainty, dict(common_kwargs), "al_hetero_margin_uncertainty"),
        (qHeteroMultiOutputBinaryBALD, {**common_kwargs, "num_samples": 8}, "al_hetero_bald"),
        (qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy, dict(common_kwargs), "al_hetero_integrated_posterior_variance_proxy"),
        (qHeteroMultiOutputBinaryPredictiveEntropy, {**common_kwargs, "output_mode": "all_positive"}, "al_hetero_predictive_entropy_all_positive"),
    ]


def hetero_multi_output_levelset_acquisition_cases(model: MultiOutputBinaryClassificationModel, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    common_kwargs: dict[str, Any] = {
        "reduction": "mean",
        "output_mode": "mean",
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
        (qHeteroMultiOutputBinaryClassEntropyAcquisition, dict(common_kwargs), "lse_hetero_class_entropy"),
        (qHeteroMultiOutputBinaryICUAcquisition, dict(common_kwargs), "lse_hetero_icu"),
        (qHeteroMultiOutputBinaryBoundaryVarianceAcquisition, {**common_kwargs, "thresholds": 0.0, "tau": 1.0}, "lse_hetero_boundary_variance"),
        (qHeteroMultiOutputBinaryLatentStraddleAcquisition, {**common_kwargs, "thresholds": 0.0, "beta": 1.0}, "lse_hetero_latent_straddle"),
        (
            qHeteroMultiOutputBinaryJointLatentStraddleAcquisition,
            {
                "beta": 1.0,
                "thresholds": 0.0,
                "uncertainty_mode": "sqrt_trace",
                "boundary_mode": "l2_mean",
                "same_batch_penalty_weight": 0.01,
                "pending_penalty_weight": 0.01,
                "observed_penalty_weight": 0.0,
                "X_observed": train_x,
                "noise_mode": "inverse_linear",
                "noise_combine": "multiply",
                "noise_penalty_lambda": 0.1,
                "noise_min_weight": 0.05,
                "noise_weight_scale": 1.0,
                "noise_model_outputs_log_var": True,
            },
            "lse_hetero_joint_latent_straddle",
        ),
    ]


def hetero_multi_output_bo_acquisition_cases(model: MultiOutputBinaryClassificationModel, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    refs = _hetero_bo_reference_objects(model, train_x)
    common_kwargs = {
        "beta": 1.0,
        "noise_penalty": 0.1,
        "default_sigma": 0.0,
        "noise_is_log_var": True,
        "samples_are_probs": False,
        "apply_sigmoid_if_needed": True,
        "sampler": refs["sampler"],
    }
    return [
        (qHeteroMultiOutputBinaryExpectedHypervolumeImprovement, {"ref_point": refs["ref_point_list"], "partitioning": refs["partitioning"], **common_kwargs}, "bo_hetero_qehvi"),
        (qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement, {"ref_point": refs["ref_point"], "X_baseline": train_x, **common_kwargs}, "bo_hetero_qnehvi"),
        (qHeteroMultiOutputBinaryNParEGO, {"X_baseline": train_x, "ref_point": refs["ref_point"], "weights": refs["weights"], **common_kwargs}, "bo_hetero_nparego"),
    ]


def hetero_multi_output_acquisition_cases(model: MultiOutputBinaryClassificationModel, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    return (
        hetero_multi_output_active_learning_acquisition_cases(model)
        + hetero_multi_output_levelset_acquisition_cases(model, train_x)
        + hetero_multi_output_bo_acquisition_cases(model, train_x)
    )


def _representative_hetero_multi_output_acquisition_cases(model: MultiOutputBinaryClassificationModel, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    names = {"al_hetero_probability_variance", "al_hetero_bald", "lse_hetero_icu", "lse_hetero_latent_straddle", "bo_hetero_qehvi", "bo_hetero_nparego"}
    return [case for case in hetero_multi_output_acquisition_cases(model, train_x) if case[2] in names]


def _constraint_hetero_multi_output_acquisition_cases(model: MultiOutputBinaryClassificationModel, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    names = {"al_hetero_probability_variance", "lse_hetero_icu", "bo_hetero_qehvi"}
    return [case for case in hetero_multi_output_acquisition_cases(model, train_x) if case[2] in names]


def _hetero_optimizer_constraint_scenarios(model: MultiOutputBinaryClassificationModel, train_x: torch.Tensor, bounds: torch.Tensor, *, mixed: bool = False):
    names = {"none", "step_only", "constraints_only", "step_k_sparse_constraints"}
    constraint_cases = [case for case in make_constraint_cases(bounds) if case["case_id"] in names]
    scenarios = []
    for acq_cls, kwargs, acq_id in _constraint_hetero_multi_output_acquisition_cases(model, train_x):
        for optimize_func, optimize_method, optimizer_id in optimizer_cases():
            for constraint_case in constraint_cases:
                case_id = f"{acq_id}__{optimizer_id}__{constraint_case['case_id']}"
                if mixed:
                    case_id = f"mixed__{case_id}"
                scenarios.append((acq_cls, kwargs, acq_id, optimize_func, optimize_method, constraint_case, case_id))
    return scenarios


def create_hetero_multi_output_binary_model_bundle(
    *,
    cat: bool = False,
    n: int = 12,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    aux_num_epochs: int = 4,
) -> dict[str, Any]:
    train_x, train_y, bounds = make_multi_output_binary_toy_data(n=n, d=d, cat=cat, m=m)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    model_cls = HeteroscedasticBinaryClassificationMixedGPModel if cat else HeteroscedasticBinaryClassificationGPModel

    models: list[Any] = []
    for j in range(train_y.shape[-1]):
        kwargs: dict[str, Any] = {
            "train_X": train_x,
            "train_Y": train_y[:, [j]],
            "input_transform": _build_input_transform(train_x, bounds, cat_dims),
            "num_inducing_points": 8,
            "aux_num_epochs": aux_num_epochs,
            "aux_lr": 0.01,
            "min_noise": 1e-6,
        }
        if cat:
            kwargs["cat_dims"] = cat_dims
        torch.manual_seed(j)
        submodel = model_cls(**kwargs)
        _fit_hetero_binary_model(submodel, num_epochs=num_epochs, lr=0.01)
        models.append(submodel)

    model = MultiOutputBinaryClassificationModel(*models)
    assert_multi_output_wrapper_training(
        model=model,
        train_x=train_x,
        train_y=train_y,
        cat_dims=cat_dims,
        submodel_assert_fn=_assert_hetero_model_training,
    )
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def hetero_multi_output_binary_model_bundle() -> dict[str, Any]:
    return create_hetero_multi_output_binary_model_bundle(cat=False)


@pytest.fixture(scope="module")
def hetero_multi_output_binary_mixed_model_bundle() -> dict[str, Any]:
    return create_hetero_multi_output_binary_model_bundle(cat=True)


def test_hetero_multi_output_binary_model_basic_behavior(hetero_multi_output_binary_model_bundle: dict[str, Any]) -> None:
    assert_multi_output_wrapper_training(model=hetero_multi_output_binary_model_bundle["model"], train_x=hetero_multi_output_binary_model_bundle["train_x"], train_y=hetero_multi_output_binary_model_bundle["train_y"], cat_dims=hetero_multi_output_binary_model_bundle["cat_dims"], submodel_assert_fn=_assert_hetero_model_training)


def test_hetero_multi_output_binary_mixed_model_basic_behavior(hetero_multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    assert_multi_output_wrapper_training(model=hetero_multi_output_binary_mixed_model_bundle["model"], train_x=hetero_multi_output_binary_mixed_model_bundle["train_x"], train_y=hetero_multi_output_binary_mixed_model_bundle["train_y"], cat_dims=hetero_multi_output_binary_mixed_model_bundle["cat_dims"], submodel_assert_fn=_assert_hetero_model_training)


def test_hetero_multi_output_binary_acquisition_forward_shapes(hetero_multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = hetero_multi_output_binary_model_bundle["model"]
    train_x = hetero_multi_output_binary_model_bundle["train_x"]
    X = make_random_batch(hetero_multi_output_binary_model_bundle["bounds"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in hetero_multi_output_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_hetero_multi_output_binary_mixed_acquisition_forward_shapes(hetero_multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    model = hetero_multi_output_binary_mixed_model_bundle["model"]
    train_x = hetero_multi_output_binary_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(hetero_multi_output_binary_mixed_model_bundle["bounds"], hetero_multi_output_binary_mixed_model_bundle["cat_dims"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in hetero_multi_output_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


@pytest.mark.slow
def test_hetero_multi_output_binary_optimize_acqf_representative_smoke(hetero_multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = hetero_multi_output_binary_model_bundle["model"]
    train_x = hetero_multi_output_binary_model_bundle["train_x"]
    bounds = hetero_multi_output_binary_model_bundle["bounds"]
    for acq_cls, kwargs, case_id in _representative_hetero_multi_output_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf(acq_function=acq_cls(model=model, **kwargs), bounds=bounds, q=2, sequential=True, num_restarts=2, raw_samples=16, options={"maxiter": 10})
        assert cands.shape == torch.Size([2, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)


@pytest.mark.slow
def test_hetero_multi_output_binary_mixed_optimize_acqf_mixed_representative_smoke(hetero_multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    model = hetero_multi_output_binary_mixed_model_bundle["model"]
    train_x = hetero_multi_output_binary_mixed_model_bundle["train_x"]
    bounds = hetero_multi_output_binary_mixed_model_bundle["bounds"]
    cat_id = hetero_multi_output_binary_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    for acq_cls, kwargs, case_id in _representative_hetero_multi_output_acquisition_cases(model, train_x):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_acqf_mixed(acq_function=acq_cls(model=model, **kwargs), bounds=bounds, q=2, fixed_features_list=fixed_features_list, num_restarts=2, raw_samples=16, options={"maxiter": 10})
        assert cands.shape == torch.Size([2, train_x.shape[-1]]), case_id
        assert torch.isfinite(cands).all(), case_id
        assert torch.isfinite(acq_value).all(), case_id
        assert_candidates_in_bounds(cands=cands, bounds=bounds)
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id


@pytest.mark.slow
def test_hetero_multi_output_binary_optimizer_constraint_case_smoke(hetero_multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = hetero_multi_output_binary_model_bundle["model"]
    train_x = hetero_multi_output_binary_model_bundle["train_x"]
    bounds = hetero_multi_output_binary_model_bundle["bounds"]
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _hetero_optimizer_constraint_scenarios(model, train_x, bounds):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_with_case(acqf=acq_cls(model=model, **kwargs), bounds=bounds, q=2, optimize_func=optimize_func, optimize_method=optimize_method, constraint_case=constraint_case, num_restarts=2, raw_samples=16, maxiter=10)
        assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=2, d=train_x.shape[-1], constraint_case=constraint_case, case_id=case_id)


@pytest.mark.slow
def test_hetero_multi_output_binary_mixed_optimizer_constraint_case_smoke(hetero_multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    model = hetero_multi_output_binary_mixed_model_bundle["model"]
    train_x = hetero_multi_output_binary_mixed_model_bundle["train_x"]
    bounds = hetero_multi_output_binary_mixed_model_bundle["bounds"]
    cat_id = hetero_multi_output_binary_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in _hetero_optimizer_constraint_scenarios(model, train_x, bounds, mixed=True):
        with maybe_suppress_botorch_initial_warnings():
            cands, acq_value = optimize_mixed_with_case(acqf=acq_cls(model=model, **kwargs), bounds=bounds, q=2, fixed_features_list=fixed_features_list, optimize_func=optimize_func, optimize_method=optimize_method, constraint_case=constraint_case, num_restarts=2, raw_samples=16, maxiter=10)
        assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=2, d=train_x.shape[-1], constraint_case=constraint_case, case_id=case_id)
        assert torch.isin(cands[:, cat_id], cat_values).all(), case_id
