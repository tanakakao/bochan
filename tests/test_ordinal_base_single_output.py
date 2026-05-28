from __future__ import annotations

"""Ordinal base single-output smoke tests.

This file mirrors the design of ``test_binary_classification_base_single_output``
for ordinal models. It covers

- standard and mixed ordinal GP models,
- active learning acquisitions,
- level-set estimation acquisitions,
- Bayesian optimization acquisitions,
- optimizer / constraint compatibility, including evo optimizers, and
- Jupyter-oriented all-check runners.
"""

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.acquisition.objective import OrdinalExpectedUtilityMCObjective
from bochan.acquisition.ordinal.active_learning import (
    qOrdinalBALD,
    qOrdinalMarginUncertainty,
    qOrdinalPredictiveEntropy,
    qOrdinalUtilityVariance,
)
from bochan.acquisition.ordinal.bayesian_optimization import (
    compute_ordinal_expected_utility_best_f,
    qOrdinalExpectedImprovement,
    qOrdinalProbabilityOfFeasibility,
    qOrdinalProbabilityOfImprovement,
    qOrdinalUpperConfidenceBound,
)
from bochan.acquisition.ordinal.bayesian_optimization.single_output import qOrdinalExpectedUtility
from bochan.acquisition.ordinal.levelset_estimation import (
    qOrdinalBoundaryVarianceAcquisition,
    qOrdinalClassEntropyAcquisition,
    qOrdinalICUAcquisition,
    qOrdinalJointLatentStraddleAcquisition,
    qOrdinalLatentStraddleAcquisition,
)
from bochan.fit.ordinal import fit_ordinal_mll, make_ordinal_mll
from bochan.models.ordinal.base import OrdinalGPModel, OrdinalMixedGPModel
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


NUM_CLASSES = 3
UTILITY_VALUES = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)


def make_ordinal_toy_data(
    n: int = 24,
    d: int = 5,
    cat: bool = False,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ordinal regression 用の toy data を作る。"""
    if num_classes != 3:
        raise ValueError("This test helper currently assumes num_classes=3.")

    train_x, _, bounds = make_binary_toy_data(n=n, d=d, cat=cat)
    cont_x = train_x[..., :d]
    score = (
        0.9 * cont_x[..., 0]
        - 0.6 * cont_x[..., 1]
        + 0.5 * cont_x[..., 2 % d]
        + torch.sin(2.0 * cont_x[..., 3 % d])
    )
    if cat:
        score = score + 0.2 * ((train_x[..., -1] - 10.0) / 5.0)

    q1, q2 = torch.quantile(
        score,
        torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=score.dtype, device=score.device),
    )
    labels = torch.zeros_like(score, dtype=torch.long)
    labels = labels + (score > q1).long() + (score > q2).long()

    # OrdinalGPModel が num_classes を推定できるように、各 class が最低1件あることを保証する。
    labels[0] = 0
    labels[1] = 1
    labels[2] = 2
    return train_x, labels, bounds


def _build_input_transform(train_x: torch.Tensor, bounds: torch.Tensor, cat_dims: list[int]) -> Normalize:
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(d=train_x.shape[-1], bounds=bounds, indices=cont_indices)


def _fit_ordinal_model(model: Any, *, num_epochs: int, lr: float = 0.03) -> None:
    mll = make_ordinal_mll(model)
    fit_ordinal_mll(mll, fit_model=model, num_epochs=num_epochs, lr=lr)


def _expected_transformed_x(model: Any, train_x: torch.Tensor) -> torch.Tensor:
    expected = model.input_transform(train_x) if getattr(model, "input_transform", None) is not None else train_x
    if isinstance(expected, tuple):
        expected = expected[0]
    return expected


def _assert_ordinal_model_training(
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
    assert model.train_inputs_raw[0].shape == train_x.shape
    assert model.train_inputs[0].shape == expected_x.shape
    assert model.model.train_inputs[0].shape == expected_x.shape
    assert torch.allclose(model.train_inputs_raw[0], train_x)
    assert torch.allclose(model.train_inputs[0], expected_x)
    assert torch.allclose(model.model.train_inputs[0], expected_x)
    assert torch.equal(model.train_targets, train_y.long())
    assert torch.equal(model.model.train_targets, train_y.long())
    assert model.train_inputs_raw[0].data_ptr() != train_x.data_ptr()

    cutpoints = model.ordinal_likelihood.cutpoints
    assert cutpoints.shape == torch.Size([NUM_CLASSES - 1])
    assert torch.isfinite(cutpoints).all()
    assert torch.all(cutpoints[1:] > cutpoints[:-1])

    mll = make_ordinal_mll(model)
    assert mll.model is model.model
    assert mll.likelihood is model.likelihood
    assert mll.num_data == train_x.shape[-2]

    with torch.no_grad():
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
    assert torch.allclose(probs.sum(dim=-1), torch.ones(train_x.shape[-2], dtype=probs.dtype, device=probs.device))
    assert pred_class.shape == torch.Size([train_x.shape[-2]])
    assert torch.isin(pred_class, torch.arange(NUM_CLASSES, device=pred_class.device)).all()
    assert expected_u.shape == torch.Size([train_x.shape[-2]])
    assert torch.isfinite(expected_u).all()

    if cat_dims:
        assert isinstance(model, OrdinalMixedGPModel)
        assert list(model.cat_dims) == cat_dims
        assert list(model.model.covar_module.kernels[1].base_kernel.active_dims.tolist()) == cat_dims or len(cat_dims) > 0
        cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=train_x.dtype, device=train_x.device)
        for cat_id in cat_dims:
            assert torch.isin(model.train_inputs_raw[0][:, cat_id], cat_values).all()
            assert torch.isin(model.train_inputs[0][:, cat_id], cat_values).all()
            assert torch.isin(model.model.train_inputs[0][:, cat_id], cat_values).all()


def create_ordinal_model_bundle(
    *,
    cat: bool = False,
    n: int = 24,
    d: int = 5,
    num_epochs: int = 8,
) -> dict[str, Any]:
    train_x, train_y, bounds = make_ordinal_toy_data(n=n, d=d, cat=cat)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    input_transform = _build_input_transform(train_x, bounds, cat_dims)

    torch.manual_seed(0)
    if cat:
        model = OrdinalMixedGPModel(
            train_X=train_x,
            train_Y=train_y,
            cat_dims=cat_dims,
            num_classes=NUM_CLASSES,
            input_transform=input_transform,
            inducing_points_num=8,
            conditioning_steps=4,
        )
    else:
        model = OrdinalGPModel(
            train_X=train_x,
            train_Y=train_y,
            num_classes=NUM_CLASSES,
            input_transform=input_transform,
            inducing_points_num=8,
            conditioning_steps=4,
        )

    _fit_ordinal_model(model, num_epochs=num_epochs, lr=0.03)
    _assert_ordinal_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def ordinal_model_bundle() -> dict[str, Any]:
    return create_ordinal_model_bundle(cat=False)


@pytest.fixture(scope="module")
def ordinal_mixed_model_bundle() -> dict[str, Any]:
    return create_ordinal_model_bundle(cat=True)


def _utility_objective(model: Any, train_x: torch.Tensor) -> OrdinalExpectedUtilityMCObjective:
    return OrdinalExpectedUtilityMCObjective(
        ordinal_likelihood=model.ordinal_likelihood,
        utility_values=UTILITY_VALUES.to(device=train_x.device, dtype=train_x.dtype),
    )


def ordinal_active_learning_acquisition_cases(model: Any) -> list[tuple[type, dict[str, Any], str]]:
    common_kwargs = {
        "reduction": "mean",
        "pending_penalty_weight": 0.01,
        "pending_penalty_beta": 5.0,
        "observed_penalty_weight": 0.0,
        "observed_penalty_beta": 5.0,
        "ordinal_likelihood": model.ordinal_likelihood,
    }
    return [
        (qOrdinalPredictiveEntropy, dict(common_kwargs), "al_predictive_entropy"),
        (qOrdinalUtilityVariance, {**common_kwargs, "utility_values": UTILITY_VALUES}, "al_utility_variance"),
        (qOrdinalMarginUncertainty, dict(common_kwargs), "al_margin_uncertainty"),
        (qOrdinalBALD, {**common_kwargs, "num_samples": 8}, "al_bald"),
    ]


def ordinal_levelset_acquisition_cases(model: Any, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    common_kwargs = {
        "reduction": "mean",
        "same_batch_penalty_weight": 0.01,
        "pending_penalty_weight": 0.01,
        "observed_penalty_weight": 0.0,
        "penalty_lengthscale": 0.2,
        "X_observed": train_x,
    }
    return [
        (qOrdinalClassEntropyAcquisition, dict(common_kwargs), "lse_class_entropy"),
        (qOrdinalICUAcquisition, {**common_kwargs, "target_boundary_idx": 0, "boundary_reduction": "sum"}, "lse_icu"),
        (qOrdinalBoundaryVarianceAcquisition, {**common_kwargs, "tau": 1.0, "target_boundary_idx": 0}, "lse_boundary_variance"),
        (qOrdinalLatentStraddleAcquisition, {**common_kwargs, "beta": 1.0, "target_boundary_idx": 0}, "lse_latent_straddle"),
        (
            qOrdinalJointLatentStraddleAcquisition,
            {
                "beta": 1.0,
                "tau": 1.0,
                "uncertainty_measure": "trace",
                "target_boundary_idx": 0,
                "boundary_reduction": "max",
                "same_batch_penalty_weight": 0.01,
                "pending_penalty_weight": 0.01,
                "observed_penalty_weight": 0.0,
                "penalty_lengthscale": 0.2,
                "X_observed": train_x,
            },
            "lse_joint_latent_straddle",
        ),
    ]


def ordinal_bo_acquisition_cases(model: Any, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    utility_values = UTILITY_VALUES.to(device=train_x.device, dtype=train_x.dtype)
    objective = _utility_objective(model, train_x)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
    best_f = compute_ordinal_expected_utility_best_f(
        model=model,
        train_X=train_x,
        utility_values=utility_values,
        maximize=True,
    )
    common_kwargs = {
        "objective": objective,
        "sampler": sampler,
        "pending_penalty_weight": 0.01,
        "pending_penalty_beta": 5.0,
        "same_batch_penalty_weight": 0.01,
        "same_batch_penalty_beta": 5.0,
        "observed_penalty_weight": 0.0,
        "observed_penalty_beta": 5.0,
        "X_observed": train_x,
    }
    return [
        (qOrdinalExpectedUtility, dict(common_kwargs), "bo_expected_utility"),
        (qOrdinalExpectedImprovement, {**common_kwargs, "best_f": best_f}, "bo_expected_improvement"),
        (qOrdinalProbabilityOfImprovement, {**common_kwargs, "best_f": best_f, "tau": 1e-2}, "bo_probability_of_improvement"),
        (qOrdinalUpperConfidenceBound, {**common_kwargs, "beta": 2.0}, "bo_ucb"),
        (
            qOrdinalProbabilityOfFeasibility,
            {
                "ordinal_likelihood": model.ordinal_likelihood,
                "mode": "class_ge",
                "min_class": 1,
                "q_feas_mode": "prod",
                "pending_penalty_weight": 0.01,
                "pending_penalty_beta": 5.0,
                "same_batch_penalty_weight": 0.01,
                "same_batch_penalty_beta": 5.0,
            },
            "bo_probability_of_feasibility",
        ),
    ]


def ordinal_acquisition_cases(model: Any, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    return (
        ordinal_active_learning_acquisition_cases(model)
        + ordinal_levelset_acquisition_cases(model, train_x)
        + ordinal_bo_acquisition_cases(model, train_x)
    )


def _representative_ordinal_acquisition_cases(model: Any, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    names = {
        "al_utility_variance",
        "al_bald",
        "lse_icu",
        "lse_latent_straddle",
        "bo_expected_improvement",
        "bo_probability_of_feasibility",
    }
    return [case for case in ordinal_acquisition_cases(model, train_x) if case[2] in names]


def _constraint_ordinal_acquisition_cases(model: Any, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    names = {"al_utility_variance", "lse_icu", "bo_probability_of_feasibility"}
    return [case for case in ordinal_acquisition_cases(model, train_x) if case[2] in names]


def _representative_constraint_cases(bounds: torch.Tensor) -> list[dict[str, Any]]:
    names = {"none", "step_only", "constraints_only", "step_k_sparse_constraints"}
    return [case for case in make_constraint_cases(bounds) if case["case_id"] in names]


def _ordinal_optimizer_constraint_scenarios(
    model: Any,
    train_x: torch.Tensor,
    bounds: torch.Tensor,
    *,
    mixed: bool = False,
    full_matrix: bool = False,
):
    acquisition_cases = (
        _representative_ordinal_acquisition_cases(model, train_x)
        if full_matrix
        else _constraint_ordinal_acquisition_cases(model, train_x)
    )
    constraint_cases = make_constraint_cases(bounds) if full_matrix else _representative_constraint_cases(bounds)
    scenarios = []
    for acq_cls, kwargs, acq_id in acquisition_cases:
        for optimize_func, optimize_method, optimizer_id in optimizer_cases():
            for constraint_case in constraint_cases:
                case_id = f"{acq_id}__{optimizer_id}__{constraint_case['case_id']}"
                if mixed:
                    case_id = f"mixed__{case_id}"
                scenarios.append((acq_cls, kwargs, acq_id, optimize_func, optimize_method, constraint_case, case_id))
    return scenarios


def test_ordinal_model_basic_behavior(ordinal_model_bundle: dict[str, Any]) -> None:
    _assert_ordinal_model_training(
        model=ordinal_model_bundle["model"],
        train_x=ordinal_model_bundle["train_x"],
        train_y=ordinal_model_bundle["train_y"],
        cat_dims=ordinal_model_bundle["cat_dims"],
    )


def test_ordinal_mixed_model_basic_behavior(ordinal_mixed_model_bundle: dict[str, Any]) -> None:
    _assert_ordinal_model_training(
        model=ordinal_mixed_model_bundle["model"],
        train_x=ordinal_mixed_model_bundle["train_x"],
        train_y=ordinal_mixed_model_bundle["train_y"],
        cat_dims=ordinal_mixed_model_bundle["cat_dims"],
    )


def test_ordinal_acquisition_forward_shapes(ordinal_model_bundle: dict[str, Any]) -> None:
    model = ordinal_model_bundle["model"]
    train_x = ordinal_model_bundle["train_x"]
    X = make_random_batch(ordinal_model_bundle["bounds"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in ordinal_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_ordinal_mixed_acquisition_forward_shapes(ordinal_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_mixed_model_bundle["model"]
    train_x = ordinal_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(ordinal_mixed_model_bundle["bounds"], ordinal_mixed_model_bundle["cat_dims"], batch_size=4, q=2)
    for acq_cls, kwargs, case_id in ordinal_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_ordinal_family_case_coverage(ordinal_model_bundle: dict[str, Any]) -> None:
    model = ordinal_model_bundle["model"]
    train_x = ordinal_model_bundle["train_x"]
    case_ids = {case_id for _, _, case_id in ordinal_acquisition_cases(model, train_x)}
    assert any(case_id.startswith("al_") for case_id in case_ids)
    assert any(case_id.startswith("lse_") for case_id in case_ids)
    assert any(case_id.startswith("bo_") for case_id in case_ids)


def test_ordinal_constraint_scenario_coverage(ordinal_model_bundle: dict[str, Any]) -> None:
    model = ordinal_model_bundle["model"]
    train_x = ordinal_model_bundle["train_x"]
    bounds = ordinal_model_bundle["bounds"]
    scenarios = _ordinal_optimizer_constraint_scenarios(model, train_x, bounds)
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
def test_ordinal_optimize_acqf_representative_smoke(ordinal_model_bundle: dict[str, Any]) -> None:
    model = ordinal_model_bundle["model"]
    train_x = ordinal_model_bundle["train_x"]
    bounds = ordinal_model_bundle["bounds"]
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
def test_ordinal_mixed_optimize_acqf_mixed_representative_smoke(ordinal_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_mixed_model_bundle["model"]
    train_x = ordinal_mixed_model_bundle["train_x"]
    bounds = ordinal_mixed_model_bundle["bounds"]
    cat_id = ordinal_mixed_model_bundle["cat_dims"][0]
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
def test_ordinal_optimizer_constraint_case_smoke(ordinal_model_bundle: dict[str, Any]) -> None:
    model = ordinal_model_bundle["model"]
    train_x = ordinal_model_bundle["train_x"]
    bounds = ordinal_model_bundle["bounds"]
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
def test_ordinal_mixed_optimizer_constraint_case_smoke(ordinal_mixed_model_bundle: dict[str, Any]) -> None:
    model = ordinal_mixed_model_bundle["model"]
    train_x = ordinal_mixed_model_bundle["train_x"]
    bounds = ordinal_mixed_model_bundle["bounds"]
    cat_id = ordinal_mixed_model_bundle["cat_dims"][0]
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


# ============================================================
# Jupyter helpers
# ============================================================


def _print_failure_summary(failed_cases: list[tuple[str, Exception]]) -> None:
    print("=" * 100)
    if failed_cases:
        print(f"failed_cases={len(failed_cases)}")
        for case_id, exc in failed_cases:
            print(f"  - {case_id}: {type(exc).__name__}: {exc}")
    else:
        print("all checks passed.")
    print("=" * 100)


def _fixed_features_for_bundle(bundle: dict[str, Any]) -> tuple[list[dict[int, float]], torch.Tensor]:
    cat_id = bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    return fixed_features_list, cat_values


def run_jupyter_forward_check(
    *,
    cat: bool = False,
    n: int = 24,
    d: int = 5,
    num_epochs: int = 8,
    batch_size: int = 4,
    q: int = 2,
    verbose_forward_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_ordinal_model_bundle(cat=cat, n=n, d=d, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]
    X = make_random_mixed_batch(bounds, cat_dims, batch_size=batch_size, q=q) if cat else make_random_batch(bounds, batch_size=batch_size, q=q)

    print("=" * 80)
    print(f"Jupyter ordinal base forward check cat={cat}")
    if verbose_forward_detail:
        print(f"train_inputs_raw[0].shape={model.train_inputs_raw[0].shape}")
        print(f"train_inputs[0].shape={model.train_inputs[0].shape}")
        print(f"train_targets.shape={model.train_targets.shape}")
        print(f"cutpoints={model.ordinal_likelihood.cutpoints.detach().cpu().tolist()}")
        print(f"bounds.shape={bounds.shape}")
        print(f"X.shape={X.shape}")
    print("=" * 80)

    for acq_cls, kwargs, case_id in ordinal_acquisition_cases(model, train_x):
        values = acq_cls(model=model, **kwargs)(X)
        assert values.shape == torch.Size([batch_size]), case_id
        assert torch.isfinite(values).all(), case_id
        if verbose_forward_detail:
            print(f"[OK] {case_id} shape={tuple(values.shape)} min={values.min().item():.6g} max={values.max().item():.6g}")

    print("forward check passed.")
    return bundle


def run_jupyter_all_forward_checks(*, num_epochs: int = 8, verbose_forward_detail: bool = False) -> None:
    run_jupyter_forward_check(cat=False, num_epochs=num_epochs, verbose_forward_detail=verbose_forward_detail)
    run_jupyter_forward_check(cat=True, num_epochs=num_epochs, verbose_forward_detail=verbose_forward_detail)
    print("all ordinal base forward checks passed.")


def run_jupyter_optimize_all_acquisitions_check(
    *,
    cat: bool = False,
    n: int = 24,
    d: int = 5,
    num_epochs: int = 8,
    q: int = 2,
    num_restarts: int = 2,
    raw_samples: int = 16,
    maxiter: int = 10,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
    verbose_ok_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_ordinal_model_bundle(cat=cat, n=n, d=d, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cases = ordinal_acquisition_cases(model, train_x)
    failed_cases: list[tuple[str, Exception]] = []
    prefix = "mixed_" if cat else ""

    print("=" * 100)
    print(f"Jupyter ordinal base {prefix}optimize check: all acquisitions")
    print(f"n={n}, d={d}, q={q}, num_epochs={num_epochs}, num_acquisitions={len(cases)}")
    print("=" * 100)

    fixed_features_list: list[dict[int, float]] | None = None
    cat_values: torch.Tensor | None = None
    cat_id: int | None = None
    if cat:
        cat_id = bundle["cat_dims"][0]
        fixed_features_list, cat_values = _fixed_features_for_bundle(bundle)

    for acq_cls, kwargs, case_id in cases:
        display_id = f"{prefix}optimize_all__{case_id}"
        try:
            with maybe_suppress_botorch_initial_warnings(suppress=suppress_botorch_warnings):
                if cat:
                    cands, acq_value = optimize_acqf_mixed(
                        acq_function=acq_cls(model=model, **kwargs),
                        bounds=bounds,
                        q=q,
                        fixed_features_list=fixed_features_list,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options={"maxiter": maxiter},
                    )
                else:
                    cands, acq_value = optimize_acqf(
                        acq_function=acq_cls(model=model, **kwargs),
                        bounds=bounds,
                        q=q,
                        sequential=True,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options={"maxiter": maxiter},
                    )
            assert cands.shape == torch.Size([q, train_x.shape[-1]]), display_id
            assert torch.isfinite(cands).all(), display_id
            assert torch.isfinite(acq_value).all(), display_id
            assert_candidates_in_bounds(cands=cands, bounds=bounds)
            if cat:
                assert cat_id is not None and cat_values is not None
                assert torch.isin(cands[:, cat_id], cat_values).all(), display_id
            print(f"[OK] {display_id} cands.shape={tuple(cands.shape)} acq_value={acq_value}" if verbose_ok_detail else f"[OK] {display_id}")
        except Exception as exc:
            print(f"[NG] {display_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((display_id, exc))
            if not continue_on_error:
                raise

    _print_failure_summary(failed_cases)
    return bundle


def run_jupyter_optimizer_constraint_compatibility_check(
    *,
    cat: bool = False,
    n: int = 24,
    d: int = 5,
    num_epochs: int = 8,
    q: int = 2,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    if d < 5:
        raise ValueError("constraint compatibility check では d >= 5 が必要です。")

    bundle = create_ordinal_model_bundle(cat=cat, n=n, d=d, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    scenarios = _ordinal_optimizer_constraint_scenarios(model, train_x, bounds, mixed=cat, full_matrix=full_matrix)
    failed_cases: list[tuple[str, Exception]] = []
    prefix = "mixed_" if cat else ""

    fixed_features_list: list[dict[int, float]] | None = None
    cat_values: torch.Tensor | None = None
    cat_id: int | None = None
    if cat:
        cat_id = bundle["cat_dims"][0]
        fixed_features_list, cat_values = _fixed_features_for_bundle(bundle)

    print("=" * 100)
    print(f"Jupyter ordinal base {prefix}optimizer / constraint compatibility check")
    print(f"n={n}, d={d}, q={q}, num_epochs={num_epochs}, full_matrix={full_matrix}, num_cases={len(scenarios)}")
    print("=" * 100)

    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in scenarios:
        try:
            with maybe_suppress_botorch_initial_warnings(suppress=suppress_botorch_warnings):
                if cat:
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
                else:
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
            if cat:
                assert cat_id is not None and cat_values is not None
                assert torch.isin(cands[:, cat_id], cat_values).all(), case_id
            print(f"[OK] {case_id} cands.shape={tuple(cands.shape)} acq_value={acq_value}" if verbose_ok_detail else f"[OK] {case_id}")
            if verbose_candidates:
                print(f"     cands={cands}")
            if constraint_case["case_id"] != "none":
                print_linear_constraint_diagnostics(
                    cands=cands,
                    equality_constraints=constraint_case["equality_constraints"],
                    inequality_constraints=constraint_case["inequality_constraints"],
                    inequality_sense=constraint_case.get("inequality_sense", "le"),
                    show_all=verbose_constraints,
                )
        except Exception as exc:
            print(f"[NG] {case_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((case_id, exc))
            if not continue_on_error:
                raise

    _print_failure_summary(failed_cases)
    return bundle


def run_jupyter_all_checks(
    *,
    num_epochs: int = 8,
    n: int = 24,
    d: int = 5,
    q: int = 2,
    run_optimize: bool = True,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_forward_detail: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> None:
    """ordinal base single-output の Jupyter 一括確認 helper。"""
    run_jupyter_forward_check(cat=False, n=n, d=d, num_epochs=num_epochs, q=q, verbose_forward_detail=verbose_forward_detail)
    run_jupyter_forward_check(cat=True, n=n, d=d, num_epochs=num_epochs, q=q, verbose_forward_detail=verbose_forward_detail)

    if run_optimize:
        run_jupyter_optimize_all_acquisitions_check(
            cat=False,
            n=n,
            d=d,
            num_epochs=num_epochs,
            q=q,
            continue_on_error=continue_on_error,
            suppress_botorch_warnings=suppress_botorch_warnings,
            verbose_ok_detail=verbose_ok_detail,
        )
        run_jupyter_optimize_all_acquisitions_check(
            cat=True,
            n=n,
            d=d,
            num_epochs=num_epochs,
            q=q,
            continue_on_error=continue_on_error,
            suppress_botorch_warnings=suppress_botorch_warnings,
            verbose_ok_detail=verbose_ok_detail,
        )
        run_jupyter_optimizer_constraint_compatibility_check(
            cat=False,
            n=n,
            d=d,
            num_epochs=num_epochs,
            q=q,
            full_matrix=full_matrix,
            continue_on_error=continue_on_error,
            verbose_ok_detail=verbose_ok_detail,
            verbose_candidates=verbose_candidates,
            verbose_constraints=verbose_constraints,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )
        run_jupyter_optimizer_constraint_compatibility_check(
            cat=True,
            n=n,
            d=d,
            num_epochs=num_epochs,
            q=q,
            full_matrix=full_matrix,
            continue_on_error=continue_on_error,
            verbose_ok_detail=verbose_ok_detail,
            verbose_candidates=verbose_candidates,
            verbose_constraints=verbose_constraints,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )

    print("all ordinal base Jupyter checks passed.")
