from __future__ import annotations

"""Binary classification base multi-output smoke tests.

The multi-output model is built as a list of independent single-output binary
classifiers and wrapped by ``MultiOutputBinaryClassificationModel``.  This file
mirrors the single-output base test while exercising multi-output active
learning, level-set estimation, Bayesian optimization acquisitions, and
Jupyter-oriented optimization / constraint compatibility runners.
"""

from typing import Any, Optional

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from gpytorch.mlls.variational_elbo import VariationalELBO

from bochan.acquisition.binary.active_learning import (
    qMultiOutputBinaryBALD,
    qMultiOutputBinaryIntegratedPosteriorVarianceProxy,
    qMultiOutputBinaryMarginUncertainty,
    qMultiOutputBinaryPredictiveEntropy,
    qMultiOutputBinaryProbabilityVariance,
)
from bochan.acquisition.binary.bayesian_optimization import (
    qMultiOutputBinaryExpectedHypervolumeImprovement,
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qMultiOutputBinaryNParEGO,
    qMultiOutputBinaryProbabilityOfFeasibility,
)
from bochan.acquisition.binary.levelset_estimation import (
    qMultiOutputBinaryBoundaryVarianceAcquisition,
    qMultiOutputBinaryClassEntropyAcquisition,
    qMultiOutputBinaryICUAcquisition,
    qMultiOutputBinaryJointLatentStraddleAcquisition,
    qMultiOutputBinaryLatentStraddleAcquisition,
)
from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.base import (
    BinaryClassificationGPModel,
    BinaryClassificationMixedGPModel,
    MultiOutputBinaryClassificationModel,
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


N_OUTPUTS = 3


def make_multi_output_binary_toy_data(
    n: int = 20,
    d: int = 5,
    cat: bool = False,
    m: int = N_OUTPUTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-output binary classification 用の toy data を作る。"""
    train_x, _, bounds = make_binary_toy_data(n=n, d=d, cat=cat)
    cont_x = train_x[..., :d]

    scores: list[torch.Tensor] = [
        0.9 * cont_x[..., 0] - 0.7 * cont_x[..., 1] + 0.4 * cont_x[..., 2 % d],
        -0.5 * cont_x[..., 0] + 0.8 * cont_x[..., 1] + 0.6 * cont_x[..., 3 % d],
        torch.sin(3.0 * cont_x[..., 0])
        + 0.5 * cont_x[..., 2 % d]
        - 0.3 * cont_x[..., 4 % d],
    ]

    if cat:
        cat_signal = (train_x[..., -1] - 10.0) / 5.0
        scores[0] = scores[0] + 0.15 * cat_signal
        scores[1] = scores[1] - 0.10 * cat_signal
        scores[2] = scores[2] + 0.20 * cat_signal

    if m > len(scores):
        for j in range(len(scores), m):
            weights = torch.linspace(
                0.1 + 0.05 * j,
                0.5 + 0.05 * j,
                d,
                dtype=train_x.dtype,
                device=train_x.device,
            )
            scores.append((cont_x * weights).sum(dim=-1))

    labels = [
        (score > score.median()).to(dtype=train_x.dtype).unsqueeze(-1)
        for score in scores[:m]
    ]
    return train_x, torch.cat(labels, dim=-1), bounds


def _build_input_transform(
    train_x: torch.Tensor,
    bounds: torch.Tensor,
    cat_dims: list[int],
) -> Normalize:
    """single-output submodel ごとに raw-space Normalize を作る。"""
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(d=train_x.shape[-1], bounds=bounds, indices=cont_indices)


def _fit_single_output_binary_model(
    model: Any,
    *,
    num_epochs: int,
    lr: float = 0.01,
) -> None:
    """single-output binary classifier を VariationalELBO で fit する。"""
    mll = VariationalELBO(
        likelihood=model.likelihood,
        model=model.model,
        num_data=model.model.train_inputs[0].shape[-2],
    )
    fit_binary_classifier_mll(mll, num_epochs=num_epochs, lr=lr)


def _assert_single_submodel_training(
    submodel: Any,
    train_x: torch.Tensor,
    train_y_j: torch.Tensor,
    *,
    cat_dims: list[int],
    output_index: int,
) -> None:
    """MultiOutput wrapper 内の single-output submodel を確認する。"""
    submodel.eval()
    assert submodel.num_outputs == 1
    assert submodel.train_inputs_raw[0].shape == train_x.shape
    assert torch.allclose(submodel.train_inputs_raw[0], train_x)
    assert submodel.train_inputs_raw[0].data_ptr() != train_x.data_ptr()
    assert torch.allclose(submodel.train_targets, train_y_j.reshape(-1))

    with torch.no_grad():
        expected_train_inputs = (
            submodel.input_transform(train_x)
            if getattr(submodel, "input_transform", None) is not None
            else train_x
        )
        if isinstance(expected_train_inputs, tuple):
            expected_train_inputs = expected_train_inputs[0]
        posterior = submodel.posterior(train_x)
        latent_posterior = submodel.latent_posterior(train_x)

    assert submodel.train_inputs[0].shape == expected_train_inputs.shape, output_index
    assert submodel.model.train_inputs[0].shape == expected_train_inputs.shape, output_index
    assert torch.allclose(submodel.train_inputs[0], expected_train_inputs), output_index
    assert torch.allclose(submodel.model.train_inputs[0], expected_train_inputs), output_index
    assert torch.allclose(submodel.model.train_targets, train_y_j.reshape(-1)), output_index
    assert posterior.mean.shape == train_y_j.shape, output_index
    assert posterior.variance.shape == train_y_j.shape, output_index
    assert torch.isfinite(posterior.mean).all(), output_index
    assert torch.isfinite(posterior.variance).all(), output_index
    assert (posterior.mean >= 0.0).all() and (posterior.mean <= 1.0).all(), output_index
    assert torch.isfinite(latent_posterior.mean).all(), output_index

    if cat_dims:
        assert list(submodel.cat_dims) == cat_dims
        cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=train_x.dtype, device=train_x.device)
        for cat_id in cat_dims:
            assert torch.isin(submodel.train_inputs_raw[0][:, cat_id], cat_values).all()
            assert torch.isin(submodel.train_inputs[0][:, cat_id], cat_values).all()
            assert torch.isin(submodel.model.train_inputs[0][:, cat_id], cat_values).all()


def _assert_multi_output_model_training(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
) -> None:
    """MultiOutputBinaryClassificationModel の基本状態を確認する。"""
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
        _assert_single_submodel_training(
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


def create_multi_output_binary_model_bundle(
    *,
    cat: bool = False,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    input_transform: Optional[Normalize] = None,
) -> dict[str, Any]:
    """出力ごとに single-output classifier を fit して multi-output wrapper を作る。"""
    train_x, train_y, bounds = make_multi_output_binary_toy_data(n=n, d=d, cat=cat, m=m)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    model_cls = BinaryClassificationMixedGPModel if cat else BinaryClassificationGPModel

    models: list[Any] = []
    for j in range(train_y.shape[-1]):
        sub_input_transform = (
            _build_input_transform(train_x, bounds, cat_dims)
            if input_transform is None
            else input_transform
        )
        kwargs: dict[str, Any] = {
            "train_X": train_x,
            "train_Y": train_y[:, [j]],
            "input_transform": sub_input_transform,
            "num_inducing_points": 8,
        }
        if cat:
            kwargs["cat_dims"] = cat_dims
        submodel = model_cls(**kwargs)
        _fit_single_output_binary_model(submodel, num_epochs=num_epochs, lr=0.01)
        models.append(submodel)

    model = MultiOutputBinaryClassificationModel(*models)
    _assert_multi_output_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def multi_output_binary_model_bundle() -> dict[str, Any]:
    return create_multi_output_binary_model_bundle(cat=False, n=20, d=5, m=N_OUTPUTS, num_epochs=4)


@pytest.fixture(scope="module")
def multi_output_binary_mixed_model_bundle() -> dict[str, Any]:
    return create_multi_output_binary_model_bundle(cat=True, n=20, d=5, m=N_OUTPUTS, num_epochs=4)


def _bo_reference_objects(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
) -> dict[str, Any]:
    """EHVI / NEHVI / NParEGO 用の軽量 baseline objects を作る。"""
    with torch.no_grad():
        y_baseline = model.probability_posterior(train_x).mean.reshape(-1, model.num_outputs)
        y_baseline = y_baseline.clamp(1e-6, 1.0 - 1e-6)

    ref_point = torch.full((model.num_outputs,), -0.05, dtype=train_x.dtype, device=train_x.device)
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=y_baseline)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
    weights = torch.ones(model.num_outputs, dtype=train_x.dtype, device=train_x.device)
    weights = weights / weights.sum()
    return {
        "ref_point": ref_point,
        "ref_point_list": ref_point.detach().cpu().tolist(),
        "partitioning": partitioning,
        "sampler": sampler,
        "weights": weights,
    }


def multi_output_active_learning_acquisition_cases(
    model: MultiOutputBinaryClassificationModel,
) -> list[tuple[type, dict[str, Any], str]]:
    output_weights = torch.ones(model.num_outputs, dtype=DTYPE, device=DEVICE)
    common_kwargs: dict[str, Any] = {
        "reduction": "mean",
        "output_mode": "mean",
        "pending_penalty_weight": 0.01,
        "pending_penalty_beta": 5.0,
        "apply_sigmoid_if_needed": False,
    }
    return [
        (qMultiOutputBinaryPredictiveEntropy, dict(common_kwargs), "al_predictive_entropy"),
        (qMultiOutputBinaryProbabilityVariance, dict(common_kwargs), "al_probability_variance"),
        (qMultiOutputBinaryMarginUncertainty, dict(common_kwargs), "al_margin_uncertainty"),
        (
            qMultiOutputBinaryBALD,
            {
                "reduction": "mean",
                "output_mode": "mean",
                "num_samples": 8,
                "pending_penalty_weight": 0.01,
                "pending_penalty_beta": 5.0,
                "apply_sigmoid_if_needed": True,
            },
            "al_bald",
        ),
        (qMultiOutputBinaryIntegratedPosteriorVarianceProxy, dict(common_kwargs), "al_integrated_posterior_variance_proxy"),
        (
            qMultiOutputBinaryProbabilityVariance,
            {**common_kwargs, "output_mode": "weighted_mean", "output_weights": output_weights},
            "al_probability_variance_weighted_mean",
        ),
        (
            qMultiOutputBinaryPredictiveEntropy,
            {**common_kwargs, "output_mode": "all_positive"},
            "al_predictive_entropy_all_positive",
        ),
    ]


def multi_output_levelset_acquisition_cases(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    output_weights = torch.ones(model.num_outputs, dtype=train_x.dtype, device=train_x.device)
    common_kwargs: dict[str, Any] = {
        "reduction": "mean",
        "output_mode": "mean",
        "pending_penalty_weight": 0.01,
        "pending_penalty_beta": 5.0,
    }
    return [
        (qMultiOutputBinaryClassEntropyAcquisition, {**common_kwargs, "apply_sigmoid_if_needed": False}, "lse_class_entropy"),
        (qMultiOutputBinaryICUAcquisition, {**common_kwargs, "apply_sigmoid_if_needed": False}, "lse_icu"),
        (qMultiOutputBinaryBoundaryVarianceAcquisition, {**common_kwargs, "thresholds": 0.0, "tau": 1.0}, "lse_boundary_variance"),
        (qMultiOutputBinaryLatentStraddleAcquisition, {**common_kwargs, "thresholds": 0.0, "beta": 1.0}, "lse_latent_straddle"),
        (
            qMultiOutputBinaryLatentStraddleAcquisition,
            {
                **common_kwargs,
                "thresholds": 0.0,
                "beta": 1.0,
                "output_mode": "weighted_mean",
                "output_weights": output_weights,
            },
            "lse_latent_straddle_weighted_mean",
        ),
        (
            qMultiOutputBinaryJointLatentStraddleAcquisition,
            {
                "beta": 1.0,
                "thresholds": 0.0,
                "uncertainty_mode": "sqrt_trace",
                "boundary_mode": "l2_mean",
                "same_batch_penalty_weight": 0.01,
                "pending_penalty_weight": 0.01,
                "observed_penalty_weight": 0.0,
                "X_observed": train_x,
            },
            "lse_joint_latent_straddle",
        ),
    ]


def multi_output_bo_acquisition_cases(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    refs = _bo_reference_objects(model, train_x)
    return [
        (
            qMultiOutputBinaryProbabilityOfFeasibility,
            {
                "num_samples": 8,
                "threshold": 0.0,
                "mode": "mc_sigmoid",
                "reduction": "mean",
                "output_mode": "all_positive",
                "pending_penalty_weight": 0.01,
                "pending_penalty_beta": 5.0,
                "samples_are_probs": False,
                "apply_sigmoid_if_needed": True,
            },
            "bo_probability_of_feasibility",
        ),
        (
            qMultiOutputBinaryExpectedHypervolumeImprovement,
            {
                "ref_point": refs["ref_point_list"],
                "partitioning": refs["partitioning"],
                "sampler": refs["sampler"],
            },
            "bo_qehvi",
        ),
        (
            qMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
            {
                "ref_point": refs["ref_point_list"],
                "X_baseline": train_x,
                "sampler": refs["sampler"],
            },
            "bo_qnehvi",
        ),
        (
            qMultiOutputBinaryNParEGO,
            {
                "X_baseline": train_x,
                "ref_point": refs["ref_point"],
                "weights": refs["weights"],
                "sampler": refs["sampler"],
                "samples_are_probs": False,
                "apply_sigmoid_if_needed": True,
            },
            "bo_nparego",
        ),
    ]


def multi_output_acquisition_cases(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    """Multi-output binary classification の全 acquisition family case。"""
    return (
        multi_output_active_learning_acquisition_cases(model)
        + multi_output_levelset_acquisition_cases(model, train_x)
        + multi_output_bo_acquisition_cases(model, train_x)
    )


def _representative_multi_output_acquisition_cases(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    names = {
        "al_probability_variance",
        "al_bald",
        "lse_latent_straddle",
        "lse_icu",
        "bo_probability_of_feasibility",
        "bo_nparego",
    }
    return [case for case in multi_output_acquisition_cases(model, train_x) if case[2] in names]


def _constraint_multi_output_acquisition_cases(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    """制約・optimizer compatibility 用に AL / LSE / BO から代表を選ぶ。"""
    names = {
        "al_probability_variance",
        "lse_icu",
        "bo_probability_of_feasibility",
    }
    return [case for case in multi_output_acquisition_cases(model, train_x) if case[2] in names]


def _representative_constraint_cases(bounds: torch.Tensor) -> list[dict[str, Any]]:
    """制約なし / step のみ / 線形制約 / step+k-sparse+線形制約を返す。"""
    names = {"none", "step_only", "constraints_only", "step_k_sparse_constraints"}
    return [case for case in make_constraint_cases(bounds) if case["case_id"] in names]


def _get_acquisition_case(model: MultiOutputBinaryClassificationModel, train_x: torch.Tensor, case_id: str):
    for acq_cls, kwargs, current_case_id in multi_output_acquisition_cases(model, train_x):
        if current_case_id == case_id:
            return acq_cls, kwargs
    raise AssertionError(f"multi-output acquisition case not found: {case_id}")


def _optimizer_constraint_scenarios(
    model: MultiOutputBinaryClassificationModel,
    train_x: torch.Tensor,
    bounds: torch.Tensor,
    *,
    mixed: bool = False,
    full_matrix: bool = False,
):
    """optimizer × constraint × acquisition の compatibility scenario を作る。"""
    scenarios = []
    acquisition_cases_for_constraints = (
        _representative_multi_output_acquisition_cases(model, train_x)
        if full_matrix
        else _constraint_multi_output_acquisition_cases(model, train_x)
    )
    constraint_cases = make_constraint_cases(bounds) if full_matrix else _representative_constraint_cases(bounds)

    for acq_cls, kwargs, acq_id in acquisition_cases_for_constraints:
        for optimize_func, optimize_method, optimizer_id in optimizer_cases():
            for constraint_case in constraint_cases:
                case_id = f"{acq_id}__{optimizer_id}__{constraint_case['case_id']}"
                if mixed:
                    case_id = f"mixed__{case_id}"
                scenarios.append((acq_cls, kwargs, acq_id, optimize_func, optimize_method, constraint_case, case_id))
    return scenarios


def test_multi_output_binary_model_basic_behavior(multi_output_binary_model_bundle: dict[str, Any]) -> None:
    _assert_multi_output_model_training(
        model=multi_output_binary_model_bundle["model"],
        train_x=multi_output_binary_model_bundle["train_x"],
        train_y=multi_output_binary_model_bundle["train_y"],
        cat_dims=multi_output_binary_model_bundle["cat_dims"],
    )


def test_multi_output_binary_mixed_model_basic_behavior(multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    _assert_multi_output_model_training(
        model=multi_output_binary_mixed_model_bundle["model"],
        train_x=multi_output_binary_mixed_model_bundle["train_x"],
        train_y=multi_output_binary_mixed_model_bundle["train_y"],
        cat_dims=multi_output_binary_mixed_model_bundle["cat_dims"],
    )


def test_multi_output_binary_acquisition_forward_shapes(multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = multi_output_binary_model_bundle["model"]
    train_x = multi_output_binary_model_bundle["train_x"]
    X = make_random_batch(multi_output_binary_model_bundle["bounds"], batch_size=4, q=2)

    for acq_cls, kwargs, case_id in multi_output_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_multi_output_binary_mixed_acquisition_forward_shapes(multi_output_binary_mixed_model_bundle: dict[str, Any]) -> None:
    model = multi_output_binary_mixed_model_bundle["model"]
    train_x = multi_output_binary_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(
        multi_output_binary_mixed_model_bundle["bounds"],
        multi_output_binary_mixed_model_bundle["cat_dims"],
        batch_size=4,
        q=2,
    )

    for acq_cls, kwargs, case_id in multi_output_acquisition_cases(model, train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_multi_output_binary_family_case_coverage(multi_output_binary_model_bundle: dict[str, Any]) -> None:
    """active_learning / levelset / bayesopt の case がすべて含まれることを確認する。"""
    model = multi_output_binary_model_bundle["model"]
    train_x = multi_output_binary_model_bundle["train_x"]
    case_ids = {case_id for _, _, case_id in multi_output_acquisition_cases(model, train_x)}
    assert any(case_id.startswith("al_") for case_id in case_ids)
    assert any(case_id.startswith("lse_") for case_id in case_ids)
    assert any(case_id.startswith("bo_") for case_id in case_ids)
    assert "bo_qehvi" in case_ids
    assert "bo_qnehvi" in case_ids
    assert "bo_nparego" in case_ids


def test_multi_output_binary_constraint_scenario_coverage(multi_output_binary_model_bundle: dict[str, Any]) -> None:
    """制約テストが AL / LSE / BO と evo optimizer を含むことを確認する。"""
    model = multi_output_binary_model_bundle["model"]
    train_x = multi_output_binary_model_bundle["train_x"]
    bounds = multi_output_binary_model_bundle["bounds"]
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
def test_multi_output_binary_optimize_acqf_representative_smoke(multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = multi_output_binary_model_bundle["model"]
    train_x = multi_output_binary_model_bundle["train_x"]
    bounds = multi_output_binary_model_bundle["bounds"]
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
def test_multi_output_binary_mixed_optimize_acqf_mixed_representative_smoke(
    multi_output_binary_mixed_model_bundle: dict[str, Any],
) -> None:
    model = multi_output_binary_mixed_model_bundle["model"]
    train_x = multi_output_binary_mixed_model_bundle["train_x"]
    bounds = multi_output_binary_mixed_model_bundle["bounds"]
    cat_id = multi_output_binary_mixed_model_bundle["cat_dims"][0]
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
def test_multi_output_binary_optimizer_constraint_case_smoke(multi_output_binary_model_bundle: dict[str, Any]) -> None:
    model = multi_output_binary_model_bundle["model"]
    train_x = multi_output_binary_model_bundle["train_x"]
    bounds = multi_output_binary_model_bundle["bounds"]
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
def test_multi_output_binary_mixed_optimizer_constraint_case_smoke(
    multi_output_binary_mixed_model_bundle: dict[str, Any],
) -> None:
    model = multi_output_binary_mixed_model_bundle["model"]
    train_x = multi_output_binary_mixed_model_bundle["train_x"]
    bounds = multi_output_binary_mixed_model_bundle["bounds"]
    cat_id = multi_output_binary_mixed_model_bundle["cat_dims"][0]
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


def run_jupyter_forward_check(
    *,
    cat: bool = False,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    batch_size: int = 4,
    q: int = 2,
    verbose_forward_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_multi_output_binary_model_bundle(cat=cat, n=n, d=d, m=m, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]
    X = make_random_mixed_batch(bounds, cat_dims, batch_size=batch_size, q=q) if cat else make_random_batch(bounds, batch_size=batch_size, q=q)

    print("=" * 80)
    print(f"Jupyter multi-output binary forward check cat={cat}")
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
    print("all multi-output binary forward checks passed.")


def run_jupyter_optimize_acqf_all_acquisitions_check(
    *,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    q: int = 2,
    num_restarts: int = 2,
    raw_samples: int = 16,
    maxiter: int = 10,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
    verbose_ok_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_multi_output_binary_model_bundle(cat=False, n=n, d=d, m=m, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cases = multi_output_acquisition_cases(model, train_x)
    failed_cases: list[tuple[str, Exception]] = []

    print("=" * 100)
    print("Jupyter multi-output optimize_acqf check: all acquisitions")
    print(f"n={n}, d={d}, m={m}, q={q}, num_epochs={num_epochs}, num_acquisitions={len(cases)}")
    print("=" * 100)

    for acq_cls, kwargs, case_id in cases:
        display_id = f"optimize_acqf__{case_id}"
        try:
            with maybe_suppress_botorch_initial_warnings(suppress=suppress_botorch_warnings):
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
            print(f"[OK] {display_id} cands.shape={tuple(cands.shape)} acq_value={acq_value}" if verbose_ok_detail else f"[OK] {display_id}")
        except Exception as exc:
            print(f"[NG] {display_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((display_id, exc))
            if not continue_on_error:
                raise

    _print_failure_summary(failed_cases)
    return bundle


def run_jupyter_optimize_acqf_mixed_all_acquisitions_check(
    *,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
    q: int = 2,
    num_restarts: int = 2,
    raw_samples: int = 16,
    maxiter: int = 10,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
    verbose_ok_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_multi_output_binary_model_bundle(cat=True, n=n, d=d, m=m, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_id = bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    cases = multi_output_acquisition_cases(model, train_x)
    failed_cases: list[tuple[str, Exception]] = []

    print("=" * 100)
    print("Jupyter mixed multi-output optimize_acqf_mixed check: all acquisitions")
    print(f"n={n}, d={d}, m={m}, q={q}, num_epochs={num_epochs}, cat_id={cat_id}, num_acquisitions={len(cases)}")
    print("=" * 100)

    for acq_cls, kwargs, case_id in cases:
        display_id = f"optimize_acqf_mixed__{case_id}"
        try:
            with maybe_suppress_botorch_initial_warnings(suppress=suppress_botorch_warnings):
                cands, acq_value = optimize_acqf_mixed(
                    acq_function=acq_cls(model=model, **kwargs),
                    bounds=bounds,
                    q=q,
                    fixed_features_list=fixed_features_list,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"maxiter": maxiter},
                )
            assert cands.shape == torch.Size([q, train_x.shape[-1]]), display_id
            assert torch.isfinite(cands).all(), display_id
            assert torch.isfinite(acq_value).all(), display_id
            assert_candidates_in_bounds(cands=cands, bounds=bounds)
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
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
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

    bundle = create_multi_output_binary_model_bundle(cat=False, n=n, d=d, m=m, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    scenarios = _optimizer_constraint_scenarios(model=model, train_x=train_x, bounds=bounds, full_matrix=full_matrix)
    failed_cases: list[tuple[str, Exception]] = []

    print("=" * 100)
    print("Jupyter multi-output optimizer / constraint compatibility check")
    print(f"n={n}, d={d}, m={m}, q={q}, num_epochs={num_epochs}, full_matrix={full_matrix}, num_cases={len(scenarios)}")
    print("=" * 100)

    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in scenarios:
        try:
            with maybe_suppress_botorch_initial_warnings(suppress=suppress_botorch_warnings):
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
            assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=q, d=train_x.shape[-1], constraint_case=constraint_case, case_id=case_id)
            print(f"[OK] {case_id} cands.shape={tuple(cands.shape)} acq_value={acq_value}" if verbose_ok_detail else f"[OK] {case_id}")
            if verbose_candidates:
                print(f"     cands={cands}")
                if train_x.shape[-1] >= 5:
                    print(f"     sum_0_1_2={cands[:, :3].sum(dim=1)}")
                    print(f"     sum_3_4={cands[:, 3:5].sum(dim=1)}")
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


def run_jupyter_mixed_optimizer_constraint_compatibility_check(
    *,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
    num_epochs: int = 4,
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

    bundle = create_multi_output_binary_model_bundle(cat=True, n=n, d=d, m=m, num_epochs=num_epochs)
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_id = bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    scenarios = _optimizer_constraint_scenarios(model=model, train_x=train_x, bounds=bounds, mixed=True, full_matrix=full_matrix)
    failed_cases: list[tuple[str, Exception]] = []

    print("=" * 100)
    print("Jupyter mixed multi-output optimizer / constraint compatibility check")
    print(f"n={n}, d={d}, m={m}, q={q}, num_epochs={num_epochs}, cat_id={cat_id}, full_matrix={full_matrix}, num_cases={len(scenarios)}")
    print("=" * 100)

    for acq_cls, kwargs, _, optimize_func, optimize_method, constraint_case, case_id in scenarios:
        try:
            with maybe_suppress_botorch_initial_warnings(suppress=suppress_botorch_warnings):
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
            assert_optimizer_compatibility_result(cands=cands, acq_value=acq_value, bounds=bounds, q=q, d=train_x.shape[-1], constraint_case=constraint_case, case_id=case_id)
            assert torch.isin(cands[:, cat_id], cat_values).all(), case_id
            print(f"[OK] {case_id} cands.shape={tuple(cands.shape)} acq_value={acq_value}" if verbose_ok_detail else f"[OK] {case_id}")
            if verbose_candidates:
                print(f"     cands={cands}")
                if train_x.shape[-1] >= 5:
                    print(f"     sum_0_1_2={cands[:, :3].sum(dim=1)}")
                    print(f"     sum_3_4={cands[:, 3:5].sum(dim=1)}")
                print(f"     cat_values={cands[:, cat_id]}")
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
    num_epochs: int = 4,
    n: int = 20,
    d: int = 5,
    m: int = N_OUTPUTS,
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
    """multi-output binary classification の Jupyter 一括確認 helper。"""
    run_jupyter_forward_check(
        cat=False,
        n=n,
        d=d,
        m=m,
        num_epochs=num_epochs,
        q=q,
        verbose_forward_detail=verbose_forward_detail,
    )
    run_jupyter_forward_check(
        cat=True,
        n=n,
        d=d,
        m=m,
        num_epochs=num_epochs,
        q=q,
        verbose_forward_detail=verbose_forward_detail,
    )

    if run_optimize:
        run_jupyter_optimize_acqf_all_acquisitions_check(
            n=n,
            d=d,
            m=m,
            num_epochs=num_epochs,
            q=q,
            continue_on_error=continue_on_error,
            suppress_botorch_warnings=suppress_botorch_warnings,
            verbose_ok_detail=verbose_ok_detail,
        )
        run_jupyter_optimize_acqf_mixed_all_acquisitions_check(
            n=n,
            d=d,
            m=m,
            num_epochs=num_epochs,
            q=q,
            continue_on_error=continue_on_error,
            suppress_botorch_warnings=suppress_botorch_warnings,
            verbose_ok_detail=verbose_ok_detail,
        )
        run_jupyter_optimizer_constraint_compatibility_check(
            n=n,
            d=d,
            m=m,
            num_epochs=num_epochs,
            q=q,
            full_matrix=full_matrix,
            continue_on_error=continue_on_error,
            verbose_ok_detail=verbose_ok_detail,
            verbose_candidates=verbose_candidates,
            verbose_constraints=verbose_constraints,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )
        run_jupyter_mixed_optimizer_constraint_compatibility_check(
            n=n,
            d=d,
            m=m,
            num_epochs=num_epochs,
            q=q,
            full_matrix=full_matrix,
            continue_on_error=continue_on_error,
            verbose_ok_detail=verbose_ok_detail,
            verbose_candidates=verbose_candidates,
            verbose_constraints=verbose_constraints,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )

    print("all multi-output binary Jupyter checks passed.")
