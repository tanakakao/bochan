from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.deep import (
    DeepKernelBinaryClassificationGPModel,
    DeepKernelBinaryClassificationMixedGPModel,
)
from tests.test_binary_classification_base_single_output import (
    DTYPE,
    DEVICE,
    acquisition_cases,
    assert_candidates_in_bounds,
    make_binary_toy_data,
    make_random_batch,
    make_random_mixed_batch,
    maybe_suppress_botorch_initial_warnings,
)


def _build_input_transform(train_x: torch.Tensor, bounds: torch.Tensor, cat_dims: list[int]) -> Normalize:
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(d=train_x.shape[-1], bounds=bounds, indices=cont_indices)


def _assert_deepkernel_model_training(
    model: Any,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
) -> None:
    """DeepKernel wrapper と inner SVGP の train data 整合性を確認する。"""
    model.eval()
    assert model.train_inputs[0].shape == train_x.shape
    assert model.train_inputs_raw[0].shape == train_x.shape
    assert torch.allclose(model.train_inputs[0], train_x)
    assert torch.allclose(model.train_inputs_raw[0], train_x)
    assert torch.allclose(model.train_targets, train_y.reshape(-1))

    with torch.no_grad():
        expected_inner_x = model.input_transform(train_x) if model.input_transform is not None else train_x
        if isinstance(expected_inner_x, tuple):
            expected_inner_x = expected_inner_x[0]
        posterior = model.posterior(train_x)
        latent_posterior = model.latent_posterior(train_x)

    assert model.model.train_inputs[0].shape == expected_inner_x.shape
    assert torch.allclose(model.model.train_inputs[0], expected_inner_x)
    assert torch.allclose(model.model.train_targets, train_y.reshape(-1))

    assert posterior.mean.shape == train_y.shape
    assert posterior.variance.shape == train_y.shape
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(latent_posterior.mean).all()
    assert (posterior.mean >= 0.0).all() and (posterior.mean <= 1.0).all()

    if cat_dims:
        assert list(model.cat_dims) == cat_dims
        assert list(model.model.cat_dims) == cat_dims
        cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=train_x.dtype, device=train_x.device)
        for cat_id in cat_dims:
            assert torch.isin(model.train_inputs[0][:, cat_id], cat_values).all()
            assert torch.isin(model.train_inputs_raw[0][:, cat_id], cat_values).all()
            assert torch.isin(model.model.train_inputs[0][:, cat_id], cat_values).all()


def create_binary_deepkernel_model_bundle(*, cat: bool = False) -> dict[str, Any]:
    train_x, train_y, bounds = make_binary_toy_data(n=16, d=5, cat=cat)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    kwargs: dict[str, Any] = {
        "train_X": train_x,
        "train_Y": train_y,
        "input_transform": _build_input_transform(train_x, bounds, cat_dims),
        "num_inducing_points": 8,
    }

    torch.manual_seed(0)
    if cat:
        kwargs["cat_dims"] = cat_dims
        model = DeepKernelBinaryClassificationMixedGPModel(**kwargs)
    else:
        model = DeepKernelBinaryClassificationGPModel(**kwargs)

    fit_binary_classifier_mll(model.make_mll(), num_epochs=8, lr=0.01)
    _assert_deepkernel_model_training(model, train_x, train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def binary_deepkernel_model_bundle() -> dict[str, Any]:
    return create_binary_deepkernel_model_bundle(cat=False)


@pytest.fixture(scope="module")
def binary_deepkernel_mixed_model_bundle() -> dict[str, Any]:
    return create_binary_deepkernel_model_bundle(cat=True)


def _representative_acquisition_cases(model: Any, train_x: torch.Tensor):
    names = {"predictive_entropy", "latent_straddle", "pof", "binary_ei", "binary_pi", "binary_ucb"}
    return [case for case in acquisition_cases(model=model, train_x=train_x) if case[2] in names]


def test_binary_deepkernel_model_basic_behavior(binary_deepkernel_model_bundle: dict[str, Any]) -> None:
    _assert_deepkernel_model_training(
        binary_deepkernel_model_bundle["model"],
        binary_deepkernel_model_bundle["train_x"],
        binary_deepkernel_model_bundle["train_y"],
        cat_dims=binary_deepkernel_model_bundle["cat_dims"],
    )


def test_binary_deepkernel_mixed_model_basic_behavior(binary_deepkernel_mixed_model_bundle: dict[str, Any]) -> None:
    _assert_deepkernel_model_training(
        binary_deepkernel_mixed_model_bundle["model"],
        binary_deepkernel_mixed_model_bundle["train_x"],
        binary_deepkernel_mixed_model_bundle["train_y"],
        cat_dims=binary_deepkernel_mixed_model_bundle["cat_dims"],
    )


def test_binary_deepkernel_acquisition_forward_shapes(binary_deepkernel_model_bundle: dict[str, Any]) -> None:
    model = binary_deepkernel_model_bundle["model"]
    train_x = binary_deepkernel_model_bundle["train_x"]
    X = make_random_batch(binary_deepkernel_model_bundle["bounds"], batch_size=4, q=2)

    for acq_cls, kwargs, case_id in acquisition_cases(model=model, train_x=train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_binary_deepkernel_mixed_acquisition_forward_shapes(binary_deepkernel_mixed_model_bundle: dict[str, Any]) -> None:
    model = binary_deepkernel_mixed_model_bundle["model"]
    train_x = binary_deepkernel_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(
        binary_deepkernel_mixed_model_bundle["bounds"],
        binary_deepkernel_mixed_model_bundle["cat_dims"],
        batch_size=4,
        q=2,
    )

    for acq_cls, kwargs, case_id in acquisition_cases(model=model, train_x=train_x):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


@pytest.mark.slow
def test_binary_deepkernel_optimize_acqf_representative_smoke(
    binary_deepkernel_model_bundle: dict[str, Any],
) -> None:
    model = binary_deepkernel_model_bundle["model"]
    train_x = binary_deepkernel_model_bundle["train_x"]
    bounds = binary_deepkernel_model_bundle["bounds"]
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
def test_binary_deepkernel_mixed_optimize_acqf_mixed_representative_smoke(
    binary_deepkernel_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_deepkernel_mixed_model_bundle["model"]
    train_x = binary_deepkernel_mixed_model_bundle["train_x"]
    bounds = binary_deepkernel_mixed_model_bundle["bounds"]
    cat_id = binary_deepkernel_mixed_model_bundle["cat_dims"][0]
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


def run_jupyter_all_checks() -> dict[str, Any]:
    """Jupyter 向け: single / mixed の主要 forward check をまとめて実行する。"""
    single_bundle = create_binary_deepkernel_model_bundle(cat=False)
    mixed_bundle = create_binary_deepkernel_model_bundle(cat=True)
    test_binary_deepkernel_acquisition_forward_shapes(single_bundle)
    test_binary_deepkernel_mixed_acquisition_forward_shapes(mixed_bundle)
    return {"single": single_bundle, "mixed": mixed_bundle}
