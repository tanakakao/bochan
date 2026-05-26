from __future__ import annotations

"""Binary classification DeepGP single-output smoke tests.

This module intentionally reuses the toy data, acquisition case definitions,
optimizer wrappers, and assertion helpers from
``test_binary_classification_base_single_output.py``.  The DeepGP-specific
differences are limited to model construction, the fit helper, and the model
state assertions required by the DeepGP wrapper.
"""

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from bochan.fit import fit_deepgp_mll
from bochan.models.classification.binary.deep import (
    BinaryClassificationDeepGPModel,
    BinaryClassificationMixedDeepGPModel,
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
    print_linear_constraint_diagnostics,
)


def _build_input_transform(
    train_x: torch.Tensor,
    bounds: torch.Tensor,
    cat_dims: list[int],
) -> Normalize:
    """DeepGP / mixed DeepGP 共通の input_transform を作る。"""
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(
        d=train_x.shape[-1],
        bounds=bounds,
        indices=cont_indices,
    )


def _fit_deepgp_binary_model(
    model: Any,
    *,
    num_epochs: int,
    lr: float = 0.01,
) -> None:
    """DeepGP binary classification 専用 fit helper を軽量設定で実行する。"""
    fit_deepgp_mll(
        model.make_mll(),
        num_epochs=num_epochs,
        lr=lr,
    )


def _assert_deepgp_model_training(
    model: Any,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
) -> None:
    """DeepGP wrapper の基本状態を確認する。"""
    model.eval()

    assert model.train_inputs[0].shape == train_x.shape
    assert model.train_inputs_raw[0].shape == train_x.shape
    assert model.train_inputs[0].dtype == train_x.dtype
    assert model.train_inputs_raw[0].dtype == train_x.dtype
    assert model.train_inputs[0].device == train_x.device
    assert model.train_inputs_raw[0].device == train_x.device

    # DeepGP wrapper は train_inputs を raw-space のまま保持する。
    assert torch.allclose(model.train_inputs[0], train_x)
    assert torch.allclose(model.train_inputs_raw[0], train_x)
    assert model.train_inputs_raw[0].data_ptr() != train_x.data_ptr()
    assert torch.allclose(model.train_targets, train_y.reshape(-1))

    with torch.no_grad():
        expected_transformed_x = (
            model.input_transform(train_x)
            if getattr(model, "input_transform", None) is not None
            else train_x
        )
        if isinstance(expected_transformed_x, tuple):
            expected_transformed_x = expected_transformed_x[0]

        transformed_x = model.transform_inputs(train_x)
        posterior = model.posterior(train_x)
        latent_posterior = model.latent_posterior(train_x)

    assert transformed_x.shape == train_x.shape
    assert torch.allclose(transformed_x, expected_transformed_x)

    assert posterior.mean.shape == train_y.shape
    assert posterior.variance.shape == train_y.shape
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert (posterior.mean >= 0.0).all() and (posterior.mean <= 1.0).all()

    assert torch.isfinite(latent_posterior.mean).all()
    assert model.make_mll().model is model

    if cat_dims:
        assert hasattr(model, "cat_dims")
        assert list(model.cat_dims) == cat_dims

        cat_values = torch.tensor(
            [5.0, 10.0, 15.0],
            dtype=train_x.dtype,
            device=train_x.device,
        )

        for cat_id in cat_dims:
            assert torch.isin(model.train_inputs[0][:, cat_id], cat_values).all()
            assert torch.isin(model.train_inputs_raw[0][:, cat_id], cat_values).all()
            assert torch.isin(transformed_x[:, cat_id], cat_values).all()


def create_binary_deepgp_model_bundle(
    *,
    cat: bool = False,
    n: int = 16,
    d: int = 5,
    num_epochs: int = 8,
) -> dict[str, Any]:
    """Jupyter / pytest 共通で使う binary DeepGP model 作成関数。"""
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
        model = BinaryClassificationMixedDeepGPModel(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
            cat_dims=cat_dims,
            hidden_dim=4,
            num_inducing=8,
            num_inducing_last=8,
        )
    else:
        model = BinaryClassificationDeepGPModel(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
            list_hidden_dims=[4],
            num_inducing=8,
        )

    _fit_deepgp_binary_model(
        model,
        num_epochs=num_epochs,
        lr=0.01,
    )
    _assert_deepgp_model_training(
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
def binary_deepgp_model_bundle() -> dict[str, Any]:
    """pytest 用: 通常 binary DeepGP model。"""
    return create_binary_deepgp_model_bundle(
        cat=False,
        n=16,
        d=5,
        num_epochs=8,
    )


@pytest.fixture(scope="module")
def binary_deepgp_mixed_model_bundle() -> dict[str, Any]:
    """pytest 用: mixed binary DeepGP model。"""
    return create_binary_deepgp_model_bundle(
        cat=True,
        n=16,
        d=5,
        num_epochs=8,
    )


def _representative_acquisition_cases(
    model: Any,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    """DeepGP optimizer smoke test 用に代表 acquisition だけを返す。"""
    names = {
        "predictive_entropy",
        "latent_straddle",
        "pof",
        "binary_ei",
        "binary_pi",
        "binary_ucb",
    }
    return [
        case
        for case in acquisition_cases(model=model, train_x=train_x)
        if case[2] in names
    ]


def _representative_constraint_cases(bounds: torch.Tensor) -> list[dict[str, Any]]:
    """DeepGP optimizer constraint smoke test 用の軽量ケースだけを返す。"""
    names = {"step_only", "step_k_sparse_constraints"}
    return [
        case
        for case in make_constraint_cases(bounds)
        if case["case_id"] in names
    ]


def _get_acquisition_case(
    model: Any,
    train_x: torch.Tensor,
    case_id: str,
):
    for acq_cls, kwargs, current_case_id in acquisition_cases(
        model=model,
        train_x=train_x,
    ):
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
        for acq_cls, kwargs, acq_id in _representative_acquisition_cases(
            model,
            train_x,
        ):
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
        train_x=train_x,
        case_id="binary_ucb",
    )
    prefix = "deepgp_mixed" if mixed else "deepgp"
    return [
        (
            acq_cls,
            kwargs,
            "binary_ucb",
            "torch",
            "adam",
            constraint_case,
            f"{prefix}__binary_ucb__torch_adam__{constraint_case['case_id']}",
        )
        for constraint_case in _representative_constraint_cases(bounds)
    ]


def test_binary_deepgp_model_basic_behavior(
    binary_deepgp_model_bundle: dict[str, Any],
) -> None:
    _assert_deepgp_model_training(
        model=binary_deepgp_model_bundle["model"],
        train_x=binary_deepgp_model_bundle["train_x"],
        train_y=binary_deepgp_model_bundle["train_y"],
        cat_dims=binary_deepgp_model_bundle["cat_dims"],
    )


def test_binary_deepgp_mixed_model_basic_behavior(
    binary_deepgp_mixed_model_bundle: dict[str, Any],
) -> None:
    _assert_deepgp_model_training(
        model=binary_deepgp_mixed_model_bundle["model"],
        train_x=binary_deepgp_mixed_model_bundle["train_x"],
        train_y=binary_deepgp_mixed_model_bundle["train_y"],
        cat_dims=binary_deepgp_mixed_model_bundle["cat_dims"],
    )


def test_binary_deepgp_acquisition_forward_shapes(
    binary_deepgp_model_bundle: dict[str, Any],
) -> None:
    model = binary_deepgp_model_bundle["model"]
    train_x = binary_deepgp_model_bundle["train_x"]
    X = make_random_batch(
        binary_deepgp_model_bundle["bounds"],
        batch_size=4,
        q=2,
    )

    for acq_cls, kwargs, case_id in acquisition_cases(
        model=model,
        train_x=train_x,
    ):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


def test_binary_deepgp_mixed_acquisition_forward_shapes(
    binary_deepgp_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_deepgp_mixed_model_bundle["model"]
    train_x = binary_deepgp_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(
        binary_deepgp_mixed_model_bundle["bounds"],
        binary_deepgp_mixed_model_bundle["cat_dims"],
        batch_size=4,
        q=2,
    )

    for acq_cls, kwargs, case_id in acquisition_cases(
        model=model,
        train_x=train_x,
    ):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4]), case_id
        assert torch.isfinite(out).all(), case_id


@pytest.mark.slow
def test_binary_deepgp_optimize_acqf_representative_smoke(
    binary_deepgp_model_bundle: dict[str, Any],
) -> None:
    model = binary_deepgp_model_bundle["model"]
    train_x = binary_deepgp_model_bundle["train_x"]
    bounds = binary_deepgp_model_bundle["bounds"]
    q = 2

    for acq_cls, kwargs, case_id in _representative_acquisition_cases(
        model,
        train_x,
    ):
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
def test_binary_deepgp_mixed_optimize_acqf_mixed_representative_smoke(
    binary_deepgp_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_deepgp_mixed_model_bundle["model"]
    train_x = binary_deepgp_mixed_model_bundle["train_x"]
    bounds = binary_deepgp_mixed_model_bundle["bounds"]
    cat_id = binary_deepgp_mixed_model_bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=DTYPE, device=DEVICE)
    q = 2

    for acq_cls, kwargs, case_id in _representative_acquisition_cases(
        model,
        train_x,
    ):
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
def test_binary_deepgp_optimizer_constraint_case_smoke(
    binary_deepgp_model_bundle: dict[str, Any],
) -> None:
    model = binary_deepgp_model_bundle["model"]
    train_x = binary_deepgp_model_bundle["train_x"]
    bounds = binary_deepgp_model_bundle["bounds"]
    q = 2

    for (
        acq_cls,
        kwargs,
        _,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in _optimizer_constraint_scenarios(model, train_x, bounds):
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
def test_binary_deepgp_mixed_optimizer_constraint_case_smoke(
    binary_deepgp_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_deepgp_mixed_model_bundle["model"]
    train_x = binary_deepgp_mixed_model_bundle["train_x"]
    bounds = binary_deepgp_mixed_model_bundle["bounds"]
    cat_id = binary_deepgp_mixed_model_bundle["cat_dims"][0]
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
        train_x,
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
# Jupyter helpers: base test と同じ構成
# ============================================================


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
    bundle = create_binary_deepgp_model_bundle(
        cat=cat,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
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
    print(f"Jupyter DeepGP forward check cat={cat}")
    if verbose_forward_detail:
        print(f"train_x.shape={train_x.shape}")
        print(f"bounds.shape={bounds.shape}")
        print(f"X.shape={X.shape}")
    print("=" * 80)

    for acq_cls, kwargs, case_id in acquisition_cases(
        model=model,
        train_x=train_x,
    ):
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


def run_jupyter_optimize_acqf_all_acquisitions_check(
    *,
    n: int = 16,
    d: int = 5,
    num_epochs: int = 8,
    q: int = 2,
    num_restarts: int = 2,
    raw_samples: int = 16,
    maxiter: int = 10,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    bundle = create_binary_deepgp_model_bundle(
        cat=False,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    failed_cases: list[tuple[str, Exception]] = []

    print("=" * 100)
    print("Jupyter DeepGP optimize_acqf check: all acquisitions")
    print(f"n={n}, d={d}, q={q}, num_epochs={num_epochs}")
    print(f"num_acquisitions={len(acquisition_cases(model, train_x))}")
    print("=" * 100)

    for acq_cls, kwargs, case_id in acquisition_cases(
        model=model,
        train_x=train_x,
    ):
        display_id = f"optimize_acqf__{case_id}"
        try:
            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings,
            ):
                cands, acq_value = optimize_acqf(
                    acq_function=acq_cls(model=model, **kwargs),
                    bounds=bounds,
                    q=q,
                    sequential=True,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"maxiter": maxiter},
                )
            assert cands.shape == torch.Size([q, train_x.shape[-1]]), case_id
            assert torch.isfinite(cands).all(), case_id
            assert torch.isfinite(acq_value).all(), case_id
            assert_candidates_in_bounds(
                cands=cands,
                bounds=bounds,
            )
            print(f"[OK] {display_id}")
        except Exception as exc:
            print(f"[NG] {display_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((display_id, exc))
            if not continue_on_error:
                raise

    print("=" * 100)
    print(
        f"failed_cases={len(failed_cases)}"
        if failed_cases
        else "all optimize_acqf acquisition checks passed."
    )
    print("=" * 100)
    return bundle


def run_jupyter_optimize_acqf_mixed_all_acquisitions_check(
    *,
    n: int = 16,
    d: int = 5,
    num_epochs: int = 8,
    q: int = 2,
    num_restarts: int = 2,
    raw_samples: int = 16,
    maxiter: int = 10,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    bundle = create_binary_deepgp_model_bundle(
        cat=True,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_id = bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor(
        [5.0, 10.0, 15.0],
        dtype=train_x.dtype,
        device=train_x.device,
    )
    failed_cases: list[tuple[str, Exception]] = []

    print("=" * 100)
    print("Jupyter DeepGP optimize_acqf_mixed check: all acquisitions")
    print(
        f"n={n}, d={d}, q={q}, "
        f"num_epochs={num_epochs}, cat_dims={bundle['cat_dims']}"
    )
    print(f"num_acquisitions={len(acquisition_cases(model, train_x))}")
    print("=" * 100)

    for acq_cls, kwargs, case_id in acquisition_cases(
        model=model,
        train_x=train_x,
    ):
        display_id = f"optimize_acqf_mixed__{case_id}"
        try:
            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings,
            ):
                cands, acq_value = optimize_acqf_mixed(
                    acq_function=acq_cls(model=model, **kwargs),
                    bounds=bounds,
                    q=q,
                    fixed_features_list=fixed_features_list,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"maxiter": maxiter},
                )
            assert cands.shape == torch.Size([q, train_x.shape[-1]]), case_id
            assert torch.isfinite(cands).all(), case_id
            assert torch.isfinite(acq_value).all(), case_id
            assert_candidates_in_bounds(
                cands=cands,
                bounds=bounds,
            )
            assert torch.isin(cands[:, cat_id], cat_values).all(), case_id
            print(f"[OK] {display_id}")
        except Exception as exc:
            print(f"[NG] {display_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((display_id, exc))
            if not continue_on_error:
                raise

    print("=" * 100)
    print(
        f"failed_cases={len(failed_cases)}"
        if failed_cases
        else "all optimize_acqf_mixed acquisition checks passed."
    )
    print("=" * 100)
    return bundle


def run_jupyter_optimizer_constraint_compatibility_check(
    *,
    n: int = 16,
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

    bundle = create_binary_deepgp_model_bundle(
        cat=False,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    scenarios = _optimizer_constraint_scenarios(
        model,
        train_x,
        bounds,
        full_matrix=full_matrix,
    )
    failed_cases: list[tuple[str, Exception]] = []

    print("=" * 100)
    print("Jupyter DeepGP optimizer / constraint compatibility check")
    print(f"n={n}, d={d}, q={q}, num_epochs={num_epochs}, full_matrix={full_matrix}")
    print(f"num_cases={len(scenarios)}")
    print("=" * 100)

    for (
        acq_cls,
        kwargs,
        _,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in scenarios:
        try:
            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings,
            ):
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
            print(
                f"[OK] {case_id} cands.shape={tuple(cands.shape)} "
                f"acq_value={acq_value}"
                if verbose_ok_detail
                else f"[OK] {case_id}"
            )
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

    print("=" * 100)
    print(
        f"failed_cases={len(failed_cases)}"
        if failed_cases
        else "all optimizer / constraint compatibility checks passed."
    )
    print("=" * 100)
    return bundle


def run_jupyter_mixed_optimizer_constraint_compatibility_check(
    *,
    n: int = 16,
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

    bundle = create_binary_deepgp_model_bundle(
        cat=True,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )
    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_id = bundle["cat_dims"][0]
    fixed_features_list = [{cat_id: 5.0}, {cat_id: 10.0}, {cat_id: 15.0}]
    cat_values = torch.tensor(
        [5.0, 10.0, 15.0],
        dtype=train_x.dtype,
        device=train_x.device,
    )
    scenarios = _optimizer_constraint_scenarios(
        model,
        train_x,
        bounds,
        mixed=True,
        full_matrix=full_matrix,
    )
    failed_cases: list[tuple[str, Exception]] = []

    print("=" * 100)
    print("Jupyter DeepGP mixed optimizer / constraint compatibility check")
    print(
        f"n={n}, d={d}, q={q}, num_epochs={num_epochs}, "
        f"full_matrix={full_matrix}, cat_dims={bundle['cat_dims']}"
    )
    print(f"num_cases={len(scenarios)}")
    print("=" * 100)

    for (
        acq_cls,
        kwargs,
        _,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in scenarios:
        try:
            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings,
            ):
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
            print(
                f"[OK] {case_id} cands.shape={tuple(cands.shape)} "
                f"acq_value={acq_value}"
                if verbose_ok_detail
                else f"[OK] {case_id}"
            )
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

    print("=" * 100)
    print(
        f"failed_cases={len(failed_cases)}"
        if failed_cases
        else "all mixed optimizer / constraint compatibility checks passed."
    )
    print("=" * 100)
    return bundle


def run_jupyter_all_checks(
    *,
    num_epochs: int = 8,
    run_optimize: bool = True,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_forward_detail: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> None:
    run_jupyter_forward_check(
        cat=False,
        num_epochs=num_epochs,
        verbose_forward_detail=verbose_forward_detail,
    )
    run_jupyter_forward_check(
        cat=True,
        num_epochs=num_epochs,
        verbose_forward_detail=verbose_forward_detail,
    )

    if run_optimize:
        run_jupyter_optimize_acqf_all_acquisitions_check(
            num_epochs=num_epochs,
            continue_on_error=continue_on_error,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )
        run_jupyter_optimize_acqf_mixed_all_acquisitions_check(
            num_epochs=num_epochs,
            continue_on_error=continue_on_error,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )
        run_jupyter_optimizer_constraint_compatibility_check(
            num_epochs=num_epochs,
            full_matrix=full_matrix,
            continue_on_error=continue_on_error,
            verbose_ok_detail=verbose_ok_detail,
            verbose_candidates=verbose_candidates,
            verbose_constraints=verbose_constraints,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )
        run_jupyter_mixed_optimizer_constraint_compatibility_check(
            num_epochs=num_epochs,
            full_matrix=full_matrix,
            continue_on_error=continue_on_error,
            verbose_ok_detail=verbose_ok_detail,
            verbose_candidates=verbose_candidates,
            verbose_constraints=verbose_constraints,
            suppress_botorch_warnings=suppress_botorch_warnings,
        )

    print("all DeepGP Jupyter checks passed.")
