from __future__ import annotations

"""Binary classification PCA / REMBO single-output smoke tests.

This module follows ``test_binary_classification_base_single_output.py`` and
reuses its toy data, acquisition cases, optimizer wrappers, and compatibility
assertions.  PCA / REMBO specific checks verify that

- public wrapper inputs stay in raw-space,
- preproject inputs match the raw input transform,
- projected inputs are actually reduced to ``n_components``, and
- the internal base model is trained in projected-space.
"""

from typing import Any, Literal

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed

from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.high_dim import (
    PCABinaryClassificationGPModel,
    PCABinaryClassificationMixedGPModel,
    REMBOBinaryClassificationGPModel,
    REMBOBinaryClassificationMixedGPModel,
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

ProjectionKind = Literal["pca", "rembo"]


N_COMPONENTS = 2
RAW_CONT_DIM = 6


def _build_input_transform(
    train_x: torch.Tensor,
    bounds: torch.Tensor,
    cat_dims: list[int],
) -> Normalize:
    """PCA / REMBO projected model 共通の raw-space input_transform を作る。"""
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(
        d=train_x.shape[-1],
        bounds=bounds,
        indices=cont_indices,
    )


def _fit_projected_binary_model(
    model: Any,
    *,
    num_epochs: int,
    lr: float = 0.01,
) -> None:
    """Projected binary classification model の MLL を軽量設定で fit する。"""
    fit_binary_classifier_mll(
        model.make_mll(),
        num_epochs=num_epochs,
        lr=lr,
    )


def _get_projector(model: Any) -> Any:
    if hasattr(model, "pca"):
        return model.pca
    if hasattr(model, "rembo"):
        return model.rembo
    raise AssertionError("Projected model must expose either pca or rembo.")


def _assert_projected_model_training(
    model: Any,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int],
    n_components: int,
) -> None:
    """PCA / REMBO wrapper と internal base model の基本状態を確認する。"""
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

    with torch.no_grad():
        expected_preproject_x = (
            model.input_transform(train_x)
            if getattr(model, "input_transform", None) is not None
            else train_x
        )
        if isinstance(expected_preproject_x, tuple):
            expected_preproject_x = expected_preproject_x[0]

        projected_x = model.transform_inputs(train_x)
        posterior = model.posterior(train_x)
        latent_posterior = model.latent_posterior(train_x)

    assert model.input_dim_original == train_x.shape[-1]
    assert model.preproject_train_input.shape == train_x.shape
    assert torch.allclose(model.preproject_train_input, expected_preproject_x)

    projector = _get_projector(model)

    if cat_dims:
        expected_projected_dim = n_components + len(cat_dims)
        assert hasattr(model, "latent_dim")
        assert model.latent_dim == n_components
        assert model.projected_train_input.shape == torch.Size([train_x.shape[0], expected_projected_dim])
        assert projected_x.shape == torch.Size([train_x.shape[0], expected_projected_dim])
        assert model.projected_train_input.shape[-1] < train_x.shape[-1]
        assert projected_x.shape[-1] < train_x.shape[-1]

        assert list(model.cat_dims) == cat_dims
        assert list(model.cont_dims) == [i for i in range(train_x.shape[-1]) if i not in cat_dims]

        latent_cat_dims = list(range(n_components, n_components + len(cat_dims)))
        assert list(model.base_model.cat_dims) == latent_cat_dims

        cat_values = torch.tensor([5.0, 10.0, 15.0], dtype=train_x.dtype, device=train_x.device)
        for raw_cat_id, latent_cat_id in zip(cat_dims, latent_cat_dims):
            assert torch.isin(model.train_inputs[0][:, raw_cat_id], cat_values).all()
            assert torch.isin(model.preproject_train_input[:, raw_cat_id], cat_values).all()
            assert torch.isin(model.projected_train_input[:, latent_cat_id], cat_values).all()
            assert torch.isin(model.base_model.train_inputs[0][:, latent_cat_id], cat_values).all()
    else:
        expected_projected_dim = n_components
        assert model.projected_train_input.shape == torch.Size([train_x.shape[0], expected_projected_dim])
        assert projected_x.shape == torch.Size([train_x.shape[0], expected_projected_dim])
        assert model.projected_train_input.shape[-1] < train_x.shape[-1]
        assert projected_x.shape[-1] < train_x.shape[-1]

    assert getattr(projector.config, "n_components") == n_components
    assert model.base_model.train_inputs[0].shape == model.projected_train_input.shape
    assert torch.allclose(model.base_model.train_inputs[0], model.projected_train_input)
    assert torch.allclose(model.base_model.train_targets, train_y.reshape(-1))
    assert model.model.train_inputs[0].shape[-1] == expected_projected_dim
    assert model.make_mll().model is model.model

    assert posterior.mean.shape == train_y.shape
    assert posterior.variance.shape == train_y.shape
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert (posterior.mean >= 0.0).all() and (posterior.mean <= 1.0).all()
    assert torch.isfinite(latent_posterior.mean).all()


def create_binary_projected_model_bundle(
    *,
    kind: ProjectionKind,
    cat: bool = False,
    n: int = 18,
    d: int = RAW_CONT_DIM,
    n_components: int = N_COMPONENTS,
    num_epochs: int = 8,
) -> dict[str, Any]:
    """Jupyter / pytest 共通で使う PCA / REMBO binary model 作成関数。"""
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
    if kind == "pca":
        model_cls = PCABinaryClassificationMixedGPModel if cat else PCABinaryClassificationGPModel
        extra_kwargs: dict[str, Any] = {"n_components": n_components}
    elif kind == "rembo":
        model_cls = REMBOBinaryClassificationMixedGPModel if cat else REMBOBinaryClassificationGPModel
        extra_kwargs = {"n_components": n_components, "seed": 0}
    else:
        raise ValueError(f"Unknown projection kind: {kind}")

    model_kwargs: dict[str, Any] = {
        "train_X": train_x,
        "train_Y": train_y,
        "input_transform": input_transform,
        "num_inducing_points": 8,
        **extra_kwargs,
    }
    if cat:
        model_kwargs["cat_dims"] = cat_dims

    model = model_cls(**model_kwargs)

    _fit_projected_binary_model(
        model,
        num_epochs=num_epochs,
        lr=0.01,
    )
    _assert_projected_model_training(
        model=model,
        train_x=train_x,
        train_y=train_y,
        cat_dims=cat_dims,
        n_components=n_components,
    )

    return {
        "model": model,
        "train_x": train_x,
        "train_y": train_y,
        "bounds": bounds,
        "cat_dims": cat_dims,
        "kind": kind,
        "n_components": n_components,
    }


@pytest.fixture(scope="module", params=["pca", "rembo"])
def binary_projected_model_bundle(request: pytest.FixtureRequest) -> dict[str, Any]:
    """pytest 用: 通常 PCA / REMBO binary classification model。"""
    return create_binary_projected_model_bundle(
        kind=request.param,
        cat=False,
        n=18,
        d=RAW_CONT_DIM,
        n_components=N_COMPONENTS,
        num_epochs=8,
    )


@pytest.fixture(scope="module", params=["pca", "rembo"])
def binary_projected_mixed_model_bundle(request: pytest.FixtureRequest) -> dict[str, Any]:
    """pytest 用: mixed PCA / REMBO binary classification model。"""
    return create_binary_projected_model_bundle(
        kind=request.param,
        cat=True,
        n=18,
        d=RAW_CONT_DIM,
        n_components=N_COMPONENTS,
        num_epochs=8,
    )


def _representative_acquisition_cases(
    model: Any,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    """Projected model optimizer smoke test 用に代表 acquisition だけを返す。"""
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
    """Projected model optimizer constraint smoke test 用の軽量ケースだけを返す。"""
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
    prefix = "projected_mixed" if mixed else "projected"
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


def test_binary_projected_model_basic_behavior(
    binary_projected_model_bundle: dict[str, Any],
) -> None:
    _assert_projected_model_training(
        model=binary_projected_model_bundle["model"],
        train_x=binary_projected_model_bundle["train_x"],
        train_y=binary_projected_model_bundle["train_y"],
        cat_dims=binary_projected_model_bundle["cat_dims"],
        n_components=binary_projected_model_bundle["n_components"],
    )


def test_binary_projected_mixed_model_basic_behavior(
    binary_projected_mixed_model_bundle: dict[str, Any],
) -> None:
    _assert_projected_model_training(
        model=binary_projected_mixed_model_bundle["model"],
        train_x=binary_projected_mixed_model_bundle["train_x"],
        train_y=binary_projected_mixed_model_bundle["train_y"],
        cat_dims=binary_projected_mixed_model_bundle["cat_dims"],
        n_components=binary_projected_mixed_model_bundle["n_components"],
    )


def test_binary_projected_acquisition_forward_shapes(
    binary_projected_model_bundle: dict[str, Any],
) -> None:
    model = binary_projected_model_bundle["model"]
    train_x = binary_projected_model_bundle["train_x"]
    X = make_random_batch(
        binary_projected_model_bundle["bounds"],
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


def test_binary_projected_mixed_acquisition_forward_shapes(
    binary_projected_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_projected_mixed_model_bundle["model"]
    train_x = binary_projected_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(
        binary_projected_mixed_model_bundle["bounds"],
        binary_projected_mixed_model_bundle["cat_dims"],
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
def test_binary_projected_optimize_acqf_representative_smoke(
    binary_projected_model_bundle: dict[str, Any],
) -> None:
    model = binary_projected_model_bundle["model"]
    train_x = binary_projected_model_bundle["train_x"]
    bounds = binary_projected_model_bundle["bounds"]
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
def test_binary_projected_mixed_optimize_acqf_mixed_representative_smoke(
    binary_projected_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_projected_mixed_model_bundle["model"]
    train_x = binary_projected_mixed_model_bundle["train_x"]
    bounds = binary_projected_mixed_model_bundle["bounds"]
    cat_id = binary_projected_mixed_model_bundle["cat_dims"][0]
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
def test_binary_projected_optimizer_constraint_case_smoke(
    binary_projected_model_bundle: dict[str, Any],
) -> None:
    model = binary_projected_model_bundle["model"]
    train_x = binary_projected_model_bundle["train_x"]
    bounds = binary_projected_model_bundle["bounds"]
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
def test_binary_projected_mixed_optimizer_constraint_case_smoke(
    binary_projected_mixed_model_bundle: dict[str, Any],
) -> None:
    model = binary_projected_mixed_model_bundle["model"]
    train_x = binary_projected_mixed_model_bundle["train_x"]
    bounds = binary_projected_mixed_model_bundle["bounds"]
    cat_id = binary_projected_mixed_model_bundle["cat_dims"][0]
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
# Jupyter helpers
# ============================================================


def run_jupyter_forward_check(
    *,
    kind: ProjectionKind = "pca",
    cat: bool = False,
    n: int = 18,
    d: int = RAW_CONT_DIM,
    n_components: int = N_COMPONENTS,
    num_epochs: int = 8,
    batch_size: int = 4,
    q: int = 2,
    verbose_forward_detail: bool = False,
) -> dict[str, Any]:
    bundle = create_binary_projected_model_bundle(
        kind=kind,
        cat=cat,
        n=n,
        d=d,
        n_components=n_components,
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
    print(f"Jupyter {kind.upper()} forward check cat={cat}")
    if verbose_forward_detail:
        print(f"train_x.shape={train_x.shape}")
        print(f"preproject_train_input.shape={model.preproject_train_input.shape}")
        print(f"projected_train_input.shape={model.projected_train_input.shape}")
        print(f"base_model.train_inputs[0].shape={model.base_model.train_inputs[0].shape}")
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


def run_jupyter_all_forward_checks(
    *,
    num_epochs: int = 8,
    verbose_forward_detail: bool = False,
) -> None:
    for kind in ["pca", "rembo"]:
        run_jupyter_forward_check(
            kind=kind,
            cat=False,
            num_epochs=num_epochs,
            verbose_forward_detail=verbose_forward_detail,
        )
        run_jupyter_forward_check(
            kind=kind,
            cat=True,
            num_epochs=num_epochs,
            verbose_forward_detail=verbose_forward_detail,
        )
    print("all PCA / REMBO forward checks passed.")
