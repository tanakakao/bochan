from __future__ import annotations
import sys
sys.path.append("..")
sys.path.append("../../src")

from typing import Any
import warnings
from contextlib import contextmanager

import pytest
import torch

from gpytorch.mlls.variational_elbo import VariationalELBO

from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.exceptions.warnings import BadInitialCandidatesWarning

from bochan.models.classification.binary.base import (
    BinaryClassificationGPModel,
    BinaryClassificationMixedGPModel,
)
from bochan.fit import fit_binary_classifier_mll

from bochan.acquisition.binary.active_learning import (
    qBinaryPredictiveEntropy,
    qBinaryBALD,
    qBinaryProbabilityVariance,
    qBinaryMarginUncertainty,
)
from bochan.acquisition.binary.levelset_estimation import (
    qBinaryLatentStraddleAcquisition,
    qBinaryICUAcquisition,
    qBinaryBoundaryVarianceAcquisition,
    qBinaryClassEntropyAcquisition,
)
from bochan.acquisition.binary.bayesian_optimization import (
    qBinaryProbabilityOfFeasibility,
    qBinaryExpectedImprovement,
    qBinaryProbabilityOfImprovement,
    qBinaryUpperConfidenceBound,
    compute_binary_best_f,
)

from bochan.constraints import make_grid_k_sparse_post_processing_func

from bochan.optim import (
    optimize_acqf_evo,
    optimize_acqf_torch,
    optimize_acqf_evo_mixed,
    optimize_acqf_torch_mixed,
)


DTYPE = torch.double
DEVICE = torch.device("cpu")


@contextmanager
def maybe_suppress_botorch_initial_warnings(
    *,
    suppress: bool = True,
):
    """
    Jupyter 実行時に BadInitialCandidatesWarning の表示を抑制する context manager。

    qBinaryExpectedImprovement などでは acquisition value が 0 になりやすく、
    smoke test では warning を品質判定に使わないため、Jupyter の出力を
    見やすくしたい場合に使う。
    """
    with warnings.catch_warnings():
        if suppress:
            warnings.simplefilter("ignore", BadInitialCandidatesWarning)
        yield



# ============================================================
# Toy data
# ============================================================

def make_binary_toy_data(
    n: int = 20,
    d: int = 5,
    cat: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Binary classification 用の小さな toy data を作る。

    Args:
        n: サンプル数。
        d: 連続変数の次元数。
        cat: True の場合、最後の列にカテゴリ値 {5, 10, 15} を追加する。

    Returns:
        train_x: shape = (n, d) または (n, d + 1)。
        train_y: shape = (n, 1)。
        bounds: shape = (2, d) または (2, d + 1)。
    """
    torch.manual_seed(0)

    cont_x = torch.rand(n, d, device=DEVICE, dtype=DTYPE)

    if cat:
        cat_x = (
            torch.randint(1, 4, (n, 1), device=DEVICE)
            .to(dtype=DTYPE)
            * 5.0
        )
        train_x = torch.cat([cont_x, cat_x], dim=-1)
    else:
        train_x = cont_x

    base_weights = torch.tensor(
        [0.1, -0.2, 0.3, -0.4, 0.5],
        device=DEVICE,
        dtype=DTYPE,
    )

    if d <= len(base_weights):
        cont_weights = base_weights[:d]
    else:
        extra_weights = torch.linspace(
            0.1,
            0.5,
            d - len(base_weights),
            device=DEVICE,
            dtype=DTYPE,
        )
        cont_weights = torch.cat([base_weights, extra_weights], dim=0)

    if cat:
        weights = torch.cat(
            [
                cont_weights,
                torch.tensor([0.1], device=DEVICE, dtype=DTYPE),
            ],
            dim=0,
        )
    else:
        weights = cont_weights

    score = (train_x * weights).sum(dim=-1, keepdim=True)

    # ラベルが片側だけにならないように中央値で二値化する。
    threshold = score.median()
    train_y = (score > threshold).to(dtype=DTYPE)

    bounds = torch.zeros(2, train_x.shape[-1], device=DEVICE, dtype=DTYPE)
    bounds[1] = 1.0

    if cat:
        bounds[0, -1] = 5.0
        bounds[1, -1] = 15.0

    return train_x, train_y, bounds


# ============================================================
# Common assertions
# ============================================================

def assert_candidates_in_bounds(
    cands: torch.Tensor,
    bounds: torch.Tensor,
    *,
    atol: float = 1e-7,
) -> None:
    """候補点が bounds 内にあることを確認する。"""
    assert torch.all(cands >= bounds[0] - atol), (
        "候補点が下限を下回っています。"
        f"cand min={cands.min(dim=0).values}, lower={bounds[0]}"
    )

    assert torch.all(cands <= bounds[1] + atol), (
        "候補点が上限を上回っています。"
        f"cand max={cands.max(dim=0).values}, upper={bounds[1]}"
    )


def assert_model_training(
    model: Any,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    cat_dims: list[int] | None = None,
    check_raw_is_clone: bool = False,
) -> None:
    """
    モデル作成・fit 後の基本状態を確認する。

    Args:
        model: BoTorch-like binary classification model。
        train_x: 元の訓練入力。
        train_y: 元の訓練ラベル。
        cat_dims: mixed model のカテゴリ列。
        check_raw_is_clone:
            True の場合、train_inputs_raw[0] が train_x と別メモリであることも確認する。
            実装上、同一参照を許す場合は False のままにする。
    """
    model.eval()

    assert model.train_inputs[0].shape == train_x.shape, (
        f"model.train_inputs({model.train_inputs[0].shape}) と "
        f"train_x({train_x.shape}) の shape が一致しません。"
    )

    assert model.train_inputs_raw[0].shape == train_x.shape
    assert model.train_inputs_raw[0].dtype == train_x.dtype
    assert model.train_inputs_raw[0].device == train_x.device

    assert torch.allclose(model.train_inputs_raw[0], train_x), (
        "model.train_inputs_raw[0] と train_x が一致しません。"
    )

    if check_raw_is_clone:
        assert model.train_inputs_raw[0].data_ptr() != train_x.data_ptr(), (
            "model.train_inputs_raw[0] が train_x と同じメモリを参照しています。"
            "detach().clone() して保存する方が安全です。"
        )

    assert torch.allclose(model.train_targets, train_y.reshape(-1)), (
        "model.train_targets と train_y が一致しません。"
    )

    with torch.no_grad():
        expected_train_inputs = model.input_transform(train_x)

    assert torch.allclose(expected_train_inputs, model.train_inputs[0]), (
        "model.input_transform(train_x) と model.train_inputs[0] が一致しません。"
    )

    with torch.no_grad():
        posterior = model.posterior(train_x)
        mean = posterior.mean

    assert mean.shape == train_y.shape, (
        f"model.posterior(train_x).mean の shape が想定外です。"
        f"expected={train_y.shape}, actual={mean.shape}"
    )

    assert torch.isfinite(mean).all(), (
        "model.posterior(train_x).mean に NaN/inf が含まれます。"
    )

    assert (0.0 <= mean).all() and (mean <= 1.0).all(), (
        "model.posterior(train_x).mean が probability range 外です。"
        f"min={mean.min().item()}, max={mean.max().item()}"
    )

    if cat_dims:
        assert hasattr(model, "cat_dims"), (
            "mixed model のはずですが、model.cat_dims がありません。"
        )

        assert list(model.cat_dims) == cat_dims, (
            f"カテゴリ列が想定と異なります。"
            f"expected={cat_dims}, actual={model.cat_dims}"
        )

        cat_values = torch.tensor(
            [5.0, 10.0, 15.0],
            dtype=train_x.dtype,
            device=train_x.device,
        )

        for cat_id in cat_dims:
            assert torch.isin(model.train_inputs[0][:, cat_id], cat_values).all(), (
                "model.train_inputs[0] のカテゴリ値が変換されている、"
                f"または想定値以外です。cat_id={cat_id}, "
                f"values={model.train_inputs[0][:, cat_id].unique()}"
            )

            assert torch.isin(model.train_inputs_raw[0][:, cat_id], cat_values).all(), (
                "model.train_inputs_raw[0] のカテゴリ値が変換されている、"
                f"または想定値以外です。cat_id={cat_id}, "
                f"values={model.train_inputs_raw[0][:, cat_id].unique()}"
            )


# ============================================================
# Model creation
# Jupyter / pytest 共通で使う本体
# ============================================================

def create_binary_model_bundle(
    *,
    cat: bool = False,
    n: int = 20,
    d: int = 5,
    num_epochs: int = 60,
    check_raw_is_clone: bool = False,
) -> dict[str, Any]:
    """
    Jupyter / pytest の両方から使える binary classification model 作成関数。

    Args:
        cat: True の場合、BinaryClassificationMixedGPModel を作成する。
        n: サンプル数。
        d: 連続変数の次元数。
        num_epochs: fit_binary_classifier_mll の epoch 数。
        check_raw_is_clone:
            train_inputs_raw が clone されていることまで確認する場合は True。

    Returns:
        model, train_x, train_y, bounds, cat_dims を含む dict。
    """
    train_x, train_y, bounds = make_binary_toy_data(
        n=n,
        d=d,
        cat=cat,
    )

    cat_dims = [d] if cat else []
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]

    input_transform = Normalize(
        d=train_x.shape[-1],
        bounds=bounds,
        indices=cont_indices,
    )

    if cat:
        model = BinaryClassificationMixedGPModel(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
            cat_dims=cat_dims,
        )
    else:
        model = BinaryClassificationGPModel(
            train_X=train_x,
            train_Y=train_y,
            input_transform=input_transform,
        )

    mll = VariationalELBO(
        likelihood=model.likelihood,
        model=model.model,
        num_data=model.model.train_inputs[0].shape[-2],
    )

    fit_binary_classifier_mll(
        mll,
        num_epochs=num_epochs,
    )

    assert_model_training(
        model=model,
        train_x=train_x,
        train_y=train_y,
        cat_dims=cat_dims,
        check_raw_is_clone=check_raw_is_clone,
    )

    return {
        "model": model,
        "train_x": train_x,
        "train_y": train_y,
        "bounds": bounds,
        "cat_dims": cat_dims,
    }


# ============================================================
# pytest fixtures
# ============================================================

@pytest.fixture(scope="module")
def binary_model_bundle() -> dict[str, Any]:
    """pytest 用: 通常 binary classification model."""
    return create_binary_model_bundle(
        cat=False,
        n=20,
        d=5,
        num_epochs=60,
    )


@pytest.fixture(scope="module")
def binary_mixed_model_bundle() -> dict[str, Any]:
    """pytest 用: mixed binary classification model."""
    return create_binary_model_bundle(
        cat=True,
        n=20,
        d=5,
        num_epochs=60,
    )


# ============================================================
# Acquisition helpers
# ============================================================

def make_safe_best_f(
    model: Any,
    train_x: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    qBinaryExpectedImprovement / qBinaryProbabilityOfImprovement 用の安全側 best_f。

    テストでは EI が全点 0 になると optimizer smoke test が不安定になるため、
    compute_binary_best_f の値から少し下げる。
    """
    best_f = compute_binary_best_f(
        model,
        train_x,
        apply_sigmoid_if_needed=True,
    )

    return (best_f - margin).clamp(min=1e-6, max=1.0 - 1e-6).detach()


def make_random_batch(
    bounds: torch.Tensor,
    *,
    batch_size: int = 8,
    q: int = 3,
) -> torch.Tensor:
    """acquisition forward 用の batch_shape x q x d 候補点を作る。"""
    d = bounds.shape[-1]

    X = torch.rand(
        batch_size,
        q,
        d,
        device=bounds.device,
        dtype=bounds.dtype,
    )

    return bounds[0] + (bounds[1] - bounds[0]) * X


def make_random_mixed_batch(
    bounds: torch.Tensor,
    cat_dims: list[int],
    *,
    batch_size: int = 8,
    q: int = 3,
) -> torch.Tensor:
    """
    mixed model 用の batch_shape x q x d 候補点を作る。

    カテゴリ列は {5, 10, 15} のいずれかにする。
    """
    X = make_random_batch(
        bounds=bounds,
        batch_size=batch_size,
        q=q,
    )

    cat_values = torch.tensor(
        [5.0, 10.0, 15.0],
        dtype=X.dtype,
        device=X.device,
    )

    for cat_id in cat_dims:
        random_idx = torch.randint(
            0,
            len(cat_values),
            X.shape[:-1],
            device=X.device,
        )
        X[..., cat_id] = cat_values[random_idx]

    return X


def acquisition_cases(
    model: Any,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    """acquisition class, kwargs, case_id の一覧を返す。"""
    safe_best_f = make_safe_best_f(
        model=model,
        train_x=train_x,
        margin=0.05,
    )

    return [
        (qBinaryPredictiveEntropy, {}, "predictive_entropy"),
        (qBinaryBALD, {}, "bald"),
        (qBinaryProbabilityVariance, {}, "probability_variance"),
        (qBinaryMarginUncertainty, {}, "margin_uncertainty"),
        (qBinaryLatentStraddleAcquisition, {}, "latent_straddle"),
        (qBinaryICUAcquisition, {}, "icu"),
        (qBinaryBoundaryVarianceAcquisition, {}, "boundary_variance"),
        (qBinaryClassEntropyAcquisition, {}, "class_entropy"),
        (qBinaryProbabilityOfFeasibility, {}, "pof"),
        (
            qBinaryExpectedImprovement,
            {
                "best_f": safe_best_f,
                "apply_sigmoid_if_needed": True,
            },
            "binary_ei",
        ),
        (
            qBinaryProbabilityOfImprovement,
            {
                "best_f": safe_best_f,
                "apply_sigmoid_if_needed": True,
                "tau": 1e-2,
            },
            "binary_pi",
        ),
        (qBinaryUpperConfidenceBound, {}, "binary_ucb"),
    ]


def representative_acquisition_cases(
    model: Any,
    train_x: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str]]:
    """
    optimizer compatibility test 用の代表 acquisition 一覧。

    全 acquisition × 全 optimizer × 全制約条件を毎回回すと重いため、
    性質の異なる代表だけに絞る。
    """
    safe_best_f = make_safe_best_f(
        model=model,
        train_x=train_x,
        margin=0.05,
    )

    return [
        (
            qBinaryPredictiveEntropy,
            {
                "reduction": "sum",
                "pending_penalty_weight": 0.1,
            },
            "predictive_entropy",
        ),
        (
            qBinaryBALD,
            {
                "reduction": "sum",
                "pending_penalty_weight": 0.1,
            },
            "bald",
        ),
        (
            qBinaryLatentStraddleAcquisition,
            {
                "reduction": "sum",
                "pending_penalty_weight": 0.1,
            },
            "latent_straddle",
        ),
        (
            qBinaryProbabilityOfFeasibility,
            {},
            "pof",
        ),
        (
            qBinaryExpectedImprovement,
            {
                "best_f": safe_best_f,
                "apply_sigmoid_if_needed": True,
            },
            "binary_ei",
        ),
        (
            qBinaryProbabilityOfImprovement,
            {
                "best_f": safe_best_f,
                "apply_sigmoid_if_needed": True,
                "tau": 1e-2,
            },
            "binary_pi",
        ),
        (
            qBinaryUpperConfidenceBound,
            {},
            "binary_ucb",
        ),
    ]


# ============================================================
# Constraint / optimizer compatibility helpers
# ============================================================

def optimizer_cases() -> list[tuple[str, str | None, str]]:
    """
    optimize_func / optimize_method の代表ケース。

    Returns:
        optimize_func, optimize_method, case_id
    """
    return [
        ("base", None, "base"),
        ("torch", "adam", "torch_adam"),
        ("evo", "cmaes", "evo_cmaes"),
        ("evo", "pso", "evo_pso"),
        ("evo", "ga", "evo_ga"),
    ]


def make_constraint_cases(
    bounds: torch.Tensor,
) -> list[dict[str, Any]]:
    """
    制約 / step / k-sparse の代表ケースを作る。

    前提:
        bounds.shape[-1] >= 5

    ケース:
        - none: 制約なし
        - step_only: step 丸めのみ
        - constraints_only: 線形制約のみ
        - step_k_sparse_constraints: step + k-sparse + 線形制約
    """
    if bounds.shape[-1] < 5:
        raise ValueError("constraint cases require d >= 5.")

    device = bounds.device
    dtype = bounds.dtype

    numeric_indices = [0, 1, 2, 3]
    steps = torch.tensor(
        [0.01, 0.02, 0.01, 0.05],
        device=device,
        dtype=dtype,
    )

    equality_constraints = [
        (
            torch.tensor([0, 1, 2], device=device),
            torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype),
            1.0,
        )
    ]

    # 本テストでは make_grid_k_sparse_post_processing_func の既定に合わせて
    # inequality_sense="le" として扱う。
    # つまり以下は x3 + x4 <= 0.2 の意味。
    # 符号処理は make_grid_k_sparse_post_processing_func 側に任せるため、
    # optimize_with_case では inequality_constraints を再変換しない。
    inequality_constraints = [
        (
            torch.tensor([3, 4], device=device),
            torch.tensor([1.0, 1.0], device=device, dtype=dtype),
            0.2,
        )
    ]

    return [
        {
            "case_id": "none",
            "numeric_indices": [],
            "steps": None,
            "comp_idx": [],
            "k": None,
            "equality_constraints": [],
            "inequality_constraints": [],
            "inequality_sense": "le",
            "use_repair": False,
            "expect_step_grid": False,
            "expect_k_sparse": False,
            "expect_constraints": False,
        },
        {
            "case_id": "step_only",
            "numeric_indices": numeric_indices,
            "steps": steps,
            "comp_idx": [],
            "k": None,
            "equality_constraints": [],
            "inequality_constraints": [],
            "inequality_sense": "le",
            "use_repair": True,
            "expect_step_grid": True,
            "expect_k_sparse": False,
            "expect_constraints": False,
        },
        {
            "case_id": "constraints_only",
            "numeric_indices": [],
            "steps": None,
            "comp_idx": [],
            "k": None,
            "equality_constraints": equality_constraints,
            "inequality_constraints": inequality_constraints,
            "inequality_sense": "le",
            "use_repair": False,
            "expect_step_grid": False,
            "expect_k_sparse": False,
            "expect_constraints": True,
        },
        {
            "case_id": "step_k_sparse_constraints",
            "numeric_indices": numeric_indices,
            "steps": steps,
            "comp_idx": [0, 1, 2],
            "k": 2,
            "equality_constraints": equality_constraints,
            "inequality_constraints": inequality_constraints,
            "inequality_sense": "le",
            "use_repair": True,
            "expect_step_grid": True,
            "expect_k_sparse": True,
            "expect_constraints": True,
        },
    ]


def build_repair_from_constraint_case(
    bounds: torch.Tensor,
    constraint_case: dict[str, Any],
):
    """constraint_case から post_processing_func を作る。"""
    if not constraint_case["use_repair"]:
        return None

    return make_grid_k_sparse_post_processing_func(
        bounds=bounds,
        numeric_indices=constraint_case["numeric_indices"],
        steps=constraint_case["steps"],
        comp_idx=constraint_case["comp_idx"],
        k=constraint_case["k"],
        equality_constraints=constraint_case["equality_constraints"],
        inequality_constraints=constraint_case["inequality_constraints"],
        inequality_sense=constraint_case.get("inequality_sense", "le"),
    )


def optimize_with_case(
    *,
    acqf: Any,
    bounds: torch.Tensor,
    q: int,
    optimize_func: str,
    optimize_method: str | None,
    constraint_case: dict[str, Any],
    num_restarts: int = 5,
    raw_samples: int = 64,
    maxiter: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    optimize_func / optimize_method / constraint_case に応じて optimizer を呼ぶ。
    """
    repair = build_repair_from_constraint_case(
        bounds=bounds,
        constraint_case=constraint_case,
    )

    equality_constraints = constraint_case["equality_constraints"]
    inequality_constraints = constraint_case["inequality_constraints"]

    if optimize_func == "base":
        return optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            sequential=True,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"maxiter": maxiter},
            post_processing_func=repair,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )

    if optimize_func == "evo":
        if optimize_method is None:
            raise ValueError("optimize_func='evo' の場合 optimize_method が必要です。")

        return optimize_acqf_evo(
            acq_function=acqf,
            method=optimize_method,
            bounds=bounds,
            q=q,
            sequential=True,
            post_processing_func=repair,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )

    if optimize_func == "torch":
        method = optimize_method or "adam"

        return optimize_acqf_torch(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            method=method,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
            options={
                "lr": 0.03,
                "num_steps": 100,
                "penalty_factor": 1e3,
            },
            post_processing_func=repair,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )

    raise ValueError(f"Unknown optimize_func: {optimize_func}")



def optimize_mixed_with_case(
    *,
    acqf: Any,
    bounds: torch.Tensor,
    q: int,
    fixed_features_list: list[dict[int, float]],
    optimize_func: str,
    optimize_method: str | None,
    constraint_case: dict[str, Any],
    num_restarts: int = 5,
    raw_samples: int = 64,
    maxiter: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    mixed model 用に optimize_func / optimize_method / constraint_case に応じて optimizer を呼ぶ。

    方針:
        - base は optimize_acqf_mixed を使う。
        - evo は optimize_acqf_evo_mixed を使う。
        - torch は optimize_acqf_torch_mixed を使う。
        - inequality_constraints の符号変換はしない。
          make_grid_k_sparse_post_processing_func 側の inequality_sense に任せる。
    """
    repair = build_repair_from_constraint_case(
        bounds=bounds,
        constraint_case=constraint_case,
    )

    equality_constraints = constraint_case["equality_constraints"]
    inequality_constraints = constraint_case["inequality_constraints"]

    if optimize_func == "base":
        return optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            fixed_features_list=fixed_features_list,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"maxiter": maxiter},
            post_processing_func=repair,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )

    if optimize_func == "evo":
        if optimize_method is None:
            raise ValueError("optimize_func='evo' の場合 optimize_method が必要です。")

        return optimize_acqf_evo_mixed(
            acq_function=acqf,
            method=optimize_method,
            bounds=bounds,
            q=q,
            fixed_features_list=fixed_features_list,
            sequential=True,
            post_processing_func=repair,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )

    if optimize_func == "torch":
        method = optimize_method or "adam"

        return optimize_acqf_torch_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            fixed_features_list=fixed_features_list,
            method=method,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
            options={
                "lr": 0.03,
                "num_steps": 100,
                "penalty_factor": 1e3,
            },
            post_processing_func=repair,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
        )

    raise ValueError(f"Unknown optimize_func: {optimize_func}")


def assert_step_grid(
    cands: torch.Tensor,
    bounds: torch.Tensor,
    numeric_indices: list[int],
    steps: torch.Tensor,
    *,
    atol: float = 1e-6,
) -> None:
    """指定された numeric_indices が step grid 上にあることを確認する。"""
    if not numeric_indices:
        return

    idx = torch.tensor(
        numeric_indices,
        device=cands.device,
        dtype=torch.long,
    )

    steps = steps.to(device=cands.device, dtype=cands.dtype)
    base = bounds[0, idx].to(device=cands.device, dtype=cands.dtype)

    z = (cands[:, idx] - base) / steps
    z_round = z.round()

    assert torch.allclose(z, z_round, atol=atol, rtol=0.0), (
        "候補点が step grid 上にありません。"
        f"indices={numeric_indices}, steps={steps}, z={z}"
    )


def assert_k_sparse(
    cands: torch.Tensor,
    comp_idx: list[int],
    k: int,
    *,
    threshold: float = 1e-8,
) -> None:
    """comp_idx 内の非ゼロ数が k 以下であることを確認する。"""
    if not comp_idx:
        return

    idx = torch.tensor(
        comp_idx,
        device=cands.device,
        dtype=torch.long,
    )

    nnz = (cands[:, idx].abs() > threshold).sum(dim=-1)

    assert (nnz <= k).all(), (
        "k-sparse 制約を満たしていません。"
        f"comp_idx={comp_idx}, k={k}, nnz={nnz}, values={cands[:, idx]}"
    )


def _tensor_to_debug_string(
    x: torch.Tensor,
    *,
    max_rows: int = 12,
) -> str:
    """
    assert 失敗時に tensor の値を読みやすく表示するための helper。

    Args:
        x: 表示対象 tensor。
        max_rows: 先頭から表示する最大行数。

    Returns:
        str: CPU 上に移した tensor の文字列表現。
    """
    x_cpu = x.detach().cpu()

    if x_cpu.ndim >= 1 and x_cpu.shape[0] > max_rows:
        head = x_cpu[:max_rows]
        return (
            f"{head}\n"
            f"... truncated: showing first {max_rows} rows of {x_cpu.shape[0]} rows"
        )

    return str(x_cpu)


def _format_constraint_debug_message(
    *,
    constraint_type: str,
    cands: torch.Tensor,
    indices: torch.Tensor,
    coefficients: torch.Tensor,
    rhs: float,
    lhs: torch.Tensor,
    tolerance: float,
    violation_mask: torch.Tensor,
    inequality_sense: str | None = None,
) -> str:
    """
    線形制約違反時の詳細メッセージを作る。

    表示するもの:
        - 制約の種類
        - 対象 indices
        - coefficients
        - rhs
        - lhs
        - lhs - rhs
        - violation mask
        - 違反している候補点
        - 違反している候補点の対象列だけ
    """
    rhs_t = torch.as_tensor(rhs, device=lhs.device, dtype=lhs.dtype)
    diff = lhs - rhs_t

    violation_rows = violation_mask.nonzero(as_tuple=False).reshape(-1)
    violated_cands = cands[violation_rows] if violation_rows.numel() > 0 else cands[:0]
    violated_values = (
        cands[violation_rows][:, indices]
        if violation_rows.numel() > 0
        else cands[:0, indices]
    )

    sense_line = (
        f"  inequality_sense={inequality_sense}\n"
        if inequality_sense is not None
        else ""
    )

    return (
        f"{constraint_type}制約を満たしていません。\n"
        f"{sense_line}"
        f"  indices={indices.detach().cpu().tolist()}\n"
        f"  coefficients={coefficients.detach().cpu().tolist()}\n"
        f"  rhs={rhs}\n"
        f"  tolerance={tolerance}\n"
        f"  lhs={_tensor_to_debug_string(lhs)}\n"
        f"  lhs_minus_rhs={_tensor_to_debug_string(diff)}\n"
        f"  violation_mask={_tensor_to_debug_string(violation_mask)}\n"
        f"  violation_rows={violation_rows.detach().cpu().tolist()}\n"
        f"  violated_cands=\n{_tensor_to_debug_string(violated_cands)}\n"
        f"  violated_values_at_indices=\n{_tensor_to_debug_string(violated_values)}\n"
        f"  all_cands=\n{_tensor_to_debug_string(cands)}"
    )


def assert_linear_constraints(
    cands: torch.Tensor,
    equality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]],
    inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]],
    *,
    inequality_sense: str = "le",
    equality_atol: float = 5e-2,
    inequality_atol: float = 5e-2,
) -> None:
    """
    線形制約を確認する。

    注意:
        optimizer や repair の実装によっては数値誤差が出るため、
        smoke test では tolerance を少し緩めにしている。

    失敗時:
        lhs, rhs, lhs-rhs, violation rows, violated candidates を
        assert message に含める。
    """
    if inequality_sense not in {"le", "ge"}:
        raise ValueError(
            f"inequality_sense must be 'le' or 'ge', got {inequality_sense!r}."
        )

    for indices, coefficients, rhs in equality_constraints:
        indices = indices.to(device=cands.device, dtype=torch.long)
        coefficients = coefficients.to(device=cands.device, dtype=cands.dtype)
        rhs_t = torch.as_tensor(rhs, device=cands.device, dtype=cands.dtype)

        lhs = (cands[:, indices] * coefficients).sum(dim=-1)
        diff_abs = (lhs - rhs_t).abs()
        violation_mask = diff_abs > equality_atol

        assert not violation_mask.any(), _format_constraint_debug_message(
            constraint_type="等式",
            cands=cands,
            indices=indices,
            coefficients=coefficients,
            rhs=rhs,
            lhs=lhs,
            tolerance=equality_atol,
            violation_mask=violation_mask,
        )

    for indices, coefficients, rhs in inequality_constraints:
        indices = indices.to(device=cands.device, dtype=torch.long)
        coefficients = coefficients.to(device=cands.device, dtype=cands.dtype)
        rhs_t = torch.as_tensor(rhs, device=cands.device, dtype=cands.dtype)

        lhs = (cands[:, indices] * coefficients).sum(dim=-1)

        if inequality_sense == "le":
            # sum(coeff * x) <= rhs
            violation_mask = lhs > rhs_t + inequality_atol
        else:
            # sum(coeff * x) >= rhs
            violation_mask = lhs < rhs_t - inequality_atol

        assert not violation_mask.any(), _format_constraint_debug_message(
            constraint_type="不等式",
            cands=cands,
            indices=indices,
            coefficients=coefficients,
            rhs=rhs,
            lhs=lhs,
            tolerance=inequality_atol,
            violation_mask=violation_mask,
            inequality_sense=inequality_sense,
        )


def compute_linear_constraint_diagnostics(
    cands: torch.Tensor,
    equality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]],
    inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]],
    *,
    inequality_sense: str = "le",
    equality_atol: float = 5e-2,
    inequality_atol: float = 5e-2,
) -> list[dict[str, Any]]:
    """
    Jupyter などで線形制約の lhs / rhs / violation を確認するための診断関数。

    assert_linear_constraints と同じ tolerance を使う。
    そのため、丸め誤差レベルの lhs-rhs は violation として表示しない。

    Returns:
        各制約について以下を含む dict の list。
            - type: "equality" or "inequality"
            - indices
            - coefficients
            - rhs
            - lhs
            - lhs_minus_rhs
            - tolerance
            - violation_mask
            - violated_cands
            - violated_values_at_indices
    """
    if inequality_sense not in {"le", "ge"}:
        raise ValueError(
            f"inequality_sense must be 'le' or 'ge', got {inequality_sense!r}."
        )

    diagnostics: list[dict[str, Any]] = []

    for indices, coefficients, rhs in equality_constraints:
        indices = indices.to(device=cands.device, dtype=torch.long)
        coefficients = coefficients.to(device=cands.device, dtype=cands.dtype)
        rhs_t = torch.as_tensor(rhs, device=cands.device, dtype=cands.dtype)

        lhs = (cands[:, indices] * coefficients).sum(dim=-1)
        diff = lhs - rhs_t
        violation_mask = diff.abs() > equality_atol
        rows = violation_mask.nonzero(as_tuple=False).reshape(-1)

        diagnostics.append(
            {
                "type": "equality",
                "indices": indices.detach().cpu(),
                "coefficients": coefficients.detach().cpu(),
                "rhs": float(rhs_t.detach().cpu()),
                "lhs": lhs.detach().cpu(),
                "lhs_minus_rhs": diff.detach().cpu(),
                "tolerance": equality_atol,
                "violation_mask": violation_mask.detach().cpu(),
                "violated_cands": cands[rows].detach().cpu(),
                "violated_values_at_indices": cands[rows][:, indices].detach().cpu()
                if rows.numel() > 0
                else cands[:0, indices].detach().cpu(),
            }
        )

    for indices, coefficients, rhs in inequality_constraints:
        indices = indices.to(device=cands.device, dtype=torch.long)
        coefficients = coefficients.to(device=cands.device, dtype=cands.dtype)
        rhs_t = torch.as_tensor(rhs, device=cands.device, dtype=cands.dtype)

        lhs = (cands[:, indices] * coefficients).sum(dim=-1)
        diff = lhs - rhs_t

        if inequality_sense == "le":
            # sum(coeff * x) <= rhs
            violation_mask = lhs > rhs_t + inequality_atol
        else:
            # sum(coeff * x) >= rhs
            violation_mask = lhs < rhs_t - inequality_atol

        rows = violation_mask.nonzero(as_tuple=False).reshape(-1)

        diagnostics.append(
            {
                "type": "inequality",
                "inequality_sense": inequality_sense,
                "indices": indices.detach().cpu(),
                "coefficients": coefficients.detach().cpu(),
                "rhs": float(rhs_t.detach().cpu()),
                "lhs": lhs.detach().cpu(),
                "lhs_minus_rhs": diff.detach().cpu(),
                "tolerance": inequality_atol,
                "violation_mask": violation_mask.detach().cpu(),
                "violated_cands": cands[rows].detach().cpu(),
                "violated_values_at_indices": cands[rows][:, indices].detach().cpu()
                if rows.numel() > 0
                else cands[:0, indices].detach().cpu(),
            }
        )

    return diagnostics


def print_linear_constraint_diagnostics(
    cands: torch.Tensor,
    equality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]],
    inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]],
    *,
    inequality_sense: str = "le",
    equality_atol: float = 5e-2,
    inequality_atol: float = 5e-2,
    show_all: bool = True,
) -> None:
    """
    線形制約の lhs / rhs / violation を print する Jupyter 向け helper。

    Args:
        equality_atol: 等式制約の許容誤差。
        inequality_atol: 不等式制約の許容誤差。
        show_all:
            True の場合、OK の制約も表示する。
            False の場合、違反がある制約だけ表示する。
    """
    diagnostics = compute_linear_constraint_diagnostics(
        cands=cands,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,
        equality_atol=equality_atol,
        inequality_atol=inequality_atol,
    )

    for i, diag in enumerate(diagnostics):
        has_violation = bool(diag["violation_mask"].any())

        if not show_all and not has_violation:
            continue

        status = "NG" if has_violation else "OK"

        print("-" * 80)
        print(f"constraint[{i}] status={status} type={diag['type']}")
        if diag["type"] == "inequality":
            print(f"inequality_sense={diag.get('inequality_sense', inequality_sense)}")
        print(f"indices={diag['indices'].tolist()}")
        print(f"coefficients={diag['coefficients'].tolist()}")
        print(f"rhs={diag['rhs']}")
        print(f"tolerance={diag['tolerance']}")
        print(f"lhs={diag['lhs']}")
        print(f"lhs-rhs={diag['lhs_minus_rhs']}")
        print(f"violation_mask={diag['violation_mask']}")
        print(f"violated_cands=\n{diag['violated_cands']}")
        print(f"violated_values_at_indices=\n{diag['violated_values_at_indices']}")



def assert_optimizer_compatibility_result(
    *,
    cands: torch.Tensor,
    acq_value: torch.Tensor,
    bounds: torch.Tensor,
    q: int,
    d: int,
    constraint_case: dict[str, Any],
    case_id: str,
) -> None:
    """optimizer compatibility test の共通 assert。"""
    assert cands.shape == torch.Size([q, d]), (
        f"{case_id}: 候補点 shape が想定外です。"
        f"expected={torch.Size([q, d])}, actual={cands.shape}"
    )

    assert torch.isfinite(cands).all(), (
        f"{case_id}: 候補点に NaN/inf が含まれます。"
        f"cands={cands}"
    )

    assert torch.isfinite(acq_value).all(), (
        f"{case_id}: acq_value に NaN/inf が含まれます。"
        f"acq_value={acq_value}"
    )

    assert_candidates_in_bounds(
        cands=cands,
        bounds=bounds,
    )

    if constraint_case["expect_step_grid"]:
        assert_step_grid(
            cands=cands,
            bounds=bounds,
            numeric_indices=constraint_case["numeric_indices"],
            steps=constraint_case["steps"],
        )

    if constraint_case["expect_k_sparse"]:
        assert_k_sparse(
            cands=cands,
            comp_idx=constraint_case["comp_idx"],
            k=constraint_case["k"],
        )

    if constraint_case["expect_constraints"]:
        assert_linear_constraints(
            cands=cands,
            equality_constraints=constraint_case["equality_constraints"],
            inequality_constraints=constraint_case["inequality_constraints"],
            inequality_sense=constraint_case.get("inequality_sense", "le"),
        )


def optimizer_constraint_smoke_scenarios(
    model: Any,
    train_x: torch.Tensor,
    bounds: torch.Tensor,
) -> list[tuple[type, dict[str, Any], str, str, str | None, dict[str, Any], str]]:
    """
    pytest 用の軽めの代表シナリオを返す。

    全組み合わせではなく、以下を一通りカバーする:
        - optimize_func: base / torch / evo
        - optimize_method: cmaes / pso / ga
        - constraint: none / step_only / constraints_only / step_k_sparse_constraints
        - acquisition: entropy / BALD / straddle / EI / UCB

    追加:
        - base + step_k_sparse_constraints も明示的に確認する。
    """
    acq_map = {
        case_id: (acqf_cls, acqf_kwargs)
        for acqf_cls, acqf_kwargs, case_id in representative_acquisition_cases(
            model,
            train_x,
        )
    }

    constraint_map = {
        case["case_id"]: case
        for case in make_constraint_cases(bounds)
    }

    raw_scenarios = [
        # ------------------------------------------------------------
        # 1. optimizer method coverage
        #    base__none は標準 optimize_acqf 全 acquisition で既に確認するため、
        #    ここでは evo / torch の互換性確認に集中する。
        # ------------------------------------------------------------
        ("predictive_entropy", "evo", "cmaes", "none"),
        ("predictive_entropy", "evo", "pso", "none"),
        ("predictive_entropy", "evo", "ga", "none"),
        ("bald", "torch", "adam", "none"),

        # ------------------------------------------------------------
        # 2. constraint / repair coverage
        #    制約処理そのものの確認には、比較的安定な acquisition を使う。
        # ------------------------------------------------------------
        ("latent_straddle", "base", None, "step_only"),
        ("bald", "torch", "adam", "constraints_only"),
        ("binary_ucb", "base", None, "step_k_sparse_constraints"),
        ("pof", "evo", "cmaes", "step_k_sparse_constraints"),

        # ------------------------------------------------------------
        # 3. Bayesian optimization acquisition coverage
        #    EI / PI は best_f 依存でゼロ張り付きや warning が起きやすいため、
        #    重い制約ケースではなく、optimizer 互換性の確認に留める。
        # ------------------------------------------------------------
        ("pof", "base", None, "constraints_only"),
        ("binary_pi", "torch", "adam", "none"),
        ("binary_ei", "evo", "cmaes", "none"),
        ("binary_ucb", "torch", "adam", "step_k_sparse_constraints"),
    ]

    scenarios = []

    for acq_id, optimize_func, optimize_method, constraint_id in raw_scenarios:
        acqf_cls, acqf_kwargs = acq_map[acq_id]
        constraint_case = constraint_map[constraint_id]

        case_id = (
            f"{acq_id}__{optimize_func}"
            f"{'' if optimize_method is None else '_' + optimize_method}"
            f"__{constraint_id}"
        )

        scenarios.append(
            (
                acqf_cls,
                acqf_kwargs,
                acq_id,
                optimize_func,
                optimize_method,
                constraint_case,
                case_id,
            )
        )

    return scenarios


# ============================================================
# pytest: Basic model tests
# ============================================================

def test_binary_model_basic_behavior(
    binary_model_bundle: dict[str, Any],
) -> None:
    """通常 binary model の基本状態を確認する。"""
    model = binary_model_bundle["model"]
    train_x = binary_model_bundle["train_x"]
    train_y = binary_model_bundle["train_y"]
    cat_dims = binary_model_bundle["cat_dims"]

    assert_model_training(
        model=model,
        train_x=train_x,
        train_y=train_y,
        cat_dims=cat_dims,
    )


def test_binary_mixed_model_basic_behavior(
    binary_mixed_model_bundle: dict[str, Any],
) -> None:
    """mixed binary model の基本状態を確認する。"""
    model = binary_mixed_model_bundle["model"]
    train_x = binary_mixed_model_bundle["train_x"]
    train_y = binary_mixed_model_bundle["train_y"]
    cat_dims = binary_mixed_model_bundle["cat_dims"]

    assert_model_training(
        model=model,
        train_x=train_x,
        train_y=train_y,
        cat_dims=cat_dims,
    )


# ============================================================
# pytest: Acquisition forward tests
# ============================================================

def test_binary_acquisition_forward_shapes(
    binary_model_bundle: dict[str, Any],
) -> None:
    """
    通常 binary model で各 acquisition の forward が
    shape 正常・finite であることを確認する。
    """
    model = binary_model_bundle["model"]
    train_x = binary_model_bundle["train_x"]
    bounds = binary_model_bundle["bounds"]

    X = make_random_batch(
        bounds=bounds,
        batch_size=8,
        q=3,
    )

    for acqf_cls, acqf_kwargs, case_id in acquisition_cases(model, train_x):
        acqf = acqf_cls(
            model=model,
            **acqf_kwargs,
        )

        values = acqf(X)

        assert values.shape == torch.Size([8]), (
            f"{case_id}: acquisition output shape が想定外です。"
            f"expected=torch.Size([8]), actual={values.shape}"
        )

        assert torch.isfinite(values).all(), (
            f"{case_id}: acquisition value に NaN/inf が含まれます。"
            f"values={values}"
        )


def test_binary_mixed_acquisition_forward_shapes(
    binary_mixed_model_bundle: dict[str, Any],
) -> None:
    """
    mixed binary model で各 acquisition の forward が
    shape 正常・finite であることを確認する。
    """
    model = binary_mixed_model_bundle["model"]
    train_x = binary_mixed_model_bundle["train_x"]
    bounds = binary_mixed_model_bundle["bounds"]
    cat_dims = binary_mixed_model_bundle["cat_dims"]

    X = make_random_mixed_batch(
        bounds=bounds,
        cat_dims=cat_dims,
        batch_size=8,
        q=3,
    )

    for acqf_cls, acqf_kwargs, case_id in acquisition_cases(model, train_x):
        acqf = acqf_cls(
            model=model,
            **acqf_kwargs,
        )

        values = acqf(X)

        assert values.shape == torch.Size([8]), (
            f"{case_id}: acquisition output shape が想定外です。"
            f"expected=torch.Size([8]), actual={values.shape}"
        )

        assert torch.isfinite(values).all(), (
            f"{case_id}: acquisition value に NaN/inf が含まれます。"
            f"values={values}"
        )


# ============================================================
# pytest: optimize_acqf smoke tests
# ============================================================

@pytest.mark.slow
def test_binary_acquisition_optimize_acqf_smoke(
    binary_model_bundle: dict[str, Any],
) -> None:
    """
    通常 binary model で各 acquisition を optimize_acqf に通す smoke test。

    注意:
        BadInitialCandidatesWarning は確認しない。
    """
    model = binary_model_bundle["model"]
    train_x = binary_model_bundle["train_x"]
    bounds = binary_model_bundle["bounds"]

    q = 3
    d = train_x.shape[-1]

    for acqf_cls, acqf_kwargs, case_id in acquisition_cases(model, train_x):
        acqf = acqf_cls(
            model=model,
            **acqf_kwargs,
        )

        cands, acq_value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            sequential=True,
            num_restarts=5,
            raw_samples=64,
            options={"maxiter": 30},
        )

        assert cands.shape == torch.Size([q, d]), (
            f"{case_id}: 候補点 shape が想定外です。"
            f"expected={torch.Size([q, d])}, actual={cands.shape}"
        )

        assert torch.isfinite(cands).all(), (
            f"{case_id}: 候補点に NaN/inf が含まれます。"
            f"cands={cands}"
        )

        assert torch.isfinite(acq_value).all(), (
            f"{case_id}: acq_value に NaN/inf が含まれます。"
            f"acq_value={acq_value}"
        )

        assert_candidates_in_bounds(
            cands=cands,
            bounds=bounds,
        )


@pytest.mark.slow
def test_binary_mixed_acquisition_optimize_acqf_mixed_smoke(
    binary_mixed_model_bundle: dict[str, Any],
) -> None:
    """
    mixed binary model で各 acquisition を optimize_acqf_mixed に通す smoke test。

    注意:
        BadInitialCandidatesWarning は確認しない。
    """
    model = binary_mixed_model_bundle["model"]
    train_x = binary_mixed_model_bundle["train_x"]
    bounds = binary_mixed_model_bundle["bounds"]
    cat_dims = binary_mixed_model_bundle["cat_dims"]

    assert cat_dims == [5]
    cat_id = cat_dims[0]

    q = 3
    d = train_x.shape[-1]

    fixed_features_list = [
        {cat_id: 5.0},
        {cat_id: 10.0},
        {cat_id: 15.0},
    ]

    cat_values = torch.tensor(
        [5.0, 10.0, 15.0],
        dtype=train_x.dtype,
        device=train_x.device,
    )

    for acqf_cls, acqf_kwargs, case_id in acquisition_cases(model, train_x):
        acqf = acqf_cls(
            model=model,
            **acqf_kwargs,
        )

        cands, acq_value = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            fixed_features_list=fixed_features_list,
            num_restarts=5,
            raw_samples=64,
            options={"maxiter": 30},
        )

        assert cands.shape == torch.Size([q, d]), (
            f"{case_id}: 候補点 shape が想定外です。"
            f"expected={torch.Size([q, d])}, actual={cands.shape}"
        )

        assert torch.isfinite(cands).all(), (
            f"{case_id}: 候補点に NaN/inf が含まれます。"
            f"cands={cands}"
        )

        assert torch.isfinite(acq_value).all(), (
            f"{case_id}: acq_value に NaN/inf が含まれます。"
            f"acq_value={acq_value}"
        )

        assert_candidates_in_bounds(
            cands=cands,
            bounds=bounds,
        )

        assert torch.isin(cands[:, cat_id], cat_values).all(), (
            f"{case_id}: カテゴリ候補が想定値以外です。"
            f"expected={cat_values}, actual={cands[:, cat_id]}"
        )


@pytest.mark.slow
def test_binary_optimizer_constraint_compatibility_smoke() -> None:
    """
    代表 acquisition で optimizer / constraint / step / k-sparse の互換性を確認する。

    代表シナリオには以下を含める:
        - base + step + k-sparse + 線形制約
        - evo(cmaes) + step + k-sparse + 線形制約
        - torch(adam) + step + k-sparse + 線形制約
    """
    bundle = create_binary_model_bundle(
        cat=False,
        n=30,
        d=5,
        num_epochs=30,
    )

    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]

    q = 3
    d = train_x.shape[-1]

    for (
        acqf_cls,
        acqf_kwargs,
        acq_id,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in optimizer_constraint_smoke_scenarios(model, train_x, bounds):
        acqf = acqf_cls(
            model=model,
            **acqf_kwargs,
        )

        cands, acq_value = optimize_with_case(
            acqf=acqf,
            bounds=bounds,
            q=q,
            optimize_func=optimize_func,
            optimize_method=optimize_method,
            constraint_case=constraint_case,
            num_restarts=5,
            raw_samples=64,
            maxiter=30,
        )

        assert_optimizer_compatibility_result(
            cands=cands,
            acq_value=acq_value,
            bounds=bounds,
            q=q,
            d=d,
            constraint_case=constraint_case,
            case_id=case_id,
        )



@pytest.mark.slow
def test_binary_mixed_optimizer_constraint_compatibility_smoke() -> None:
    """
    mixed binary model でも optimizer / constraint / step / k-sparse の互換性を確認する。

    代表シナリオは通常モデルと同じものを使い、カテゴリ列は fixed_features_list で
    {5, 10, 15} の候補として扱う。
    """
    bundle = create_binary_model_bundle(
        cat=True,
        n=30,
        d=5,
        num_epochs=30,
    )

    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]

    assert cat_dims == [5]
    cat_id = cat_dims[0]

    fixed_features_list = [
        {cat_id: 5.0},
        {cat_id: 10.0},
        {cat_id: 15.0},
    ]

    cat_values = torch.tensor(
        [5.0, 10.0, 15.0],
        dtype=train_x.dtype,
        device=train_x.device,
    )

    q = 3
    d_total = train_x.shape[-1]

    for (
        acqf_cls,
        acqf_kwargs,
        acq_id,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in optimizer_constraint_smoke_scenarios(model, train_x, bounds):
        mixed_case_id = f"mixed__{case_id}"

        acqf = acqf_cls(
            model=model,
            **acqf_kwargs,
        )

        cands, acq_value = optimize_mixed_with_case(
            acqf=acqf,
            bounds=bounds,
            q=q,
            fixed_features_list=fixed_features_list,
            optimize_func=optimize_func,
            optimize_method=optimize_method,
            constraint_case=constraint_case,
            num_restarts=5,
            raw_samples=64,
            maxiter=30,
        )

        assert_optimizer_compatibility_result(
            cands=cands,
            acq_value=acq_value,
            bounds=bounds,
            q=q,
            d=d_total,
            constraint_case=constraint_case,
            case_id=mixed_case_id,
        )

        assert torch.isin(cands[:, cat_id], cat_values).all(), (
            f"{mixed_case_id}: カテゴリ候補が想定値以外です。"
            f"expected={cat_values}, actual={cands[:, cat_id]}"
        )


# ============================================================
# Jupyter helpers
# ============================================================

def run_jupyter_forward_check(
    *,
    cat: bool = False,
    n: int = 20,
    d: int = 5,
    num_epochs: int = 60,
    batch_size: int = 8,
    q: int = 3,
    verbose_forward_detail: bool = False,
) -> dict[str, Any]:
    """
    Jupyter 上で acquisition forward の確認をするための helper。

    Args:
        verbose_forward_detail:
            True の場合、shape / min / max / finite を表示する。
            False の場合、OK ケースは表示せず、NG の場合だけ表示する。
    """
    bundle = create_binary_model_bundle(
        cat=cat,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )

    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]

    if cat:
        X = make_random_mixed_batch(
            bounds=bounds,
            cat_dims=cat_dims,
            batch_size=batch_size,
            q=q,
        )
    else:
        X = make_random_batch(
            bounds=bounds,
            batch_size=batch_size,
            q=q,
        )

    print("=" * 80)
    print(f"Jupyter forward check cat={cat}")
    if verbose_forward_detail:
        print(f"train_x.shape={train_x.shape}")
        print(f"bounds.shape={bounds.shape}")
        print(f"X.shape={X.shape}")
    print("=" * 80)

    for acqf_cls, acqf_kwargs, case_id in acquisition_cases(model, train_x):
        try:
            acqf = acqf_cls(
                model=model,
                **acqf_kwargs,
            )

            values = acqf(X)

            assert values.shape == torch.Size([batch_size]), (
                f"{case_id}: acquisition output shape が想定外です。"
                f"expected={torch.Size([batch_size])}, actual={values.shape}"
            )
            assert torch.isfinite(values).all(), (
                f"{case_id}: acquisition value に NaN/inf が含まれます。"
                f"values={values}"
            )

            if verbose_forward_detail:
                print(
                    f"[OK] {case_id} "
                    f"shape={tuple(values.shape)} "
                    f"min={values.min().item():.6g} "
                    f"max={values.max().item():.6g} "
                    f"finite={bool(torch.isfinite(values).all())}"
                )

        except Exception as exc:
            print(f"[NG] {case_id} {type(exc).__name__}")
            print(str(exc))
            raise

    print("forward check passed.")

    return bundle



def run_jupyter_optimize_acqf_all_acquisitions_check(
    *,
    n: int = 20,
    d: int = 5,
    num_epochs: int = 30,
    q: int = 3,
    num_restarts: int = 5,
    raw_samples: int = 64,
    maxiter: int = 30,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    """
    Jupyter 上で通常 binary model + optimize_acqf を全 acquisition で確認する。

    方針:
        - acquisition は acquisition_cases() の全件を試す。
        - optimizer は標準の optimize_acqf のみ。
        - evo / torch / 制約 / step / k-sparse はここでは扱わない。

    Example:
        bundle = run_jupyter_optimize_acqf_all_acquisitions_check(num_epochs=30)
    """
    bundle = create_binary_model_bundle(
        cat=False,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )

    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]

    print("=" * 100)
    print("Jupyter optimize_acqf check: all acquisitions")
    print(f"n={n}, d={d}, q={q}, num_epochs={num_epochs}")
    print(f"num_acquisitions={len(acquisition_cases(model, train_x))}")
    print("=" * 100)

    failed_cases: list[tuple[str, Exception]] = []

    for acqf_cls, acqf_kwargs, case_id in acquisition_cases(model, train_x):
        display_id = f"optimize_acqf__{case_id}"

        try:
            acqf = acqf_cls(
                model=model,
                **acqf_kwargs,
            )

            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings
            ):
                cands, acq_value = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=q,
                    sequential=True,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"maxiter": maxiter},
                )

            assert cands.shape == torch.Size([q, train_x.shape[-1]]), (
                f"{case_id}: 候補点 shape が想定外です。"
                f"expected={torch.Size([q, train_x.shape[-1]])}, actual={cands.shape}"
            )
            assert torch.isfinite(cands).all(), (
                f"{case_id}: 候補点に NaN/inf が含まれます。cands={cands}"
            )
            assert torch.isfinite(acq_value).all(), (
                f"{case_id}: acq_value に NaN/inf が含まれます。acq_value={acq_value}"
            )
            assert_candidates_in_bounds(cands, bounds)

            print(f"[OK] {display_id}")

        except Exception as exc:
            print(f"[NG] {display_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((display_id, exc))

            if not continue_on_error:
                raise

    print("=" * 100)
    if failed_cases:
        print(f"failed_cases={len(failed_cases)}")
        for case_id, exc in failed_cases:
            print(f"  - optimize_acqf__{case_id}: {type(exc).__name__}: {exc}")
    else:
        print("all optimize_acqf acquisition checks passed.")
    print("=" * 100)

    return bundle


def run_jupyter_optimize_acqf_mixed_all_acquisitions_check(
    *,
    n: int = 20,
    d: int = 5,
    num_epochs: int = 30,
    q: int = 3,
    num_restarts: int = 5,
    raw_samples: int = 64,
    maxiter: int = 30,
    continue_on_error: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    """
    Jupyter 上で mixed binary model + optimize_acqf_mixed を全 acquisition で確認する。

    方針:
        - acquisition は acquisition_cases() の全件を試す。
        - optimizer は標準の optimize_acqf_mixed のみ。
        - evo / torch / 制約 / step / k-sparse はここでは扱わない。

    Example:
        mixed_bundle = run_jupyter_optimize_acqf_mixed_all_acquisitions_check(num_epochs=30)
    """
    bundle = create_binary_model_bundle(
        cat=True,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )

    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]

    assert cat_dims == [d]
    cat_id = cat_dims[0]

    fixed_features_list = [
        {cat_id: 5.0},
        {cat_id: 10.0},
        {cat_id: 15.0},
    ]

    cat_values = torch.tensor(
        [5.0, 10.0, 15.0],
        dtype=train_x.dtype,
        device=train_x.device,
    )

    print("=" * 100)
    print("Jupyter optimize_acqf_mixed check: all acquisitions")
    print(f"n={n}, d={d}, q={q}, num_epochs={num_epochs}, cat_dims={cat_dims}")
    print(f"num_acquisitions={len(acquisition_cases(model, train_x))}")
    print("=" * 100)

    failed_cases: list[tuple[str, Exception]] = []

    for acqf_cls, acqf_kwargs, case_id in acquisition_cases(model, train_x):
        display_id = f"optimize_acqf_mixed__{case_id}"

        try:
            acqf = acqf_cls(
                model=model,
                **acqf_kwargs,
            )

            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings
            ):
                cands, acq_value = optimize_acqf_mixed(
                    acq_function=acqf,
                    bounds=bounds,
                    q=q,
                    fixed_features_list=fixed_features_list,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"maxiter": maxiter},
                )

            assert cands.shape == torch.Size([q, train_x.shape[-1]]), (
                f"{case_id}: 候補点 shape が想定外です。"
                f"expected={torch.Size([q, train_x.shape[-1]])}, actual={cands.shape}"
            )
            assert torch.isfinite(cands).all(), (
                f"{case_id}: 候補点に NaN/inf が含まれます。cands={cands}"
            )
            assert torch.isfinite(acq_value).all(), (
                f"{case_id}: acq_value に NaN/inf が含まれます。acq_value={acq_value}"
            )
            assert_candidates_in_bounds(cands, bounds)
            assert torch.isin(cands[:, cat_id], cat_values).all(), (
                f"{case_id}: カテゴリ候補が想定値以外です。"
                f"expected={cat_values}, actual={cands[:, cat_id]}"
            )

            print(f"[OK] {display_id}")

        except Exception as exc:
            print(f"[NG] {display_id} {type(exc).__name__}")
            print(str(exc))
            failed_cases.append((display_id, exc))

            if not continue_on_error:
                raise

    print("=" * 100)
    if failed_cases:
        print(f"failed_cases={len(failed_cases)}")
        for case_id, exc in failed_cases:
            print(f"  - optimize_acqf_mixed__{case_id}: {type(exc).__name__}: {exc}")
    else:
        print("all optimize_acqf_mixed acquisition checks passed.")
    print("=" * 100)

    return bundle



def run_jupyter_optimizer_constraint_compatibility_check(
    *,
    n: int = 30,
    d: int = 5,
    num_epochs: int = 30,
    q: int = 3,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    """
    Jupyter 上で optimizer / constraint / step / k-sparse の動作確認をする。

    Args:
        n: 初期データ数。
        d: 入力次元。制約ケースは d >= 5 を想定。
        num_epochs: fit epoch 数。
        q: q-batch size。
        full_matrix:
            False:
                pytest と同じ代表シナリオのみ実行。
            True:
                representative acquisitions × optimizer cases × constraint cases
                の全組み合わせを実行。
        continue_on_error:
            True の場合、失敗しても次のケースに進む。
            False の場合、最初の失敗で例外を投げる。
        verbose_ok_detail:
            True の場合、OK ケースで cands.shape と acq_value も表示する。
            False の場合、OK ケースは [OK ] case_id のみ表示する。
        verbose_candidates:
            True の場合、OK ケースでも cands や簡易集計を表示する。
            False の場合、cands や簡易集計は表示しない。
        verbose_constraints:
            True の場合、OK ケースでも制約診断を表示する。
            False の場合、制約違反がある場合だけ表示する。

    Returns:
        bundle。
    """
    if d < 5:
        raise ValueError("constraint compatibility check では d >= 5 が必要です。")

    bundle = create_binary_model_bundle(
        cat=False,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )

    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]

    if full_matrix:
        scenarios = []
        for acqf_cls, acqf_kwargs, acq_id in representative_acquisition_cases(model, train_x):
            for optimize_func, optimize_method, optimizer_id in optimizer_cases():
                for constraint_case in make_constraint_cases(bounds):
                    constraint_id = constraint_case["case_id"]

                    case_id = f"{acq_id}__{optimizer_id}__{constraint_id}"

                    scenarios.append(
                        (
                            acqf_cls,
                            acqf_kwargs,
                            acq_id,
                            optimize_func,
                            optimize_method,
                            constraint_case,
                            case_id,
                        )
                    )
    else:
        scenarios = optimizer_constraint_smoke_scenarios(
            model=model,
            train_x=train_x,
            bounds=bounds,
        )

    print("=" * 100)
    print("Jupyter optimizer / constraint compatibility check")
    print(
        f"n={n}, d={d}, q={q}, num_epochs={num_epochs}, "
        f"full_matrix={full_matrix}, "
        f"verbose_ok_detail={verbose_ok_detail}, "
        f"verbose_candidates={verbose_candidates}, "
        f"verbose_constraints={verbose_constraints}"
    )
    print(f"num_cases={len(scenarios)}")
    print("=" * 100)

    failed_cases: list[tuple[str, Exception]] = []

    for (
        acqf_cls,
        acqf_kwargs,
        acq_id,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in scenarios:
        try:
            acqf = acqf_cls(
                model=model,
                **acqf_kwargs,
            )

            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings
            ):
                cands, acq_value = optimize_with_case(
                    acqf=acqf,
                    bounds=bounds,
                    q=q,
                    optimize_func=optimize_func,
                    optimize_method=optimize_method,
                    constraint_case=constraint_case,
                    num_restarts=5,
                    raw_samples=64,
                    maxiter=30,
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

            if verbose_ok_detail:
                print(
                    f"[OK] {case_id} "
                    f"cands.shape={tuple(cands.shape)} "
                    f"acq_value={acq_value}"
                )
            else:
                print(f"[OK] {case_id}")

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

    print("=" * 100)
    if failed_cases:
        print(f"failed_cases={len(failed_cases)}")
        for case_id, exc in failed_cases:
            print(f"  - {case_id}: {type(exc).__name__}: {exc}")
    else:
        print("all optimizer / constraint compatibility checks passed.")
    print("=" * 100)

    return bundle


def run_jupyter_mixed_optimizer_constraint_compatibility_check(
    *,
    n: int = 30,
    d: int = 5,
    num_epochs: int = 30,
    q: int = 3,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> dict[str, Any]:
    """
    Jupyter 上で mixed model の optimizer / constraint / step / k-sparse 動作確認をする。

    Args:
        n: 初期データ数。
        d: 連続変数の次元。mixed model では全体次元は d + 1。
        num_epochs: fit epoch 数。
        q: q-batch size。
        full_matrix:
            False:
                pytest と同じ代表シナリオのみ実行。
            True:
                representative acquisitions × optimizer cases × constraint cases
                の全組み合わせを実行。
        continue_on_error:
            True の場合、失敗しても次のケースに進む。
            False の場合、最初の失敗で例外を投げる。
        verbose_ok_detail:
            True の場合、OK ケースで cands.shape と acq_value も表示する。
            False の場合、OK ケースは [OK ] case_id のみ表示する。
        verbose_candidates:
            True の場合、OK ケースでも cands や簡易集計を表示する。
            False の場合、cands や簡易集計は表示しない。
        verbose_constraints:
            True の場合、OK ケースでも制約診断を表示する。
            False の場合、制約違反がある場合だけ表示する。

    Returns:
        bundle。
    """
    if d < 5:
        raise ValueError("constraint compatibility check では d >= 5 が必要です。")

    bundle = create_binary_model_bundle(
        cat=True,
        n=n,
        d=d,
        num_epochs=num_epochs,
    )

    model = bundle["model"]
    train_x = bundle["train_x"]
    bounds = bundle["bounds"]
    cat_dims = bundle["cat_dims"]

    assert cat_dims == [d]
    cat_id = cat_dims[0]

    fixed_features_list = [
        {cat_id: 5.0},
        {cat_id: 10.0},
        {cat_id: 15.0},
    ]

    cat_values = torch.tensor(
        [5.0, 10.0, 15.0],
        dtype=train_x.dtype,
        device=train_x.device,
    )

    if full_matrix:
        scenarios = []
        for acqf_cls, acqf_kwargs, acq_id in representative_acquisition_cases(model, train_x):
            for optimize_func, optimize_method, optimizer_id in optimizer_cases():
                for constraint_case in make_constraint_cases(bounds):
                    constraint_id = constraint_case["case_id"]
                    case_id = f"mixed__{acq_id}__{optimizer_id}__{constraint_id}"

                    scenarios.append(
                        (
                            acqf_cls,
                            acqf_kwargs,
                            acq_id,
                            optimize_func,
                            optimize_method,
                            constraint_case,
                            case_id,
                        )
                    )
    else:
        scenarios = []
        for (
            acqf_cls,
            acqf_kwargs,
            acq_id,
            optimize_func,
            optimize_method,
            constraint_case,
            case_id,
        ) in optimizer_constraint_smoke_scenarios(model, train_x, bounds):
            scenarios.append(
                (
                    acqf_cls,
                    acqf_kwargs,
                    acq_id,
                    optimize_func,
                    optimize_method,
                    constraint_case,
                    f"mixed__{case_id}",
                )
            )

    print("=" * 100)
    print("Jupyter mixed optimizer / constraint compatibility check")
    print(
        f"n={n}, d={d}, q={q}, num_epochs={num_epochs}, "
        f"full_matrix={full_matrix}, "
        f"verbose_ok_detail={verbose_ok_detail}, "
        f"verbose_candidates={verbose_candidates}, "
        f"verbose_constraints={verbose_constraints}"
    )
    print(f"cat_dims={cat_dims}")
    print(f"num_cases={len(scenarios)}")
    print("=" * 100)

    failed_cases: list[tuple[str, Exception]] = []

    for (
        acqf_cls,
        acqf_kwargs,
        acq_id,
        optimize_func,
        optimize_method,
        constraint_case,
        case_id,
    ) in scenarios:
        try:
            acqf = acqf_cls(
                model=model,
                **acqf_kwargs,
            )

            with maybe_suppress_botorch_initial_warnings(
                suppress=suppress_botorch_warnings
            ):
                cands, acq_value = optimize_mixed_with_case(
                    acqf=acqf,
                    bounds=bounds,
                    q=q,
                    fixed_features_list=fixed_features_list,
                    optimize_func=optimize_func,
                    optimize_method=optimize_method,
                    constraint_case=constraint_case,
                    num_restarts=5,
                    raw_samples=64,
                    maxiter=30,
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

            assert torch.isin(cands[:, cat_id], cat_values).all(), (
                f"{case_id}: カテゴリ候補が想定値以外です。"
                f"expected={cat_values}, actual={cands[:, cat_id]}"
            )

            if verbose_ok_detail:
                print(
                    f"[OK] {case_id} "
                    f"cands.shape={tuple(cands.shape)} "
                    f"cat={cands[:, cat_id].detach().cpu().tolist()} "
                    f"acq_value={acq_value}"
                )
            else:
                print(f"[OK] {case_id}")

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

    print("=" * 100)
    if failed_cases:
        print(f"failed_cases={len(failed_cases)}")
        for case_id, exc in failed_cases:
            print(f"  - {case_id}: {type(exc).__name__}: {exc}")
    else:
        print("all mixed optimizer / constraint compatibility checks passed.")
    print("=" * 100)

    return bundle



def run_jupyter_all_checks(
    *,
    num_epochs: int = 30,
    run_optimize: bool = True,
    full_matrix: bool = False,
    continue_on_error: bool = False,
    verbose_forward_detail: bool = False,
    verbose_ok_detail: bool = False,
    verbose_candidates: bool = False,
    verbose_constraints: bool = False,
    suppress_botorch_warnings: bool = True,
) -> None:
    """
    Jupyter 上でまとめて確認する helper。

    Args:
        num_epochs: fit の epoch 数。
        run_optimize:
            False の場合は forward check のみ。
            True の場合は以下を確認する。デフォルトは True。
                1. optimize_acqf 全 acquisition
                2. optimize_acqf_mixed 全 acquisition
                3. evo / torch / 制約 / step / k-sparse の代表 acquisition
                   通常モデルと mixed モデルの両方
        full_matrix:
            optimizer / constraint compatibility で代表 acquisition × optimizer × 制約ケースの
            全組み合わせを回すか。optimize_acqf / optimize_acqf_mixed 側は常に全 acquisition。
        verbose_forward_detail:
            True の場合、forward check の OK ケースで shape / min / max / finite も表示する。
            False の場合、forward check は NG の場合のみ表示する。
        continue_on_error:
            True の場合、失敗しても次のケースに進む。
        verbose_ok_detail:
            True の場合、optimizer / constraint compatibility の OK ケースで
            cands.shape と acq_value も表示する。
            False の場合、OK ケースは [OK ] case_id のみ表示する。
        verbose_candidates:
            True の場合、optimizer / constraint compatibility の OK ケースでも
            cands や簡易集計を表示する。
        verbose_constraints:
            True の場合、optimizer / constraint compatibility の OK ケースでも
            制約診断を表示する。
    """
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
        # 1) 標準 optimize_acqf / optimize_acqf_mixed は全 acquisition で確認する。
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

        # 2) evo / torch / 制約 / step / k-sparse は代表 acquisition のみで確認する。
        #    通常モデルと mixed モデルの両方を実行する。
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

    print("all Jupyter checks passed.")
