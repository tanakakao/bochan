from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize

from bochan.models.classification.binary.robust import (
    HeteroscedasticBinaryClassificationGPModel,
    HeteroscedasticBinaryClassificationMixedGPModel,
)
from bochan.acquisition.binary.active_learning import (
    qHeteroBinaryPredictiveEntropy,
    qHeteroBinaryBALD,
    qHeteroBinaryProbabilityVariance,
    qHeteroBinaryMarginUncertainty,
)
from bochan.acquisition.binary.levelset_estimation import (
    qHeteroBinaryLatentStraddleAcquisition,
    qHeteroBinaryICUAcquisition,
    qHeteroBinaryBoundaryVarianceAcquisition,
    qHeteroBinaryClassEntropyAcquisition,
)
from bochan.acquisition.binary.bayesian_optimization import (
    qHeteroBinaryExpectedImprovement,
    qHeteroBinaryProbabilityOfImprovement,
    qHeteroBinaryUpperConfidenceBound,
    compute_hetero_binary_classification_best_f,
)
from tests.test_binary_classification_base_single_output import (
    make_binary_toy_data,
    assert_model_training,
    make_random_batch,
    make_random_mixed_batch,
)


def _build_input_transform(train_x: torch.Tensor, bounds: torch.Tensor, cat_dims: list[int]) -> Normalize:
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(d=train_x.shape[-1], bounds=bounds, indices=cont_indices)


def create_binary_hetero_model_bundle(*, cat: bool = False) -> dict[str, Any]:
    train_x, train_y, bounds = make_binary_toy_data(cat=cat)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    kwargs: dict[str, Any] = {
        "train_X": train_x,
        "train_Y": train_y,
        "train_Yvar": 0.05 * torch.ones_like(train_y),
        "input_transform": _build_input_transform(train_x, bounds, cat_dims),
    }
    if cat:
        kwargs["cat_dims"] = cat_dims
        model = HeteroscedasticBinaryClassificationMixedGPModel(**kwargs)
    else:
        model = HeteroscedasticBinaryClassificationGPModel(**kwargs)

    assert_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "bounds": bounds, "cat_dims": cat_dims}


def hetero_acquisition_cases(model: Any, train_x: torch.Tensor) -> list[tuple[type, dict[str, Any], str]]:
    best_f = compute_hetero_binary_classification_best_f(model, train_x, apply_sigmoid_if_needed=True)
    return [
        (qHeteroBinaryPredictiveEntropy, {}, "hetero_predictive_entropy"),
        (qHeteroBinaryBALD, {}, "hetero_bald"),
        (qHeteroBinaryProbabilityVariance, {}, "hetero_prob_var"),
        (qHeteroBinaryMarginUncertainty, {}, "hetero_margin"),
        (qHeteroBinaryLatentStraddleAcquisition, {}, "hetero_straddle"),
        (qHeteroBinaryICUAcquisition, {}, "hetero_icu"),
        (qHeteroBinaryBoundaryVarianceAcquisition, {}, "hetero_bv"),
        (qHeteroBinaryClassEntropyAcquisition, {}, "hetero_class_entropy"),
        (qHeteroBinaryExpectedImprovement, {"best_f": (best_f - 0.05).clamp(1e-6, 1-1e-6), "apply_sigmoid_if_needed": True}, "hetero_ei"),
        (qHeteroBinaryProbabilityOfImprovement, {"best_f": (best_f - 0.05).clamp(1e-6, 1-1e-6), "apply_sigmoid_if_needed": True}, "hetero_pi"),
        (qHeteroBinaryUpperConfidenceBound, {}, "hetero_ucb"),
    ]


@pytest.fixture(scope="module")
def binary_hetero_model_bundle() -> dict[str, Any]:
    return create_binary_hetero_model_bundle(cat=False)


@pytest.fixture(scope="module")
def binary_hetero_mixed_model_bundle() -> dict[str, Any]:
    return create_binary_hetero_model_bundle(cat=True)


def test_binary_hetero_acquisition_forward_shapes(binary_hetero_model_bundle: dict[str, Any]) -> None:
    model = binary_hetero_model_bundle["model"]
    X = make_random_batch(binary_hetero_model_bundle["bounds"], batch_size=4, q=2)
    for acq_cls, kwargs, _ in hetero_acquisition_cases(model, binary_hetero_model_bundle["train_x"]):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4])
        assert torch.isfinite(out).all()


def test_binary_hetero_mixed_acquisition_forward_shapes(binary_hetero_mixed_model_bundle: dict[str, Any]) -> None:
    model = binary_hetero_mixed_model_bundle["model"]
    X = make_random_mixed_batch(
        binary_hetero_mixed_model_bundle["bounds"],
        binary_hetero_mixed_model_bundle["cat_dims"],
        batch_size=4,
        q=2,
    )
    for acq_cls, kwargs, _ in hetero_acquisition_cases(model, binary_hetero_mixed_model_bundle["train_x"]):
        out = acq_cls(model=model, **kwargs)(X)
        assert out.shape == torch.Size([4])
        assert torch.isfinite(out).all()


def run_jupyter_all_checks(*, num_epochs: int = 0) -> dict[str, Any]:
    """Jupyter 向け: hetero 専用 acquisition を含む一括 check。"""
    _ = num_epochs  # API 互換用（本テストでは学習ループ未使用）

    single_bundle = create_binary_hetero_model_bundle(cat=False)
    mixed_bundle = create_binary_hetero_model_bundle(cat=True)

    test_binary_hetero_acquisition_forward_shapes(single_bundle)
    test_binary_hetero_mixed_acquisition_forward_shapes(mixed_bundle)

    return {"single": single_bundle, "mixed": mixed_bundle}
