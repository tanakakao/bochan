from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.models.transforms.input import Normalize

from bochan.models.classification.binary.robust import (
    OutlierRelevancePursuitBinaryClassificationGPModel,
    OutlierRelevancePursuitBinaryClassificationMixedGPModel,
)
from tests.test_binary_classification_base_single_output import (
    make_binary_toy_data,
    assert_model_training,
    acquisition_cases,
    make_random_batch,
    make_random_mixed_batch,
)


def _build_input_transform(train_x: torch.Tensor, bounds: torch.Tensor, cat_dims: list[int]) -> Normalize:
    cont_indices = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    return Normalize(d=train_x.shape[-1], bounds=bounds, indices=cont_indices)


def create_binary_rrp_model_bundle(*, cat: bool = False) -> dict[str, Any]:
    train_x, train_y, bounds = make_binary_toy_data(cat=cat)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    kwargs: dict[str, Any] = {
        "train_X": train_x,
        "train_Y": train_y,
        "input_transform": _build_input_transform(train_x, bounds, cat_dims),
    }

    if cat:
        kwargs["cat_dims"] = cat_dims
        model = OutlierRelevancePursuitBinaryClassificationMixedGPModel(**kwargs)
    else:
        model = OutlierRelevancePursuitBinaryClassificationGPModel(**kwargs)

    assert_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "bounds": bounds, "cat_dims": cat_dims}


@pytest.fixture(scope="module")
def binary_rrp_model_bundle() -> dict[str, Any]:
    return create_binary_rrp_model_bundle(cat=False)


@pytest.fixture(scope="module")
def binary_rrp_mixed_model_bundle() -> dict[str, Any]:
    return create_binary_rrp_model_bundle(cat=True)


def test_binary_rrp_acquisition_forward_shapes(binary_rrp_model_bundle: dict[str, Any]) -> None:
    model = binary_rrp_model_bundle["model"]
    train_x = binary_rrp_model_bundle["train_x"]
    X = make_random_batch(binary_rrp_model_bundle["bounds"], batch_size=4, q=2)

    for acq_cls, kwargs, _ in acquisition_cases(model=model, train_x=train_x):
        acq = acq_cls(model=model, **kwargs)
        out = acq(X)
        assert out.shape == torch.Size([4])
        assert torch.isfinite(out).all()


def test_binary_rrp_mixed_acquisition_forward_shapes(binary_rrp_mixed_model_bundle: dict[str, Any]) -> None:
    model = binary_rrp_mixed_model_bundle["model"]
    train_x = binary_rrp_mixed_model_bundle["train_x"]
    X = make_random_mixed_batch(binary_rrp_mixed_model_bundle["bounds"], binary_rrp_mixed_model_bundle["cat_dims"], batch_size=4, q=2)

    for acq_cls, kwargs, _ in acquisition_cases(model=model, train_x=train_x):
        acq = acq_cls(model=model, **kwargs)
        out = acq(X)
        assert out.shape == torch.Size([4])
        assert torch.isfinite(out).all()


def run_jupyter_all_checks(*, num_epochs: int = 0) -> dict[str, Any]:
    """Jupyter 向け: single / mixed の主要 forward check をまとめて実行する。"""
    _ = num_epochs  # API 互換用（本テストでは学習ループ未使用）

    single_bundle = create_binary_rrp_model_bundle(cat=False)
    mixed_bundle = create_binary_rrp_model_bundle(cat=True)

    test_binary_rrp_acquisition_forward_shapes(single_bundle)
    test_binary_rrp_mixed_acquisition_forward_shapes(mixed_bundle)

    return {"single": single_bundle, "mixed": mixed_bundle}
