from __future__ import annotations

from typing import Any

from bochan.models.classification.binary.high_dim import (
    PCABinaryClassificationGPModel,
    PCABinaryClassificationMixedGPModel,
    REMBOBinaryClassificationGPModel,
    REMBOBinaryClassificationMixedGPModel,
)
from tests.test_binary_classification_base_single_output import make_binary_toy_data, assert_model_training


def create_binary_pca_model_bundle(*, cat: bool = False) -> dict[str, Any]:
    train_x, train_y, _ = make_binary_toy_data(cat=cat, d=6)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    if cat:
        model = PCABinaryClassificationMixedGPModel(train_X=train_x, train_Y=train_y, cat_dims=cat_dims, projected_dim=3)
    else:
        model = PCABinaryClassificationGPModel(train_X=train_x, train_Y=train_y, projected_dim=3)
    assert_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "cat_dims": cat_dims}


def create_binary_rembo_model_bundle(*, cat: bool = False) -> dict[str, Any]:
    train_x, train_y, _ = make_binary_toy_data(cat=cat, d=6)
    cat_dims = [train_x.shape[-1] - 1] if cat else []
    if cat:
        model = REMBOBinaryClassificationMixedGPModel(train_X=train_x, train_Y=train_y, cat_dims=cat_dims, projected_dim=3)
    else:
        model = REMBOBinaryClassificationGPModel(train_X=train_x, train_Y=train_y, projected_dim=3)
    assert_model_training(model=model, train_x=train_x, train_y=train_y, cat_dims=cat_dims)
    return {"model": model, "train_x": train_x, "train_y": train_y, "cat_dims": cat_dims}


def test_pca_binary_basic_behavior() -> None:
    create_binary_pca_model_bundle(cat=False)


def test_pca_mixed_binary_basic_behavior() -> None:
    create_binary_pca_model_bundle(cat=True)


def test_rembo_binary_basic_behavior() -> None:
    create_binary_rembo_model_bundle(cat=False)


def test_rembo_mixed_binary_basic_behavior() -> None:
    create_binary_rembo_model_bundle(cat=True)


def run_jupyter_all_checks() -> dict[str, Any]:
    """Jupyter 向け: PCA/REMBO の single/mixed 基本確認を一括実行。"""
    pca_single = create_binary_pca_model_bundle(cat=False)
    pca_mixed = create_binary_pca_model_bundle(cat=True)
    rembo_single = create_binary_rembo_model_bundle(cat=False)
    rembo_mixed = create_binary_rembo_model_bundle(cat=True)

    return {
        "pca_single": pca_single,
        "pca_mixed": pca_mixed,
        "rembo_single": rembo_single,
        "rembo_mixed": rembo_mixed,
    }
