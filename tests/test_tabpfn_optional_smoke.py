from __future__ import annotations

from importlib.metadata import version

import pytest
import torch

pytest.importorskip("tabpfn")

from bochan.models.classification.binary.foundation import (  # noqa: E402
    TabPFNBinaryClassificationModel,
    TabPFNMixedBinaryClassificationModel,
)
from bochan.models.classification.multiclass.foundation import (  # noqa: E402
    TabPFNMixedMulticlassClassificationModel,
    TabPFNMulticlassClassificationModel,
)
from bochan.models.regression.foundation import (  # noqa: E402
    TabPFNMixedRegressorModel,
    TabPFNRegressorModel,
)


def _mixed_X() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 10.0],
            [0.2, 20.0],
            [0.4, 30.0],
            [0.6, 10.0],
            [0.8, 20.0],
            [1.0, 30.0],
        ],
        dtype=torch.float32,
    )


def test_tabpfn_optional_dependency_is_supported_major_version() -> None:
    major, minor, *_ = (int(part) for part in version("tabpfn").split(".")[:2])
    assert (major, minor) >= (8, 2)
    assert major < 9


def test_real_tabpfn_regressor_constructors_accept_normal_and_mixed_contracts() -> None:
    X = _mixed_X()
    Y = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9], [1.1]], dtype=torch.float32)

    normal = TabPFNRegressorModel(
        train_X=X[:, :1],
        train_Y=Y,
        device="cpu",
        n_estimators=1,
    )
    mixed = TabPFNMixedRegressorModel(
        train_X=X,
        train_Y=Y,
        cat_dims=[1],
        device="cpu",
        n_estimators=1,
    )

    assert type(normal.estimator).__name__ == "TabPFNRegressor"
    assert type(mixed.estimator).__name__ == "TabPFNRegressor"
    assert normal.estimator.categorical_features_indices is None
    assert mixed.estimator.categorical_features_indices == [1]


def test_real_tabpfn_classifier_constructors_accept_binary_and_multiclass_contracts() -> None:
    X = _mixed_X()
    Y_binary = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    Y_multiclass = torch.tensor([[0], [1], [2], [0], [1], [2]], dtype=torch.long)

    binary = TabPFNBinaryClassificationModel(
        train_X=X[:, :1],
        train_Y=Y_binary,
        device="cpu",
        n_estimators=1,
    )
    mixed_binary = TabPFNMixedBinaryClassificationModel(
        train_X=X,
        train_Y=Y_binary,
        cat_dims=[1],
        device="cpu",
        n_estimators=1,
    )
    multiclass = TabPFNMulticlassClassificationModel(
        train_X=X[:, :1],
        train_Y=Y_multiclass,
        device="cpu",
        n_estimators=1,
    )
    mixed_multiclass = TabPFNMixedMulticlassClassificationModel(
        train_X=X,
        train_Y=Y_multiclass,
        cat_dims=[1],
        device="cpu",
        n_estimators=1,
    )

    assert type(binary.estimator).__name__ == "TabPFNClassifier"
    assert type(multiclass.estimator).__name__ == "TabPFNClassifier"
    assert binary.estimator.categorical_features_indices is None
    assert mixed_binary.estimator.categorical_features_indices == [1]
    assert multiclass.estimator.categorical_features_indices is None
    assert mixed_multiclass.estimator.categorical_features_indices == [1]
