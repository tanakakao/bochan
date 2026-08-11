from __future__ import annotations

import inspect

import pytest
import torch

from bochan.models.ordinal.deep import (
    DeepKernelOrdinalGPModel,
    DeepKernelOrdinalMixedGPModel,
    DeepOrdinalGPModel,
    DeepOrdinalMixedGPModel,
)


DEEP_ORDINAL_CLASSES = [
    DeepOrdinalGPModel,
    DeepOrdinalMixedGPModel,
    DeepKernelOrdinalGPModel,
    DeepKernelOrdinalMixedGPModel,
]


@pytest.mark.parametrize("model_cls", DEEP_ORDINAL_CLASSES)
def test_num_classes_is_optional_in_public_signature(model_cls) -> None:
    parameter = inspect.signature(model_cls.__init__).parameters["num_classes"]

    assert parameter.default is None


def test_ordinal_deepgp_infers_num_classes_from_train_y() -> None:
    train_X = torch.rand(6, 2, dtype=torch.double)
    train_Y = torch.tensor([[0], [1], [2], [1], [0], [2]])

    model = DeepOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=None,
        hidden_dims=[2],
        num_inducing=4,
    )

    assert model.num_classes == 3
    assert model.ordinal_likelihood.num_classes == 3


def test_ordinal_deepkernel_infers_num_classes_from_train_y() -> None:
    train_X = torch.rand(6, 2, dtype=torch.double)
    train_Y = torch.tensor([0, 1, 2, 1, 0, 2])

    model = DeepKernelOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=None,
        hidden_dims=[4, 2],
        num_inducing=4,
        input_transform=None,
    )

    assert model.num_classes == 3
    assert model.ordinal_likelihood.num_classes == 3


def test_explicit_num_classes_is_preserved() -> None:
    train_X = torch.rand(5, 2, dtype=torch.double)
    train_Y = torch.tensor([0, 1, 2, 1, 0])

    model = DeepOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=4,
        hidden_dims=[2],
        num_inducing=4,
    )

    assert model.num_classes == 4
