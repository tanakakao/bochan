from __future__ import annotations

import inspect

import pytest
import torch

from bochan.models.classification.multiclass.high_dim import (
    PCAMulticlassClassificationGPModel,
    PCAMulticlassClassificationMixedGPModel,
    REMBOMulticlassClassificationGPModel,
    REMBOMulticlassClassificationMixedGPModel,
)


DTYPE = torch.double
DEVICE = torch.device("cpu")
MODEL_CASES = (
    pytest.param(PCAMulticlassClassificationGPModel, False, id="pca"),
    pytest.param(REMBOMulticlassClassificationGPModel, False, id="rembo"),
    pytest.param(PCAMulticlassClassificationMixedGPModel, True, id="pca-mixed"),
    pytest.param(REMBOMulticlassClassificationMixedGPModel, True, id="rembo-mixed"),
)


def _train_data(
    n: int = 9,
    d: int = 3,
    *,
    mixed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    columns = [x]
    for index in range(1, d):
        columns.append((x + 0.17 * index).remainder(1.0))

    train_x = torch.cat(columns, dim=-1)
    if mixed:
        categories = torch.tensor(
            [0.0, 1.0, 2.0],
            dtype=DTYPE,
            device=DEVICE,
        )
        train_x[:, -1] = categories.repeat((n + 2) // 3)[:n]

    train_y = torch.tensor(
        [0, 1, 2] * ((n + 2) // 3),
        dtype=torch.long,
        device=DEVICE,
    )[:n]
    return train_x, train_y


def _model_kwargs(train_x: torch.Tensor, *, mixed: bool) -> dict:
    kwargs = {
        "n_components": 2,
        "num_inducing_points": min(4, train_x.shape[-2]),
    }
    if mixed:
        kwargs["cat_dims"] = [train_x.shape[-1] - 1]
    return kwargs


@pytest.mark.parametrize(("model_cls", "mixed"), MODEL_CASES)
def test_projected_multiclass_num_classes_defaults_to_none(
    model_cls,
    mixed: bool,
) -> None:
    del mixed
    signature = inspect.signature(model_cls)

    assert signature.parameters["num_classes"].default is None


@pytest.mark.parametrize(("model_cls", "mixed"), MODEL_CASES)
def test_projected_multiclass_model_infers_num_classes(
    model_cls,
    mixed: bool,
) -> None:
    train_x, train_y = _train_data(mixed=mixed)

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        **_model_kwargs(train_x, mixed=mixed),
    )

    assert model.num_classes == 3
    assert model.num_outputs == 3
    assert model.base_model.num_classes == 3
    assert model.base_model.likelihood.num_classes == 3


@pytest.mark.parametrize(("model_cls", "mixed"), MODEL_CASES)
def test_projected_multiclass_model_keeps_explicit_num_classes(
    model_cls,
    mixed: bool,
) -> None:
    train_x, _ = _train_data(n=6, mixed=mixed)
    train_y = torch.tensor(
        [0, 0, 2, 2, 0, 2],
        dtype=torch.long,
        device=DEVICE,
    )

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        num_classes=4,
        **_model_kwargs(train_x, mixed=mixed),
    )

    assert model.num_classes == 4
    assert model.num_outputs == 4
    assert model.base_model.num_classes == 4
    assert model.base_model.likelihood.num_classes == 4
