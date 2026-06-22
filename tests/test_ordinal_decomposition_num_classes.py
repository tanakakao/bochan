from __future__ import annotations

import pytest
import torch

from bochan.models.ordinal.high_dim import (
    PCAOrdinalGPModel,
    PCAOrdinalMixedGPModel,
    REMBOOrdinalGPModel,
    REMBOOrdinalMixedGPModel,
)


DTYPE = torch.double
DEVICE = torch.device("cpu")
MODEL_CASES = (
    pytest.param(PCAOrdinalGPModel, False, id="pca"),
    pytest.param(REMBOOrdinalGPModel, False, id="rembo"),
    pytest.param(PCAOrdinalMixedGPModel, True, id="pca-mixed"),
    pytest.param(REMBOOrdinalMixedGPModel, True, id="rembo-mixed"),
)


def _train_data(n: int = 9, d: int = 3, *, mixed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    cols = [x]
    for j in range(1, d):
        cols.append((x + 0.17 * j).remainder(1.0))
    train_x = torch.cat(cols, dim=-1)
    if mixed:
        train_x[:, -1] = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE).repeat((n + 2) // 3)[:n]
    train_y = torch.tensor([0, 1, 2] * ((n + 2) // 3), dtype=torch.long, device=DEVICE)[:n]
    return train_x, train_y


def _model_kwargs(train_x: torch.Tensor, *, mixed: bool) -> dict:
    if not mixed:
        return {}
    return {"cat_dims": [train_x.shape[-1] - 1]}


@pytest.mark.parametrize(("model_cls", "mixed"), MODEL_CASES)
def test_projected_ordinal_model_infers_num_classes(model_cls, mixed: bool) -> None:
    train_x, train_y = _train_data(mixed=mixed)

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        n_components=2,
        inducing_points_num=4,
        **_model_kwargs(train_x, mixed=mixed),
    )

    assert model.num_classes == 3
    assert model.base_model.num_classes == 3


@pytest.mark.parametrize(("model_cls", "mixed"), MODEL_CASES)
def test_projected_ordinal_model_rejects_invalid_inferred_labels(model_cls, mixed: bool) -> None:
    train_x, _ = _train_data(n=3, mixed=mixed)
    train_y = torch.tensor([0, 2, 3], dtype=torch.long, device=DEVICE)

    with pytest.raises(ValueError, match="consecutive integers"):
        model_cls(
            train_X=train_x,
            train_Y=train_y,
            n_components=2,
            inducing_points_num=3,
            **_model_kwargs(train_x, mixed=mixed),
        )


@pytest.mark.parametrize(("model_cls", "mixed"), MODEL_CASES)
def test_projected_ordinal_model_keeps_explicit_num_classes(model_cls, mixed: bool) -> None:
    train_x, _ = _train_data(n=6, mixed=mixed)
    train_y = torch.tensor([0, 0, 2, 2, 0, 2], dtype=torch.long, device=DEVICE)

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        n_components=2,
        inducing_points_num=3,
        **_model_kwargs(train_x, mixed=mixed),
    )

    assert model.num_classes == 3
    assert model.base_model.num_classes == 3
