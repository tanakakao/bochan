from __future__ import annotations

import pytest
import torch

from bochan.models.ordinal.high_dim import PCAOrdinalGPModel, REMBOOrdinalGPModel


DTYPE = torch.double
DEVICE = torch.device("cpu")
MODEL_CLASSES = (PCAOrdinalGPModel, REMBOOrdinalGPModel)


def _train_data(n: int = 9, d: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.linspace(0.0, 1.0, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    cols = [x]
    for j in range(1, d):
        cols.append((x + 0.17 * j).remainder(1.0))
    train_x = torch.cat(cols, dim=-1)
    train_y = torch.tensor([0, 1, 2] * ((n + 2) // 3), dtype=torch.long, device=DEVICE)[:n]
    return train_x, train_y


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_projected_ordinal_model_infers_num_classes(model_cls) -> None:
    train_x, train_y = _train_data()

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        n_components=2,
        inducing_points_num=4,
    )

    assert model.num_classes == 3
    assert model.base_model.num_classes == 3


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_projected_ordinal_model_rejects_invalid_inferred_labels(model_cls) -> None:
    train_x, _ = _train_data(n=3)
    train_y = torch.tensor([0, 2, 3], dtype=torch.long, device=DEVICE)

    with pytest.raises(ValueError, match="consecutive integers"):
        model_cls(
            train_X=train_x,
            train_Y=train_y,
            n_components=2,
            inducing_points_num=3,
        )


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_projected_ordinal_model_keeps_explicit_num_classes(model_cls) -> None:
    train_x, _ = _train_data(n=6)
    train_y = torch.tensor([0, 0, 2, 2, 0, 2], dtype=torch.long, device=DEVICE)

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        n_components=2,
        inducing_points_num=3,
    )

    assert model.num_classes == 3
    assert model.base_model.num_classes == 3
