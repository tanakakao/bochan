from __future__ import annotations

import inspect

import pytest
import torch

from bochan.models.ordinal.robust import (
    HeteroscedasticOrdinalGPModel,
    HeteroscedasticOrdinalMixedGPModel,
    RobustRelevancePursuitOrdinalGPModel,
    RobustRelevancePursuitOrdinalMixedGPModel,
)
from bochan.models.ordinal.robust import heteroscedastic as heteroscedastic_module
from bochan.models.ordinal.robust import relevance_pursuit as relevance_pursuit_module


DTYPE = torch.double
DEVICE = torch.device("cpu")
MODEL_CASES = (
    pytest.param(RobustRelevancePursuitOrdinalGPModel, False, False, id="rrp"),
    pytest.param(RobustRelevancePursuitOrdinalMixedGPModel, True, False, id="rrp-mixed"),
    pytest.param(HeteroscedasticOrdinalGPModel, False, True, id="hetero"),
    pytest.param(HeteroscedasticOrdinalMixedGPModel, True, True, id="hetero-mixed"),
)


class _DummyNoiseModel(torch.nn.Module):
    """Minimal registered noise model for constructor-only tests."""


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


def _model_kwargs(
    train_x: torch.Tensor,
    *,
    mixed: bool,
    heteroscedastic: bool,
) -> dict:
    kwargs = {"num_inducing": min(4, train_x.shape[-2])}
    if mixed:
        kwargs["cat_dims"] = [train_x.shape[-1] - 1]
    if heteroscedastic:
        kwargs["train_Yvar"] = torch.full(
            (train_x.shape[-2], 1),
            0.1,
            dtype=train_x.dtype,
            device=train_x.device,
        )
    return kwargs


@pytest.fixture(autouse=True)
def _mock_heteroscedastic_noise_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid exact-GP optimization while retaining constructor behavior."""

    def _fit_noise_model(**kwargs):
        return _DummyNoiseModel()

    def _predict_noise_var(*, ref_like: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full_like(ref_like, 0.1)

    monkeypatch.setattr(
        heteroscedastic_module,
        "fit_noise_model_single",
        _fit_noise_model,
    )
    monkeypatch.setattr(
        heteroscedastic_module,
        "fit_noise_model_mixed",
        _fit_noise_model,
    )
    monkeypatch.setattr(
        heteroscedastic_module,
        "predict_noise_var_from_log_noise_model",
        _predict_noise_var,
    )


@pytest.mark.parametrize(("model_cls", "mixed", "heteroscedastic"), MODEL_CASES)
def test_robust_ordinal_num_classes_defaults_to_none(
    model_cls,
    mixed: bool,
    heteroscedastic: bool,
) -> None:
    del mixed, heteroscedastic
    signature = inspect.signature(model_cls)

    assert signature.parameters["num_classes"].default is None


@pytest.mark.parametrize(("model_cls", "mixed", "heteroscedastic"), MODEL_CASES)
def test_robust_ordinal_model_infers_num_classes(
    model_cls,
    mixed: bool,
    heteroscedastic: bool,
) -> None:
    train_x, train_y = _train_data(mixed=mixed)

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        **_model_kwargs(
            train_x,
            mixed=mixed,
            heteroscedastic=heteroscedastic,
        ),
    )

    assert model.num_classes == 3
    assert model.likelihood.num_classes == 3


@pytest.mark.parametrize(("model_cls", "mixed", "heteroscedastic"), MODEL_CASES)
def test_robust_ordinal_model_rejects_invalid_inferred_labels(
    model_cls,
    mixed: bool,
    heteroscedastic: bool,
) -> None:
    train_x, _ = _train_data(n=3, mixed=mixed)
    train_y = torch.tensor([0, 2, 3], dtype=torch.long, device=DEVICE)

    with pytest.raises(ValueError, match="consecutive integers"):
        model_cls(
            train_X=train_x,
            train_Y=train_y,
            **_model_kwargs(
                train_x,
                mixed=mixed,
                heteroscedastic=heteroscedastic,
            ),
        )


@pytest.mark.parametrize(("model_cls", "mixed", "heteroscedastic"), MODEL_CASES)
def test_robust_ordinal_model_keeps_explicit_num_classes(
    model_cls,
    mixed: bool,
    heteroscedastic: bool,
) -> None:
    train_x, _ = _train_data(n=6, mixed=mixed)
    train_y = torch.tensor([0, 0, 2, 2, 0, 2], dtype=torch.long, device=DEVICE)

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        **_model_kwargs(
            train_x,
            mixed=mixed,
            heteroscedastic=heteroscedastic,
        ),
    )

    assert model.num_classes == 3
    assert model.likelihood.num_classes == 3


def test_direct_robust_module_imports_use_public_inference_classes() -> None:
    assert (
        relevance_pursuit_module.RobustRelevancePursuitOrdinalGPModel
        is RobustRelevancePursuitOrdinalGPModel
    )
    assert (
        relevance_pursuit_module.RobustRelevancePursuitOrdinalMixedGPModel
        is RobustRelevancePursuitOrdinalMixedGPModel
    )
    assert (
        heteroscedastic_module.HeteroscedasticOrdinalGPModel
        is HeteroscedasticOrdinalGPModel
    )
    assert (
        heteroscedastic_module.HeteroscedasticOrdinalMixedGPModel
        is HeteroscedasticOrdinalMixedGPModel
    )
