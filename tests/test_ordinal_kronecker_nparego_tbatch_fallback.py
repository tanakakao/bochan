from __future__ import annotations

import importlib

import pytest
import torch


bo_module = importlib.import_module(
    "bochan.acquisition.ordinal.bayesian_optimization"
)


class _DummyTBatchSafeNParEGO(
    bo_module._TBatchSafeMultiOutputOrdinalNParEGO
):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def _evaluate_single_tbatch(self, X_single: torch.Tensor) -> torch.Tensor:
        # A differentiable scalar unique to each optimizer row.
        return X_single.sum()


def test_collapsed_kronecker_tbatch_is_evaluated_per_optimizer_row(
    monkeypatch,
) -> None:
    def collapsed_forward(self, X):
        raise RuntimeError(
            bo_module._NPAREGO_TBATCH_SHAPE_ERROR
            + " value.shape=(128,), q=1, batch_shape=(2,)."
        )

    monkeypatch.setattr(
        bo_module._qMultiOutputOrdinalNParEGO,
        "forward",
        collapsed_forward,
    )

    X = torch.arange(10, dtype=torch.double).reshape(2, 1, 5)
    X.requires_grad_(True)
    acquisition = _DummyTBatchSafeNParEGO()

    result = acquisition.forward(X)

    assert result.shape == torch.Size([2])
    torch.testing.assert_close(result, X.sum(dim=(-2, -1)))

    result.sum().backward()
    torch.testing.assert_close(X.grad, torch.ones_like(X))


def test_multidimensional_tbatch_shape_is_restored(monkeypatch) -> None:
    def collapsed_forward(self, X):
        raise RuntimeError(bo_module._NPAREGO_TBATCH_SHAPE_ERROR)

    monkeypatch.setattr(
        bo_module._qMultiOutputOrdinalNParEGO,
        "forward",
        collapsed_forward,
    )

    X = torch.arange(30, dtype=torch.double).reshape(2, 3, 1, 5)
    acquisition = _DummyTBatchSafeNParEGO()

    result = acquisition.forward(X)

    assert result.shape == torch.Size([2, 3])
    torch.testing.assert_close(result, X.sum(dim=(-2, -1)))


def test_unrelated_runtime_error_is_not_suppressed(monkeypatch) -> None:
    def unrelated_forward(self, X):
        raise RuntimeError("posterior covariance is invalid")

    monkeypatch.setattr(
        bo_module._qMultiOutputOrdinalNParEGO,
        "forward",
        unrelated_forward,
    )

    acquisition = _DummyTBatchSafeNParEGO()

    with pytest.raises(RuntimeError, match="posterior covariance is invalid"):
        acquisition.forward(torch.zeros(2, 1, 5, dtype=torch.double))


def test_shape_error_without_tbatch_is_not_suppressed(monkeypatch) -> None:
    def collapsed_forward(self, X):
        raise RuntimeError(bo_module._NPAREGO_TBATCH_SHAPE_ERROR)

    monkeypatch.setattr(
        bo_module._qMultiOutputOrdinalNParEGO,
        "forward",
        collapsed_forward,
    )

    acquisition = _DummyTBatchSafeNParEGO()

    with pytest.raises(RuntimeError, match="Expected scalarized NParEGO"):
        acquisition.forward(torch.zeros(1, 5, dtype=torch.double))
