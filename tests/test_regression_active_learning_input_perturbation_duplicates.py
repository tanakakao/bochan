from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.models.model import Model
from torch import Tensor

from bochan.acquisition.regression.active_learning import qRegressionPosteriorVariance


class _RepeatedEvaluationModel(Model):
    """Minimal model exposing q*n_w evaluation rows for each raw candidate."""

    def __init__(self, n_w: int = 4) -> None:
        super().__init__()
        self.n_w = int(n_w)

    @property
    def num_outputs(self) -> int:
        return 1

    def transform_inputs(self, X: Tensor, input_transform=None) -> Tensor:
        del input_transform
        return X.repeat_interleave(self.n_w, dim=-2)

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs,
    ):
        del output_indices, observation_noise, posterior_transform, kwargs
        Xt = self.transform_inputs(X)
        mean = torch.zeros(*Xt.shape[:-1], 1, dtype=X.dtype, device=X.device)
        variance = torch.ones_like(mean)
        return SimpleNamespace(mean=mean, variance=variance)


def _acquisition(**kwargs) -> qRegressionPosteriorVariance:
    return qRegressionPosteriorVariance(
        model=_RepeatedEvaluationModel(n_w=4),
        n_w=4,
        **kwargs,
    )


def test_perturbation_replicas_are_not_hard_same_batch_duplicates() -> None:
    """q*n_w evaluation replicas must not invalidate distinct raw candidates."""

    acqf = _acquisition()
    X = torch.tensor(
        [
            [[0.2], [0.8]],
            [[0.1], [0.9]],
        ],
        dtype=torch.double,
    )

    value = acqf(X)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()
    torch.testing.assert_close(value, torch.ones_like(value))


def test_raw_same_batch_duplicates_remain_hard_excluded() -> None:
    """The fix must preserve duplicate exclusion for nominal q candidates."""

    acqf = _acquisition()
    X = torch.tensor([[[0.5], [0.5]]], dtype=torch.double)

    value = acqf(X)

    assert torch.isneginf(value).all()


def test_raw_pending_duplicate_remains_hard_excluded() -> None:
    """Pending exclusion must compare raw candidates, not perturbation replicas."""

    acqf = _acquisition()
    acqf.set_X_pending(torch.tensor([[0.5]], dtype=torch.double))

    duplicate = acqf(torch.tensor([[[0.5]]], dtype=torch.double))
    distinct = acqf(torch.tensor([[[0.6]]], dtype=torch.double))

    assert torch.isneginf(duplicate).all()
    assert torch.isfinite(distinct).all()
