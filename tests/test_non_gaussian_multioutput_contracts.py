"""Contracts for correlated and independent non-Gaussian multi-output models."""

from __future__ import annotations

import pytest
import torch
from botorch.posteriors.posterior_list import PosteriorList

from bochan.api.registry.model import MODEL_REGISTRY
from bochan.models.regression.beta.base import (
    BetaMultiTaskGPModel,
    KroneckerMultiTaskBetaGPModel,
    WideBetaMultiTaskGPModel,
)
from bochan.models.regression.gamma.base import GammaGPModel
from bochan.models.regression.multioutput import NonGaussianModelList


@pytest.mark.parametrize(
    ("family", "stem"),
    [
        ("beta", "Beta"),
        ("gamma", "Gamma"),
        ("poisson", "Poisson"),
        ("negative_binomial", "NegativeBinomial"),
    ],
)
def test_registry_separates_all_three_correlated_contracts(family: str, stem: str) -> None:
    """Both public task registries expose long, wide, and Kronecker classes."""
    expected = {
        f"{family}_multitask": f"{stem}MultiTaskGPModel",
        f"{family}_wide_multitask": f"Wide{stem}MultiTaskGPModel",
        f"{family}_kronecker": f"KroneckerMultiTask{stem}GPModel",
    }
    for task in ("regression", "multi_objective"):
        for key, class_name in expected.items():
            assert MODEL_REGISTRY["normal"][task][key].__name__ == class_name


def test_long_beta_contract_and_sparse_wide_conversion() -> None:
    """Long observations retain only observed cells and expose task correlation."""
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.double)
    Y = torch.tensor([0.2, 0.7, 0.4], dtype=torch.double)
    model = BetaMultiTaskGPModel(X, Y, task_feature=1, num_tasks=2, num_inducing_points=3)
    assert model.model.train_targets.numel() == 3
    assert model.observed_mask.sum() == 3
    assert model.posterior(torch.tensor([[0.5]], dtype=torch.double)).mean.shape == (1, 2)
    assert model.task_covar_matrix.shape == (2, 2)


@pytest.mark.parametrize("bad_task", [0.5, -1.0, 2.0])
def test_long_beta_rejects_invalid_task_ids(bad_task: float) -> None:
    """Task ids must be integer and within the declared range."""
    X = torch.tensor([[0.0, 0.0], [1.0, bad_task]], dtype=torch.double)
    with pytest.raises(ValueError):
        BetaMultiTaskGPModel(X, torch.tensor([0.2, 0.4]), task_feature=1, num_tasks=2)


def test_wide_and_kronecker_missing_target_contracts() -> None:
    """Wide models omit NaNs while Kronecker models reject incomplete blocks."""
    X = torch.linspace(0, 1, 3, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor([[0.2, 0.3], [0.4, torch.nan], [0.6, 0.7]], dtype=torch.double)
    wide = WideBetaMultiTaskGPModel(X, Y, num_inducing_points=4)
    assert wide.model.train_targets.numel() == 5
    with pytest.raises(ValueError, match="wide_multitask"):
        KroneckerMultiTaskBetaGPModel(X, Y)


def test_non_gaussian_model_list_preserves_native_posteriors() -> None:
    """Independent outputs remain a PosteriorList without a Gaussian proxy."""
    X = torch.linspace(0.1, 0.9, 4, dtype=torch.double).unsqueeze(-1)
    first = GammaGPModel(X, 1 + X)
    second = GammaGPModel(X, 2 + X)
    model = NonGaussianModelList(first, second)
    posterior = model.posterior(X[:2])
    assert isinstance(posterior, PosteriorList)
    assert model.num_outputs == 2
    assert model.subset_output([1]).num_outputs == 1
