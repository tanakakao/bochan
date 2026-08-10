"""Regression tests for PCA model default dimensionality."""

import pytest
import torch

from bochan.models.classification.binary.high_dim import (
    PCABinaryClassificationGPModel,
    REMBOBinaryClassificationGPModel,
)
from bochan.models.classification.multiclass.high_dim import (
    PCAMulticlassClassificationGPModel,
    REMBOMulticlassClassificationGPModel,
)
from bochan.models.ordinal.high_dim import PCAOrdinalGPModel, REMBOOrdinalGPModel
from bochan.models.regression.gaussian.high_dim import PCAGaussianGPModel, REMBOGaussianGPModel
from bochan.models.regression.non_gaussian.beta.high_dim import PCABetaGPModel, REMBOBetaGPModel
from bochan.models.regression.non_gaussian.gamma.high_dim import PCAGammaGPModel, REMBOGammaGPModel
from bochan.models.regression.non_gaussian.negative_binomial.high_dim import (
    PCANegativeBinomialGPModel,
    REMBONegativeBinomialGPModel,
)
from bochan.models.regression.non_gaussian.poisson.high_dim import (
    PCAPoissonGPModel,
    REMBOPoissonGPModel,
)


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda X: PCAGaussianGPModel(X, torch.randn(8, 1, dtype=torch.double)),
        lambda X: PCABetaGPModel(X, torch.full((8, 1), 0.5, dtype=torch.double)),
        lambda X: PCAGammaGPModel(X, torch.ones(8, 1, dtype=torch.double)),
        lambda X: PCAPoissonGPModel(X, torch.ones(8, 1, dtype=torch.double)),
        lambda X: PCANegativeBinomialGPModel(X, torch.ones(8, 1, dtype=torch.double)),
        lambda X: PCABinaryClassificationGPModel(
            X, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.double)
        ),
        lambda X: PCAMulticlassClassificationGPModel(
            X, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        ),
        lambda X: PCAOrdinalGPModel(X, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])),
    ],
)
def test_pca_models_default_to_two_components(model_factory):
    """Verify every PCA model family projects to two components by default.

    Args:
        model_factory: Callable that constructs a PCA model without a dimension
            argument.
    """
    train_X = torch.randn(8, 3, dtype=torch.double)

    assert model_factory(train_X)._projected_train_X.shape[-1] == 2


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda X: REMBOGaussianGPModel(X, torch.randn(8, 1, dtype=torch.double)),
        lambda X: REMBOBetaGPModel(X, torch.full((8, 1), 0.5, dtype=torch.double)),
        lambda X: REMBOGammaGPModel(X, torch.ones(8, 1, dtype=torch.double)),
        lambda X: REMBOPoissonGPModel(X, torch.ones(8, 1, dtype=torch.double)),
        lambda X: REMBONegativeBinomialGPModel(X, torch.ones(8, 1, dtype=torch.double)),
        lambda X: REMBOBinaryClassificationGPModel(
            X, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.double)
        ),
        lambda X: REMBOMulticlassClassificationGPModel(
            X, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        ),
        lambda X: REMBOOrdinalGPModel(X, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])),
    ],
)
def test_rembo_models_default_to_two_components(model_factory):
    """Verify every REMBO model family projects to two components by default.

    Args:
        model_factory: Callable that constructs a REMBO model without a dimension
            argument.
    """
    train_X = torch.randn(8, 3, dtype=torch.double)

    assert model_factory(train_X)._projected_train_X.shape[-1] == 2
