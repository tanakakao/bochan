"""Compatibility exports for non-Gaussian regression GP models.

This package keeps the historical import path
``bochan.models.regression.non_gaussian`` available while the concrete model
implementations live under distribution-specific packages such as
``bochan.models.regression.poisson.non_gaussian``.
"""

from .poisson import PoissonGPModel, PoissonMixedGPModel, PoissonLogLikelihood, PoissonPosterior
from .beta import BetaGPModel, BetaMixedGPModel, BetaLogLikelihood, BetaPosterior
from .gamma import GammaGPModel, GammaMixedGPModel, GammaLogLikelihood, GammaPosterior
from .negative_binomial import (
    NegativeBinomialGPModel,
    NegativeBinomialMixedGPModel,
    NegativeBinomialLogLikelihood,
    NegativeBinomialPosterior,
)

__all__ = [
    "PoissonGPModel",
    "PoissonMixedGPModel",
    "PoissonLogLikelihood",
    "PoissonPosterior",
    "BetaGPModel",
    "BetaMixedGPModel",
    "BetaLogLikelihood",
    "BetaPosterior",
    "GammaGPModel",
    "GammaMixedGPModel",
    "GammaLogLikelihood",
    "GammaPosterior",
    "NegativeBinomialGPModel",
    "NegativeBinomialMixedGPModel",
    "NegativeBinomialLogLikelihood",
    "NegativeBinomialPosterior",
]
