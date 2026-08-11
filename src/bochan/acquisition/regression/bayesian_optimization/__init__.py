"""Regression Bayesian optimization acquisitions.

Standard single-output regression BO uses BoTorch directly. Custom classes in
this package cover multi-objective and heteroscedastic extensions.
"""

from .hetero_multi_output import (
    qHeteroMultiOutputRegressionDecoupledExpectedHypervolumeImprovement,
    qHeteroMultiOutputRegressionNParEGO,
)
from .hetero_multi_output_autograd import (
    qHeteroMultiOutputRegressionExpectedHypervolumeImprovement,
    qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement,
)
from .hetero_single_output import (
    qHeteroRegressionExpectedImprovement,
    qHeteroRegressionProbabilityOfImprovement,
    qHeteroRegressionUpperConfidenceBound,
)
from .multi_output import (
    qMultiOutputRegressionExpectedHypervolumeImprovement,
    qMultiOutputRegressionLogExpectedHypervolumeImprovement,
    qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement,
    qMultiOutputRegressionNoisyExpectedHypervolumeImprovement,
    qMultiOutputRegressionNParEGO,
)


class qRegressionExpectedHypervolumeImprovement(
    qMultiOutputRegressionExpectedHypervolumeImprovement
):
    """Regression qEHVI with domain-first naming."""


class qRegressionNoisyExpectedHypervolumeImprovement(
    qMultiOutputRegressionNoisyExpectedHypervolumeImprovement
):
    """Regression qNEHVI with domain-first naming."""


if qMultiOutputRegressionLogExpectedHypervolumeImprovement is not None:

    class qRegressionLogExpectedHypervolumeImprovement(
        qMultiOutputRegressionLogExpectedHypervolumeImprovement
    ):
        """Regression qLogEHVI with domain-first naming."""

else:
    qRegressionLogExpectedHypervolumeImprovement = None


if qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement is not None:

    class qRegressionLogNoisyExpectedHypervolumeImprovement(
        qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement
    ):
        """Regression qLogNEHVI with domain-first naming."""

else:
    qRegressionLogNoisyExpectedHypervolumeImprovement = None


class qRegressionNParEGO(qMultiOutputRegressionNParEGO):
    """Regression NParEGO with domain-first naming."""


class qHeteroRegressionExpectedHypervolumeImprovement(
    qHeteroMultiOutputRegressionExpectedHypervolumeImprovement
):
    """Heteroscedastic regression qEHVI with domain-first naming."""


class qHeteroRegressionNoisyExpectedHypervolumeImprovement(
    qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement
):
    """Heteroscedastic regression qNEHVI with domain-first naming."""


class qHeteroRegressionNParEGO(qHeteroMultiOutputRegressionNParEGO):
    """Heteroscedastic regression NParEGO with domain-first naming."""


class qHeteroRegressionDecoupledExpectedHypervolumeImprovement(
    qHeteroMultiOutputRegressionDecoupledExpectedHypervolumeImprovement
):
    """Decoupled heteroscedastic regression qEHVI with domain-first naming."""


__all__ = [
    "qHeteroRegressionDecoupledExpectedHypervolumeImprovement",
    "qHeteroRegressionExpectedHypervolumeImprovement",
    "qHeteroRegressionExpectedImprovement",
    "qHeteroRegressionNParEGO",
    "qHeteroRegressionNoisyExpectedHypervolumeImprovement",
    "qHeteroRegressionProbabilityOfImprovement",
    "qHeteroRegressionUpperConfidenceBound",
    "qRegressionExpectedHypervolumeImprovement",
    "qRegressionLogExpectedHypervolumeImprovement",
    "qRegressionLogNoisyExpectedHypervolumeImprovement",
    "qRegressionNParEGO",
    "qRegressionNoisyExpectedHypervolumeImprovement",
]
