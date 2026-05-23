
from __future__ import annotations

"""
Multi-output regression Bayesian Optimization.

通常の multi-output / multi-objective regression では BoTorch 標準の
qEHVI / qNEHVI / qNParEGO 系を使う前提です。

このファイルでは再実装せず、プロジェクト内で family 名を揃えるための
import surface だけを提供します。
"""

from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement as qMultiOutputRegressionExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement as qMultiOutputRegressionNoisyExpectedHypervolumeImprovement,
)

try:
    from botorch.acquisition.multi_objective.monte_carlo import (
        qLogExpectedHypervolumeImprovement as qMultiOutputRegressionLogExpectedHypervolumeImprovement,
        qLogNoisyExpectedHypervolumeImprovement as qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement,
    )
except Exception:  # pragma: no cover
    qMultiOutputRegressionLogExpectedHypervolumeImprovement = None
    qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement = None


