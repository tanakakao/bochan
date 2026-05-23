from __future__ import annotations

"""
Single-output regression Bayesian optimization acquisitions.

通常の single-output regression BO は BoTorch 標準 acquisition を使う前提です。
このファイルでは再実装しません。

Recommended BoTorch classes:
    - qExpectedImprovement
    - qLogExpectedImprovement
    - qNoisyExpectedImprovement
    - qProbabilityOfImprovement
    - qUpperConfidenceBound
    - qKnowledgeGradient
"""

__all__: list[str] = []
