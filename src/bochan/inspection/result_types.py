"""Serializable result objects for feature inspection."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from torch import Tensor


def _json_value(value: Any) -> Any:
    """Convert nested inspection data to JSON-safe Python values."""
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


@dataclass
class ImportanceSummary:
    """Repeat-level distribution of one raw-space importance."""

    values: Tensor | list[float] | None
    mean: float
    std: float
    minimum: float
    maximum: float
    median: float
    normalized_mean: float | None = None
    rank: int | None = None


@dataclass
class FeatureImportanceEntry:
    """Importance and provenance for a feature or feature group."""

    name: str
    indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_type: str
    role: str
    importance: ImportanceSummary
    baseline_metric: float
    metric_name: str
    metric_direction: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportanceMethodResult:
    """All entries produced by one predictive method."""

    method: str
    entries: dict[str, FeatureImportanceEntry]
    baseline_metrics: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputFeatureImportanceResult:
    """Predictive importance and diagnostics for one output."""

    output_name: str
    task_type: str
    predictive_methods: dict[str, ImportanceMethodResult]
    noise_methods: dict[str, ImportanceMethodResult] = field(default_factory=dict)
    classwise_methods: dict[Any, dict[str, ImportanceMethodResult]] | None = None
    model_diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureImportanceResult:
    """Output-oriented feature inspection result."""

    outputs: dict[str, OutputFeatureImportanceResult]
    predictive_methods: tuple[str, ...]
    diagnostic_methods: tuple[str, ...]
    evaluation_space: str
    n_repeats: int
    feature_names: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output(self) -> OutputFeatureImportanceResult:
        """Return the sole output, rejecting ambiguous multi-output access."""
        if len(self.outputs) != 1:
            raise RuntimeError("output is available only for a single-output result.")
        return next(iter(self.outputs.values()))

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-safe representation."""
        return _json_value(asdict(self))


@dataclass
class CrossValidatedImportanceSummary:
    """Between-fold distribution, separate from within-fold repeats."""

    fold_values: Tensor | list[float]
    mean: float
    std: float
    minimum: float
    maximum: float
    median: float
    mean_rank: float
    rank_std: float
    valid_fold_count: int
    within_fold_repeat_std: list[float] = field(default_factory=list)


@dataclass
class CrossValidatedMethodResult:
    """Fold aggregation for one importance method."""

    method: str
    entries: dict[str, CrossValidatedImportanceSummary]
    fold_results: list[ImportanceMethodResult]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossValidatedOutputImportance:
    """Cross-validated importance for one output."""

    output_name: str
    task_type: str
    predictive_methods: dict[str, CrossValidatedMethodResult]
    noise_methods: dict[str, CrossValidatedMethodResult]
    fold_diagnostics: list[dict[str, Any]]
    diagnostic_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CrossValidatedFeatureImportanceResult:
    """Output-oriented aggregation of validation-fold importances."""

    outputs: dict[str, CrossValidatedOutputImportance]
    feature_names: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-safe representation."""
        return _json_value(asdict(self))
