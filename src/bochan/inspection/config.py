"""Configuration for raw-space feature inspection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

PredictiveImportanceMethod = Literal["permutation"]
DiagnosticMethod = Literal[
    "auto",
    "ard",
    "kernel_components",
    "saas",
    "pca",
    "rembo",
    "vae",
    "deepkernel",
    "deepgp",
    "heteroscedastic",
    "rrp",
    "multitask",
    "multifidelity",
]

_DIAGNOSTICS = {
    "auto",
    "ard",
    "kernel_components",
    "saas",
    "pca",
    "rembo",
    "vae",
    "deepkernel",
    "deepgp",
    "heteroscedastic",
    "rrp",
    "multitask",
    "multifidelity",
}


@dataclass(frozen=True)
class FeatureGroup:
    """Columns that must be permuted with one common row permutation."""

    name: str
    indices: tuple[int, ...]
    role: str = "group"


@dataclass
class FeatureImportanceConfig:
    """Configure predictive importance and lightweight model diagnostics."""

    predictive_methods: Sequence[PredictiveImportanceMethod] = field(default_factory=lambda: ["permutation"])
    diagnostic_methods: Sequence[DiagnosticMethod] = field(default_factory=lambda: ["auto"])
    n_repeats: int = 10
    random_state: int | None = 0
    scoring: str | Any = "auto"
    scoring_direction: Literal["auto", "minimize", "maximize"] = "auto"
    compute_noise_importance: bool = True
    compute_classwise_importance: bool = False
    normalize_importance: bool = False
    clip_negative_importance: bool = False
    feature_groups: Sequence[FeatureGroup] | None = None
    feature_roles: dict[int | str, str] | None = None
    return_per_repeat_values: bool = True
    batch_size: int | None = None
    unsupported_method_policy: Literal["raise", "warn", "skip"] = "warn"
    error_policy: Literal["raise", "warn", "skip"] = "warn"

    def __post_init__(self) -> None:
        """Reject invalid settings before prediction begins."""
        if not self.predictive_methods:
            raise ValueError("predictive_methods must not be empty.")
        unsupported = set(self.predictive_methods) - {"permutation"}
        if unsupported:
            raise ValueError(f"Unsupported predictive importance method(s): {sorted(unsupported)}")
        invalid = set(self.diagnostic_methods) - _DIAGNOSTICS
        if invalid:
            raise ValueError(f"Unsupported diagnostic method(s): {sorted(invalid)}")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive when provided.")
        if self.scoring_direction not in {"auto", "minimize", "maximize"}:
            raise ValueError("scoring_direction must be auto, minimize, or maximize.")
        if self.unsupported_method_policy not in {"raise", "warn", "skip"}:
            raise ValueError("unsupported_method_policy must be raise, warn, or skip.")
        if self.error_policy not in {"raise", "warn", "skip"}:
            raise ValueError("error_policy must be raise, warn, or skip.")
        for group in self.feature_groups or ():
            if not group.indices or len(group.indices) != len(set(group.indices)):
                raise ValueError(f"Feature group {group.name!r} must contain unique indices.")
