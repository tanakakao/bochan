"""Objective-role, direction, and acquisition-family helpers for the web workflow."""

from __future__ import annotations

from typing import Any

_WEB_TARGET_ROLES_KEY = "web_target_roles"


class _WebLevelSetThresholds(list[float]):
    """Thresholds that keep the Web app on the hybrid objective scale.

    The Web workflow fits ``HybridMultiOutputModel`` even for a single target so
    classification / ordinal settings can be expressed as scalar objective
    channels.  Contextual short-name resolution otherwise unwraps that model and
    selects task-specific level-set acquisitions, whose keyword contracts differ
    from the Web workflow (``threshold`` vs ``thresholds``, target classes,
    ordinal boundary indices, and so on).

    ``AcquisitionConfig`` recognizes the private resolver hook below and pins the
    regression level-set acquisition that operates directly on the Hybrid
    objective posterior.  This keeps multiclass class utilities and ordinal rank
    utilities defined by ``OutputSpec`` effective during level-set estimation.
    """

    _CLASS_NAMES = {
        "straddle": ("qRegressionStraddle", "qMultiOutputRegressionStraddle"),
        "boundaryvariance": (
            "qRegressionBoundaryVariance",
            "qMultiOutputRegressionBoundaryVariance",
        ),
        "icu": ("qRegressionICU", "qMultiOutputRegressionICU"),
    }

    @staticmethod
    def _normalize_name(name: str) -> str:
        return (
            str(name)
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
            .lower()
        )

    def _resolve_acqf_kwargs(
        self,
        *,
        name: str,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Any | None]:
        normalized = self._normalize_name(name)
        class_names = self._CLASS_NAMES.get(normalized)
        if class_names is None:
            return kwargs, None

        from bochan.acquisition.regression import levelset_estimation

        multi_output = len(self) > 1
        class_name = class_names[1 if multi_output else 0]
        acqf_cls = getattr(levelset_estimation, class_name)

        resolved = dict(kwargs)
        if multi_output:
            # Strip the list subclass after it has carried the resolver metadata.
            resolved["thresholds"] = [float(value) for value in self]
            return resolved, acqf_cls

        resolved["threshold"] = float(self[0])
        resolved.pop("thresholds", None)
        # These are multi-output-only settings.  In particular,
        # ``weighted_mean`` is not a valid single-output output_reduction value.
        resolved.pop("output_weights", None)
        resolved.pop("output_reduction", None)
        return resolved, acqf_cls


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(vars(value))


def apply_target_roles(
    target_settings: list[dict[str, Any]],
    model_kwargs: dict[str, Any],
    *,
    directions: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach optimization-role and direction metadata without changing legacy schemas."""

    cleaned_kwargs = dict(model_kwargs)
    raw_roles = cleaned_kwargs.pop(_WEB_TARGET_ROLES_KEY, {})
    roles = {
        str(target): _mapping(value)
        for target, value in dict(raw_roles or {}).items()
    }
    normalized: list[dict[str, Any]] = []
    for setting in target_settings:
        updated = dict(setting)
        target = str(updated["target"])
        role = roles.get(target, {})
        optimize = bool(role.get("optimize", updated.get("optimize", True)))
        direction = str(
            role.get(
                "direction",
                updated.get("direction", directions.get(target, "maximize")),
            )
        ).lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"{target}: direction must be maximize or minimize, got {direction!r}."
            )
        if updated.get("goal") == "target" and not optimize:
            raise ValueError(
                f"{target}: a target-value objective cannot be disabled. "
                "Use an above/below constraint for a constraint-only output."
            )
        updated["optimize"] = optimize
        updated["direction"] = direction
        normalized.append(updated)

    if not any(bool(setting["optimize"]) for setting in normalized):
        raise ValueError("At least one target must be selected as an optimization objective.")
    return normalized, cleaned_kwargs


def optimized_settings(target_settings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return settings participating in the acquisition objective."""

    return [setting for setting in target_settings if bool(setting.get("optimize", True))]


def optimized_targets(target_settings: list[dict[str, Any]]) -> list[str]:
    return [str(setting["target"]) for setting in optimized_settings(target_settings)]


def target_directions(target_settings: list[dict[str, Any]]) -> dict[str, str]:
    """Return the display direction for all modeled outputs."""

    return {
        str(setting["target"]): (
            "maximize"
            if setting.get("goal") == "target"
            else str(setting.get("direction", "maximize"))
        )
        for setting in target_settings
    }


def output_spec_kwargs(meta: dict[str, Any]) -> dict[str, Any]:
    """Map one target to the hybrid model's objective-space output definition."""

    task = str(meta["internal_task"])
    goal = str(meta["goal"])
    direction = str(meta.get("direction", "maximize"))
    sign = -1.0 if direction == "minimize" else 1.0

    if task == "regression":
        return {
            "sign": sign,
            "eq_target": (
                float(meta["configured_value"]) if goal == "target" else None
            ),
        }
    if task in {"binary", "multiclass"}:
        n_classes = int(meta["num_classes"])
        class_indices = [int(index) for index in meta.get("class_indices", [])]
        utilities = [
            1.0 if index in class_indices else 0.0
            for index in range(n_classes)
        ]
        return {
            "positive_class": class_indices[0],
            "utility_values": utilities,
            "sign": sign,
        }
    if task == "ordinal":
        n_classes = int(meta["num_classes"])
        if goal == "target":
            target_ranks = [int(index) for index in meta.get("class_indices", [])]
            utilities = [
                -min(abs(index - target_rank) for target_rank in target_ranks)
                for index in range(n_classes)
            ]
            sign = 1.0
        else:
            utilities = list(range(n_classes))
        return {"utility_values": utilities, "sign": sign}
    raise ValueError(f"Unsupported internal target task: {task}")


def objective_values_direct(
    train_y: Any,
    target_settings: list[dict[str, Any]],
) -> Any:
    """Transform direct multitask regression observations into objective space."""

    import torch

    columns = []
    for index, setting in enumerate(target_settings):
        values = train_y[:, index]
        if setting.get("goal") == "target":
            values = -torch.abs(values - float(setting["value"]))
        elif setting.get("direction") == "minimize":
            values = -values
        columns.append(values.unsqueeze(-1))
    return torch.cat(columns, dim=-1)


def select_optimized_values(
    values: Any,
    *,
    target_columns: list[str],
    objective_targets: list[str],
) -> Any:
    """Select objective channels while retaining all outputs in the fitted model."""

    indices = [target_columns.index(target) for target in objective_targets]
    return values[..., indices]


def objective_weights(
    *,
    target_columns: list[str],
    objective_targets: list[str],
) -> list[float]:
    """Return full-output weights for uncertainty and boundary acquisitions."""

    selected = set(objective_targets)
    return [1.0 if target in selected else 0.0 for target in target_columns]


def level_set_thresholds(
    *,
    target_columns: list[str],
    target_metadata: dict[str, dict[str, Any]],
    objective_targets: list[str],
) -> list[float]:
    """Return full-output thresholds in the hybrid model's objective space.

    Classification outputs are represented as the probability / expected utility
    of the class set selected in the Web UI.  Ordinal outputs use expected rank,
    or negative distance to selected target ranks for ``goal='target'``.  The
    returned structured list preserves that objective-space contract when
    ``AcquisitionConfig`` resolves the level-set acquisition class.
    """

    selected = set(objective_targets)
    thresholds: list[float] = []
    for target in target_columns:
        if target not in selected:
            thresholds.append(0.0)
            continue
        meta = target_metadata[target]
        goal = str(meta["goal"])
        if goal == "none":
            raise ValueError(
                f"{target}: level-set estimation requires above, below, or target."
            )
        if goal == "target":
            thresholds.append(0.0)
            continue
        task = str(meta["internal_task"])
        raw_threshold = (
            float(meta["class_index"])
            if task == "ordinal"
            else float(meta["configured_value"])
        )
        sign = -1.0 if meta.get("direction") == "minimize" else 1.0
        thresholds.append(sign * raw_threshold)
    return _WebLevelSetThresholds(thresholds)


def build_target_constraint_config(
    request: Any,
    *,
    target_settings: list[dict[str, Any]],
    target_metadata: dict[str, dict[str, Any]],
    target_columns: list[str],
    directions: dict[str, str],
    hybrid_model: bool,
) -> Any | None:
    """Build feasibility rules independently from optimization direction."""

    if all(bool(setting.get("legacy")) for setting in target_settings):
        from .target_settings import _build_outcome_constraint_config

        return _build_outcome_constraint_config(
            request,
            target_columns=target_columns,
            directions=directions,
        )

    from bochan.acquisition.feasible import FeasibilityConstraintSpec
    from bochan.api import OutcomeConstraintConfig

    specs: list[Any] = []
    for setting in target_settings:
        goal = str(setting["goal"])
        if goal not in {"above", "below"}:
            continue
        target = str(setting["target"])
        meta = target_metadata[target]
        task = str(meta["internal_task"])
        raw_threshold = (
            float(meta["class_index"])
            if task == "ordinal"
            else float(meta["configured_value"])
        )
        direction = str(meta.get("direction", "maximize"))
        sign = -1.0 if direction == "minimize" else 1.0
        threshold = sign * raw_threshold
        if sign > 0:
            sense = "ge" if goal == "above" else "le"
        else:
            sense = "le" if goal == "above" else "ge"
        output: Any = target if hybrid_model else target_columns.index(target)
        specs.append(
            FeasibilityConstraintSpec(
                output=output,
                threshold=threshold,
                sense=sense,
            )
        )
    return OutcomeConstraintConfig(constraints=specs) if specs else None


def best_observed(
    original_targets: Any,
    encoded_targets: Any,
    target_settings: list[dict[str, Any]],
    target_metadata: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Return target-wise observed summaries using objective direction."""

    values: dict[str, float] = {}
    for setting in target_settings:
        target = str(setting["target"])
        meta = target_metadata[target]
        goal = str(setting["goal"])
        direction = str(setting.get("direction", "maximize"))
        if meta["internal_task"] == "regression":
            series = original_targets[target]
            if goal == "target" and not setting.get("legacy"):
                target_value = float(meta["configured_value"])
                index = (series - target_value).abs().idxmin()
                values[target] = float(series.loc[index])
            elif direction == "minimize":
                values[target] = float(series.min())
            else:
                values[target] = float(series.max())
        elif meta["internal_task"] in {"binary", "multiclass"}:
            class_indices = [int(index) for index in meta.get("class_indices", [])]
            values[target] = float(encoded_targets[target].isin(class_indices).mean())
        else:
            series = encoded_targets[target]
            if goal == "target":
                target_indices = [int(index) for index in meta.get("class_indices", [])]
                values[target] = (
                    float(target_indices[0]) if target_indices else float(series.max())
                )
            elif direction == "minimize":
                values[target] = float(series.min())
            else:
                values[target] = float(series.max())
    return values


__all__ = [
    "apply_target_roles",
    "best_observed",
    "build_target_constraint_config",
    "level_set_thresholds",
    "objective_values_direct",
    "objective_weights",
    "optimized_settings",
    "optimized_targets",
    "output_spec_kwargs",
    "select_optimized_values",
    "target_directions",
]
