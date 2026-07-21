"""Target-setting normalization and heterogeneous model configuration helpers."""

from __future__ import annotations

from typing import Any

_WEB_TARGET_SETTINGS_KEY = "web_target_settings"


def _mapping(value: Any) -> dict[str, Any]:
    """Convert Pydantic models, namespaces, and mappings to a plain dict."""

    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(vars(value))


def _list_values(value: Any) -> list[Any]:
    """Normalize an optional scalar or sequence to a list."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _resolve_targets(request: Any) -> tuple[list[str], dict[str, str]]:
    """Resolve backward-compatible target and direction settings."""

    target_columns = [str(value) for value in list(request.target_columns or [])]
    if not target_columns and request.target_column:
        target_columns = [str(request.target_column)]
    if not target_columns:
        raise ValueError("At least one target column is required.")
    if len(set(target_columns)) != len(target_columns):
        raise ValueError("target_columns must not contain duplicates.")

    requested_directions = dict(request.directions or {})
    directions = {
        target: str(requested_directions.get(target, request.direction or "maximize"))
        for target in target_columns
    }
    invalid = {
        target: direction
        for target, direction in directions.items()
        if direction not in {"maximize", "minimize"}
    }
    if invalid:
        raise ValueError(f"Invalid target directions: {invalid}")
    return target_columns, directions


def _resolve_target_settings(
    request: Any,
    *,
    target_columns: list[str],
    directions: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve one task definition and an optional constraint per target."""

    model_kwargs = dict(request.model_kwargs or {})
    raw_settings = model_kwargs.pop(_WEB_TARGET_SETTINGS_KEY, None)
    if raw_settings is None:
        settings = [
            {
                "target": target,
                "task_type": "regression",
                "goal": "below" if directions[target] == "minimize" else "above",
                "value": None,
                "target_class": None,
                "target_classes": [],
                "class_order": [],
                "target_values": [],
                "legacy": True,
            }
            for target in target_columns
        ]
        return settings, model_kwargs

    settings = [_mapping(value) for value in list(raw_settings)]
    if len(settings) != len(target_columns):
        raise ValueError(
            "Exactly one task setting is required for every target column. "
            f"Expected {len(target_columns)}, got {len(settings)}."
        )
    setting_targets = [str(setting.get("target", "")) for setting in settings]
    if len(set(setting_targets)) != len(setting_targets):
        raise ValueError("Target settings must not contain duplicate targets.")
    if set(setting_targets) != set(target_columns):
        raise ValueError(
            "Target settings must match target_columns exactly. "
            f"targets={target_columns}, settings={setting_targets}"
        )

    ordered = {str(setting["target"]): setting for setting in settings}
    normalized: list[dict[str, Any]] = []
    for target in target_columns:
        setting = dict(ordered[target])
        task_type = str(setting.get("task_type", "regression")).lower()
        if task_type not in {"regression", "classification", "ordinal"}:
            raise ValueError(
                f"{target}: task_type must be regression, classification, or ordinal."
            )
        goal = str(setting.get("goal", "none")).lower()
        if goal not in {"none", "above", "below", "target"}:
            raise ValueError(
                f"{target}: goal must be none, above, below, or target."
            )

        value = setting.get("value")
        target_class = setting.get("target_class")
        target_classes = _list_values(setting.get("target_classes"))
        class_order = _list_values(setting.get("class_order"))
        target_values = _list_values(setting.get("target_values"))

        # Compatibility with the previous `goal=target, value=<class>` contract.
        if task_type == "classification" and target_class is None and not target_classes:
            if goal == "target" and value is not None and str(value).strip() != "":
                target_class = value
                target_classes = [value]
        if task_type == "ordinal" and goal == "target" and not target_values:
            if value is not None and str(value).strip() != "":
                target_values = [value]

        normalized.append(
            {
                "target": target,
                "task_type": task_type,
                "goal": goal,
                "value": value,
                "target_class": target_class,
                "target_classes": target_classes,
                "class_order": class_order,
                "target_values": target_values,
                "legacy": False,
            }
        )
    return normalized, model_kwargs


def _validate_columns(
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
) -> None:
    """Validate selected feature and target columns."""

    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    missing = [
        column
        for column in [*feature_columns, *target_columns]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    overlap = sorted(set(feature_columns).intersection(target_columns))
    if overlap:
        raise ValueError(
            f"Target columns must not be included in feature columns: {overlap}"
        )


def _clean_rows(
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    drop_missing: bool,
) -> Any:
    """Drop or reject rows missing any selected feature or target."""

    selected = [*feature_columns, *target_columns]
    if not drop_missing:
        if data[selected].isna().any().any():
            raise ValueError(
                "Missing values are present. Enable drop_missing or clean the dataset first."
            )
        return data.reset_index(drop=True)
    return data.dropna(subset=selected).reset_index(drop=True)


def _default_ordered_classes(series: Any) -> list[Any]:
    """Return deterministic classes while preserving numeric order when possible."""

    import pandas as pd

    values = series.dropna().unique().tolist()
    if pd.api.types.is_numeric_dtype(series):
        return sorted(values, key=float)
    return sorted(values, key=lambda value: str(value))


def _class_index(classes: list[Any], requested: Any, *, target: str) -> int:
    """Resolve a user-visible class or rank value to its encoded index."""

    for index, value in enumerate(classes):
        if value == requested or str(value) == str(requested):
            return index
    try:
        requested_number = float(requested)
    except (TypeError, ValueError):
        requested_number = None
    if requested_number is not None:
        for index, value in enumerate(classes):
            try:
                if float(value) == requested_number:
                    return index
            except (TypeError, ValueError):
                continue
    raise ValueError(
        f"{target}: configured value {requested!r} is not present in observed values {classes!r}."
    )


def _ordered_classes(series: Any, requested_order: list[Any], *, target: str) -> list[Any]:
    """Apply a complete user-defined ordinal class order when supplied."""

    observed = _default_ordered_classes(series)
    if not requested_order:
        return observed
    if len(requested_order) != len(observed):
        raise ValueError(
            f"{target}: class_order must contain every observed class exactly once."
        )
    resolved = [observed[_class_index(observed, value, target=target)] for value in requested_order]
    if len({str(value) for value in resolved}) != len(observed):
        raise ValueError(
            f"{target}: class_order must not contain duplicate classes."
        )
    if {str(value) for value in resolved} != {str(value) for value in observed}:
        raise ValueError(
            f"{target}: class_order must match observed classes exactly."
        )
    return resolved


def _resolve_class_indices(
    classes: list[Any],
    requested: list[Any],
    *,
    target: str,
    label: str,
) -> list[int]:
    """Resolve and de-duplicate one or more configured classes."""

    indices: list[int] = []
    for value in requested:
        index = _class_index(classes, value, target=target)
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError(f"{target}: at least one {label} is required.")
    return indices


def _encode_targets(
    data: Any,
    target_settings: list[dict[str, Any]],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Encode heterogeneous targets to a dense numeric training matrix."""

    import pandas as pd

    encoded: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for setting in target_settings:
        target = str(setting["target"])
        task_type = str(setting["task_type"])
        goal = str(setting["goal"])
        series = data[target]

        if task_type == "regression":
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.isna().any():
                raise ValueError(f"{target}: regression target must be numeric.")
            configured_value: float | None
            if goal == "none":
                configured_value = None
            else:
                try:
                    configured_value = float(setting["value"])
                except (TypeError, ValueError) as exc:
                    if setting.get("legacy"):
                        configured_value = float("nan")
                    else:
                        raise ValueError(
                            f"{target}: regression threshold or target value must be numeric."
                        ) from exc
            encoded[target] = numeric.to_numpy(dtype=float)
            metadata[target] = {
                **setting,
                "internal_task": "regression",
                "configured_value": configured_value,
                "classes": None,
                "class_order": None,
                "class_index": None,
                "class_indices": [],
                "num_classes": None,
            }
            continue

        classes = _ordered_classes(
            series,
            _list_values(setting.get("class_order")) if task_type == "ordinal" else [],
            target=target,
        )
        if len(classes) < 2:
            raise ValueError(f"{target}: {task_type} requires at least two classes.")
        class_map = {str(value): index for index, value in enumerate(classes)}
        coded = series.astype(str).map(class_map)
        if coded.isna().any():
            raise ValueError(f"{target}: failed to encode target classes.")

        if task_type == "classification":
            internal_task = "binary" if len(classes) == 2 else "multiclass"
            if internal_task == "binary":
                requested_class = setting.get("target_class")
                if requested_class is None:
                    requested_values = _list_values(setting.get("target_classes"))
                    requested_class = requested_values[0] if requested_values else None
                if requested_class is None:
                    # Preserve the old binary above/below convention (class index 1).
                    requested_class = classes[1]
                class_indices = [
                    _class_index(classes, requested_class, target=target)
                ]
            else:
                requested_values = _list_values(setting.get("target_classes"))
                if not requested_values and setting.get("target_class") is not None:
                    requested_values = [setting.get("target_class")]
                if not requested_values and goal == "target" and setting.get("value") is not None:
                    requested_values = [setting.get("value")]
                class_indices = _resolve_class_indices(
                    classes,
                    requested_values,
                    target=target,
                    label="target class",
                )

            if goal in {"above", "below"}:
                try:
                    configured_value: Any = float(setting["value"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{target}: classification probability threshold must be numeric."
                    ) from exc
                if not 0.0 <= configured_value <= 1.0:
                    raise ValueError(
                        f"{target}: classification probability threshold must be in [0, 1]."
                    )
            else:
                configured_value = None
            target_classes = [classes[index] for index in class_indices]
            class_index = class_indices[0]
            target_values: list[Any] = []
        else:
            internal_task = "ordinal"
            if goal == "target":
                class_indices = _resolve_class_indices(
                    classes,
                    _list_values(setting.get("target_values")),
                    target=target,
                    label="target ordinal class",
                )
                configured_value = [classes[index] for index in class_indices]
                class_index = class_indices[0]
                target_values = list(configured_value)
            elif goal in {"above", "below"}:
                if setting.get("value") is None or str(setting.get("value")).strip() == "":
                    raise ValueError(f"{target}: ordinal boundary class is required.")
                class_index = _class_index(classes, setting["value"], target=target)
                class_indices = [class_index]
                configured_value = classes[class_index]
                target_values = []
            else:
                class_index = None
                class_indices = []
                configured_value = None
                target_values = []
            target_classes = []

        encoded[target] = coded.to_numpy(dtype=float)
        metadata[target] = {
            **setting,
            "internal_task": internal_task,
            "configured_value": configured_value,
            "classes": classes,
            "class_order": classes if internal_task == "ordinal" else None,
            "class_index": int(class_index) if class_index is not None else None,
            "class_indices": [int(index) for index in class_indices],
            "target_classes": target_classes,
            "target_values": target_values,
            "num_classes": len(classes),
        }

    return pd.DataFrame(encoded), metadata


def _model_kwargs(
    base_kwargs: dict[str, Any],
    *,
    model_type: str,
    n_features: int,
    target_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add safe defaults required by projected and discrete-output models."""

    kwargs = dict(base_kwargs)
    if model_type in {"pca", "rembo"}:
        kwargs.setdefault("n_components", max(1, min(2, int(n_features))))
    if target_meta and target_meta["internal_task"] in {"ordinal", "multiclass"}:
        kwargs.setdefault("num_classes", int(target_meta["num_classes"]))
    return kwargs


def _output_spec_kwargs(meta: dict[str, Any]) -> dict[str, Any]:
    """Map a target setting to the hybrid model's scalar objective channel."""

    task = str(meta["internal_task"])
    goal = str(meta["goal"])
    if task == "regression":
        return {
            "sign": -1.0 if goal == "below" else 1.0,
            "eq_target": (
                float(meta["configured_value"]) if goal == "target" else None
            ),
        }
    if task in {"binary", "multiclass"}:
        n_classes = int(meta["num_classes"])
        class_indices = [int(index) for index in meta.get("class_indices", [])]
        utilities = [1.0 if index in class_indices else 0.0 for index in range(n_classes)]
        return {
            "positive_class": class_indices[0],
            "utility_values": utilities,
            "sign": -1.0 if goal == "below" else 1.0,
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
            sign = -1.0 if goal == "below" else 1.0
        return {"utility_values": utilities, "sign": sign}
    raise ValueError(f"Unsupported internal target task: {task}")


def _build_outcome_constraint_config(
    request: Any,
    *,
    target_columns: list[str],
    directions: dict[str, str],
) -> Any | None:
    """Convert legacy original-scale regression constraints to model outputs."""

    constraints = list(request.outcome_constraints or [])
    if not constraints:
        return None

    from bochan.api import OutcomeConstraintConfig

    target_to_index = {target: index for index, target in enumerate(target_columns)}
    output_indices: list[int] = []
    operators: list[str] = []
    thresholds: list[float] = []
    for constraint in constraints:
        target = str(constraint.target)
        if target not in target_to_index:
            raise ValueError(f"Outcome constraint target is not selected: {target}")
        operator = str(constraint.operator)
        threshold = float(constraint.value)
        if directions[target] == "minimize":
            operator = "le" if operator == ">=" else "ge"
            threshold = -threshold
        else:
            operator = "ge" if operator == ">=" else "le"
        output_indices.append(target_to_index[target])
        operators.append(operator)
        thresholds.append(threshold)

    return OutcomeConstraintConfig(
        output_indices=output_indices,
        operators=operators,
        thresholds=thresholds,
    )


def _build_target_constraint_config(
    request: Any,
    *,
    target_settings: list[dict[str, Any]],
    target_metadata: dict[str, dict[str, Any]],
    target_columns: list[str],
    directions: dict[str, str],
    hybrid_model: bool,
) -> Any | None:
    """Build one feasibility rule for each configured above/below target."""

    if all(bool(setting.get("legacy")) for setting in target_settings):
        return _build_outcome_constraint_config(
            request,
            target_columns=target_columns,
            directions=directions,
        )

    from bochan.acquisition.feasible import FeasibilityConstraintSpec
    from bochan.api import OutcomeConstraintConfig

    specs: list[Any] = []
    for setting in target_settings:
        if setting["goal"] not in {"above", "below"}:
            continue
        target = str(setting["target"])
        meta = target_metadata[target]
        task = str(meta["internal_task"])
        goal = str(meta["goal"])
        threshold = (
            float(meta["class_index"])
            if task == "ordinal"
            else float(meta["configured_value"])
        )
        output: Any = target if hybrid_model else target_columns.index(target)
        if hybrid_model and goal == "below":
            sense = "ge"
            threshold = -threshold
        else:
            sense = "ge" if goal == "above" else "le"
        specs.append(
            FeasibilityConstraintSpec(
                output=output,
                threshold=threshold,
                sense=sense,
            )
        )

    return OutcomeConstraintConfig(constraints=specs) if specs else None


def _reference_point(values: Any) -> Any:
    """Build a dominated reference point in maximization objective space."""

    import torch

    lower = values.min(dim=0).values
    upper = values.max(dim=0).values
    scale = (upper - lower).abs()
    fallback = torch.maximum(lower.abs(), upper.abs()).clamp_min(1.0)
    margin = torch.where(scale > 1e-12, scale * 0.1, fallback * 0.1)
    return lower - margin


def _as_2d(value: Any, *, n_rows: int) -> Any:
    """Normalize posterior mean/variance to shape ``[n, m]``."""

    import torch

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim > 2:
        tensor = tensor.reshape(n_rows, -1)
    if tensor.ndim != 2 or tensor.shape[0] != n_rows:
        raise RuntimeError(
            f"Could not normalize prediction to [n, m]. shape={tuple(tensor.shape)}"
        )
    return tensor


def _objective_values_direct(train_y: Any, target_settings: list[dict[str, Any]]) -> Any:
    """Transform direct multitask regression observations into objective space."""

    import torch

    columns = []
    for index, setting in enumerate(target_settings):
        values = train_y[:, index]
        goal = str(setting["goal"])
        if goal == "below":
            values = -values
        elif goal == "target":
            values = -torch.abs(values - float(setting["value"]))
        columns.append(values.unsqueeze(-1))
    return torch.cat(columns, dim=-1)


__all__ = [
    "_as_2d",
    "_build_outcome_constraint_config",
    "_build_target_constraint_config",
    "_clean_rows",
    "_encode_targets",
    "_model_kwargs",
    "_objective_values_direct",
    "_output_spec_kwargs",
    "_reference_point",
    "_resolve_target_settings",
    "_resolve_targets",
    "_validate_columns",
]
