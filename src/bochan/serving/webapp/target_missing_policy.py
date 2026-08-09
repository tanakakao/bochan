"""Request-local missing-value policy for the Web target workflow.

The policy is consumed through normal imports from :mod:`target_settings` and
:mod:`tabular_backend`. It does not replace workflow functions or mutate model
registries at runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_WEB_FEATURE_MISSING_KEY = "web_feature_missing"
_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_target_missing_state",
    default=None,
)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(vars(value))


def _request_targets(request: Any) -> list[str]:
    targets = [str(value) for value in list(getattr(request, "target_columns", []) or [])]
    target = getattr(request, "target_column", None)
    if not targets and target:
        targets = [str(target)]
    return targets


def _categorical_feature_columns(request: Any) -> list[str]:
    columns: list[str] = []
    for raw in list(getattr(request, "search_space", []) or []):
        setting = _mapping(raw)
        if str(setting.get("type", "auto")).lower() != "categorical":
            continue
        name = str(setting.get("name", ""))
        if name and name not in columns:
            columns.append(name)
    return columns


def _feature_missing_settings(request: Any) -> dict[str, Any]:
    model_kwargs = dict(getattr(request, "model_kwargs", {}) or {})
    raw = _mapping(model_kwargs.get(_WEB_FEATURE_MISSING_KEY))
    strategy = str(raw.get("strategy", "drop")).lower()
    if strategy not in {"drop", "impute"}:
        raise ValueError("Feature missing strategy must be 'drop' or 'impute'.")
    continuous = str(raw.get("continuous_impute_strategy", "mean")).lower()
    if continuous not in {"mean", "iterative"}:
        raise ValueError(
            "continuous_impute_strategy must be 'mean' or 'iterative'."
        )
    categorical = str(raw.get("categorical_impute_strategy", "mode")).lower()
    if categorical != "mode":
        raise ValueError("categorical_impute_strategy currently supports only 'mode'.")
    max_iter = int(raw.get("impute_max_iter", 10))
    if max_iter < 1:
        raise ValueError("impute_max_iter must be at least 1.")
    random_state = raw.get("impute_random_state")
    if random_state is not None:
        random_state = int(random_state)
    return {
        "feature_missing_strategy": strategy,
        "continuous_impute_strategy": continuous,
        "categorical_impute_strategy": categorical,
        "impute_random_state": random_state,
        "impute_max_iter": max_iter,
        "multiple_impute_sample_posterior": bool(
            raw.get("multiple_impute_sample_posterior", False)
        ),
        "categorical_feature_columns": _categorical_feature_columns(request),
    }


@contextmanager
def target_missing_run(request: Any) -> Iterator[dict[str, Any]]:
    """Activate one request-local feature / target missing-value policy."""

    targets = _request_targets(request)
    requested_model_type = str(getattr(request, "model_type", "base"))
    preserve = requested_model_type == "multitask" and len(targets) > 1
    feature_settings = _feature_missing_settings(request)
    state: dict[str, Any] = {
        "requested_model_type": requested_model_type,
        "target_columns": targets,
        "preserve_target_missing": preserve,
        "policy": "wide_multitask" if preserve else "drop_rows",
        "target_missing_detected": False,
        "target_missing_counts": {},
        "feature_missing_detected": False,
        "feature_missing_counts": {},
        "feature_impute_values": {},
        "dropped_feature_rows": 0,
        "dropped_target_rows": 0,
        "dropped_all_target_missing_rows": 0,
        "multitask_variant": None,
        "effective_model_type": requested_model_type,
        # Observed targets are never overwritten with posterior means. This flag
        # remains explicit in the response metadata to make that fact inspectable.
        "acquisition_baseline_completed": False,
        **feature_settings,
    }
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def current_target_missing_state() -> dict[str, Any] | None:
    """Return the mutable state for source-level workflow helpers."""

    return _STATE.get()


def current_target_missing_report() -> dict[str, Any]:
    """Return a copy of the active request report."""

    return dict(_STATE.get() or {})


def _safe_impute_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _impute_feature_columns(
    data: Any,
    feature_columns: list[str],
    state: dict[str, Any],
) -> Any:
    """Reuse the tabular feature-imputation implementation."""

    import pandas as pd

    from bochan.tabular.config import TabularDataConfig
    from bochan.tabular.converter import _apply_missing_value_strategy

    config = TabularDataConfig(
        input_cols=feature_columns,
        target_cols=[],
        categorical_cols=list(state.get("categorical_feature_columns") or []),
        missing_strategy="impute",
        continuous_impute_strategy=str(state["continuous_impute_strategy"]),
        categorical_impute_strategy=str(state["categorical_impute_strategy"]),
        impute_targets=False,
        impute_random_state=state.get("impute_random_state"),
        impute_max_iter=int(state["impute_max_iter"]),
        multiple_impute_sample_posterior=bool(
            state.get("multiple_impute_sample_posterior")
        ),
    )
    prepared, impute_values, _ = _apply_missing_value_strategy(
        work=data.copy(),
        input_cols=feature_columns,
        target_cols=[],
        config=config,
        pd=pd,
    )
    if prepared[feature_columns].isna().any().any():
        raise ValueError("Feature imputation left unresolved missing values.")
    state["feature_impute_values"] = {
        str(column): _safe_impute_value(value)
        for column, value in dict(impute_values or {}).items()
    }
    return prepared


def clean_rows(
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    drop_missing: bool,
) -> Any:
    """Apply independent explanatory-variable and target missing policies."""

    from .target_settings_core import _clean_rows as core_clean_rows

    state = _STATE.get()
    if state is None:
        return core_clean_rows(
            data,
            feature_columns,
            target_columns,
            drop_missing=drop_missing,
        )

    work = data.copy()
    feature_counts = {
        feature: int(work[feature].isna().sum()) for feature in feature_columns
    }
    state["feature_missing_counts"] = feature_counts
    state["feature_missing_detected"] = any(
        count > 0 for count in feature_counts.values()
    )

    if state.get("feature_missing_strategy") == "impute":
        work = _impute_feature_columns(work, feature_columns, state)
        state["dropped_feature_rows"] = 0
    else:
        feature_missing = work[feature_columns].isna().any(axis=1)
        if bool(feature_missing.any()) and not drop_missing:
            raise ValueError(
                "Missing feature values are present. Select row deletion or "
                "feature imputation."
            )
        state["dropped_feature_rows"] = int(feature_missing.sum())
        work = work.loc[~feature_missing].copy()

    preserve = bool(state.get("preserve_target_missing"))
    if preserve:
        all_targets_missing = work[target_columns].isna().all(axis=1)
        state["dropped_all_target_missing_rows"] = int(all_targets_missing.sum())
        state["dropped_target_rows"] = int(all_targets_missing.sum())
        work = work.loc[~all_targets_missing].copy()
        missing_counts = {
            target: int(work[target].isna().sum()) for target in target_columns
        }
        empty_targets = [
            target for target in target_columns if int(work[target].notna().sum()) == 0
        ]
        if empty_targets:
            raise ValueError(
                "Every multitask target requires at least one observed value. "
                f"Targets without observations: {empty_targets}."
            )
    else:
        missing_counts = {
            target: int(work[target].isna().sum()) for target in target_columns
        }
        target_missing = work[target_columns].isna().any(axis=1)
        state["dropped_target_rows"] = int(target_missing.sum())
        work = work.loc[~target_missing].copy()

    state["target_missing_counts"] = missing_counts
    state["target_missing_detected"] = any(
        count > 0 for count in missing_counts.values()
    )
    return work.reset_index(drop=True)


def encode_targets(
    data: Any,
    target_settings: list[dict[str, Any]],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Preserve regression NaNs while deriving metadata from observed values."""

    from .target_settings_core import _encode_targets as core_encode_targets

    state = _STATE.get()
    preserve = bool(state and state.get("preserve_target_missing"))
    target_names = [str(value["target"]) for value in target_settings]
    if not preserve or not data[target_names].isna().any().any():
        return core_encode_targets(data, target_settings)

    if any(
        str(setting.get("task_type", "regression")) != "regression"
        for setting in target_settings
    ):
        return core_encode_targets(data, target_settings)

    import pandas as pd

    metadata_data = data.copy()
    numeric_targets: dict[str, Any] = {}
    for setting in target_settings:
        target = str(setting["target"])
        series = data[target]
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = series.notna() & numeric.isna()
        if bool(invalid.any()):
            raise ValueError(f"{target}: regression target must be numeric.")
        if int(numeric.notna().sum()) == 0:
            raise ValueError(f"{target}: regression target requires an observed value.")
        numeric_targets[target] = numeric
        # Metadata thresholds/classes are evaluated by the existing encoder. The
        # fill is local to this temporary metadata frame and never becomes train Y.
        metadata_data[target] = numeric.fillna(float(numeric.mean()))

    encoded, metadata = core_encode_targets(metadata_data, target_settings)
    for target, numeric in numeric_targets.items():
        encoded[target] = numeric.to_numpy(dtype=float)
    return encoded, metadata


def resolve_target_settings(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve target settings and remove Web-only feature-missing metadata."""

    from .target_settings_core import _resolve_target_settings as core_resolve

    settings, model_kwargs = core_resolve(*args, **kwargs)
    cleaned = dict(model_kwargs)
    cleaned.pop(_WEB_FEATURE_MISSING_KEY, None)
    return settings, cleaned


def model_variant(model: Any) -> tuple[str | None, str | None]:
    """Return the fitted multitask variant and effective model key."""

    class_name = type(model).__name__.lower()
    if "kronecker" in class_name:
        return "kronecker", "kronecker"
    if "widemultitask" in class_name:
        return "wide_multitask", "multitask"
    return None, None


__all__ = [
    "clean_rows",
    "current_target_missing_report",
    "current_target_missing_state",
    "encode_targets",
    "model_variant",
    "resolve_target_settings",
    "target_missing_run",
]
