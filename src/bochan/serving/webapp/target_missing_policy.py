"""Target-missing policy and adaptive multi-task model selection for the Web API.

The Web workbench uses one user-facing ``multitask`` model option.  This module
keeps partially observed multi-output regression rows for that option and chooses
an implementation from the actual target matrix:

- incomplete targets -> ``WideMultiTaskGP`` (NaNs are unobserved task cells),
- complete targets -> ``PerturbationSupportedKroneckerMultiTaskGP``.

Single-objective runs and non-multitask multi-objective runs continue to remove
rows with missing targets.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_target_missing_state",
    default=None,
)
_ORIGINAL_CLEAN_ROWS: Any | None = None
_ORIGINAL_ENCODE_TARGETS: Any | None = None
_ORIGINAL_FIT_TABULAR_OPTIMIZER: Any | None = None
_INSTALLED = False


def _request_targets(request: Any) -> list[str]:
    targets = [str(value) for value in list(getattr(request, "target_columns", []) or [])]
    target = getattr(request, "target_column", None)
    if not targets and target:
        targets = [str(target)]
    return targets


@contextmanager
def target_missing_run(request: Any) -> Iterator[dict[str, Any]]:
    """Activate one request-local missing-target policy and return its report."""

    targets = _request_targets(request)
    requested_model_type = str(getattr(request, "model_type", "base"))
    preserve = requested_model_type == "multitask" and len(targets) > 1
    state: dict[str, Any] = {
        "requested_model_type": requested_model_type,
        "target_columns": targets,
        "preserve_target_missing": preserve,
        "policy": "wide_multitask" if preserve else "drop_rows",
        "target_missing_detected": False,
        "target_missing_counts": {},
        "dropped_feature_rows": 0,
        "dropped_target_rows": 0,
        "dropped_all_target_missing_rows": 0,
        "multitask_variant": None,
        "effective_model_type": requested_model_type,
        "acquisition_baseline_completed": False,
    }
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def current_target_missing_report() -> dict[str, Any]:
    """Return a copy of the active request report."""

    return dict(_STATE.get() or {})


def _default_clean_rows(
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    drop_missing: bool,
) -> Any:
    from .target_settings import _clean_rows

    return _clean_rows(
        data,
        feature_columns,
        target_columns,
        drop_missing=drop_missing,
    )


def clean_rows(
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    drop_missing: bool,
) -> Any:
    """Apply the request-local row policy used by ``workflows_tabular``."""

    state = _STATE.get()
    preserve = bool(state and state.get("preserve_target_missing"))
    original = _ORIGINAL_CLEAN_ROWS or _default_clean_rows
    if not preserve:
        cleaned = original(
            data,
            feature_columns,
            target_columns,
            drop_missing=drop_missing,
        )
        if state is not None:
            state["target_missing_counts"] = {
                target: int(data[target].isna().sum()) for target in target_columns
            }
            state["target_missing_detected"] = any(
                count > 0 for count in state["target_missing_counts"].values()
            )
            state["dropped_target_rows"] = int(len(data) - len(cleaned))
        return cleaned

    work = data.copy()
    feature_missing = work[feature_columns].isna().any(axis=1)
    if bool(feature_missing.any()) and not drop_missing:
        raise ValueError(
            "Missing feature values are present. Enable drop_missing or clean the "
            "feature columns first."
        )
    before_features = len(work)
    work = work.loc[~feature_missing].copy()

    all_targets_missing = work[target_columns].isna().all(axis=1)
    before_empty_targets = len(work)
    work = work.loc[~all_targets_missing].copy()
    missing_counts = {
        target: int(work[target].isna().sum()) for target in target_columns
    }
    empty_targets = [target for target in target_columns if work[target].notna().sum() == 0]
    if empty_targets:
        raise ValueError(
            "Every multitask target requires at least one observed value. "
            f"Targets without observations: {empty_targets}."
        )

    if state is not None:
        state["target_missing_counts"] = missing_counts
        state["target_missing_detected"] = any(count > 0 for count in missing_counts.values())
        state["dropped_feature_rows"] = int(before_features - len(work) - (before_empty_targets - len(work)))
        state["dropped_all_target_missing_rows"] = int(before_empty_targets - len(work))
        state["dropped_target_rows"] = int(before_empty_targets - len(work))
    return work.reset_index(drop=True)


def _default_encode_targets(
    data: Any,
    target_settings: list[dict[str, Any]],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    from .target_settings import _encode_targets

    return _encode_targets(data, target_settings)


def encode_targets(
    data: Any,
    target_settings: list[dict[str, Any]],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Preserve regression NaNs while reusing the normal target metadata encoder."""

    state = _STATE.get()
    preserve = bool(state and state.get("preserve_target_missing"))
    original = _ORIGINAL_ENCODE_TARGETS or _default_encode_targets
    if not preserve or not data[[str(value["target"]) for value in target_settings]].isna().any().any():
        return original(data, target_settings)

    if any(str(setting.get("task_type", "regression")) != "regression" for setting in target_settings):
        return original(data, target_settings)

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
        metadata_data[target] = numeric.fillna(float(numeric.mean()))

    encoded, metadata = original(metadata_data, target_settings)
    for target, numeric in numeric_targets.items():
        encoded[target] = numeric.to_numpy(dtype=float)
    return encoded, metadata


def adaptive_multitask_gp(train_X: Any, train_Y: Any, **kwargs: Any) -> Any:
    """Build WideMultiTaskGP for incomplete Y, otherwise build KroneckerMultiTaskGP."""

    import torch

    train_X = torch.as_tensor(train_X)
    train_Y = torch.as_tensor(train_Y, device=train_X.device)
    if bool(torch.isnan(train_Y).any()):
        from bochan.models.wide_multitask_variants import WideMultiTaskGP

        model = WideMultiTaskGP(train_X=train_X, train_Y=train_Y, **kwargs)
        variant = "wide_multitask"
        effective_model_type = "multitask"
    else:
        from bochan.models.regression.gaussian import (
            PerturbationSupportedKroneckerMultiTaskGP,
        )

        model = PerturbationSupportedKroneckerMultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            **kwargs,
        )
        variant = "kronecker"
        effective_model_type = "kronecker"

    setattr(model, "web_multitask_variant", variant)
    setattr(model, "web_effective_model_type", effective_model_type)
    state = _STATE.get()
    if state is not None:
        state["multitask_variant"] = variant
        state["effective_model_type"] = effective_model_type
    return model


def _posterior_training_mean(model: Any, X: Any, *, n_outputs: int) -> Any:
    """Return one posterior-mean row per original training input."""

    import torch

    with torch.no_grad():
        mean = model.posterior(X).mean.detach()
    n_rows = int(X.shape[0])
    flat = mean.reshape(-1, n_outputs)
    if int(flat.shape[0]) % n_rows != 0:
        raise RuntimeError(
            "Could not align multitask posterior means with training rows: "
            f"mean.shape={tuple(mean.shape)}, n_rows={n_rows}."
        )
    return flat.reshape(n_rows, -1, n_outputs).mean(dim=1)


def fit_tabular_optimizer(**kwargs: Any) -> Any:
    """Fit normally, then complete only the acquisition baseline for missing targets."""

    original = _ORIGINAL_FIT_TABULAR_OPTIMIZER
    if original is None:
        from .tabular_backend import fit_tabular_optimizer as original

    optimizer = original(**kwargs)
    dataset = optimizer.dataset
    if dataset is None or dataset.Y is None:
        return optimizer

    import torch

    observed_y = dataset.Y.detach().clone()
    missing_mask = torch.isnan(observed_y)
    model = optimizer.bo.model
    variant = getattr(model, "web_multitask_variant", None)
    state = _STATE.get()
    if state is not None and variant:
        state["multitask_variant"] = variant
        state["effective_model_type"] = getattr(
            model,
            "web_effective_model_type",
            "multitask" if variant == "wide_multitask" else "kronecker",
        )

    optimizer.web_observed_target_tensor = observed_y
    optimizer.web_target_missing_mask = missing_mask
    if bool(missing_mask.any()):
        predicted = _posterior_training_mean(
            model,
            dataset.X,
            n_outputs=int(observed_y.shape[-1]),
        ).to(dtype=observed_y.dtype, device=observed_y.device)
        dataset.Y = torch.where(missing_mask, predicted, observed_y)
        if state is not None:
            state["acquisition_baseline_completed"] = True
    return optimizer


def install_workflow_adapters(workflows_tabular: Any) -> None:
    """Install request-aware adapters once and register the adaptive Web model."""

    global _INSTALLED
    global _ORIGINAL_CLEAN_ROWS
    global _ORIGINAL_ENCODE_TARGETS
    global _ORIGINAL_FIT_TABULAR_OPTIMIZER

    if _INSTALLED:
        return
    _ORIGINAL_CLEAN_ROWS = workflows_tabular._clean_rows
    _ORIGINAL_ENCODE_TARGETS = workflows_tabular._encode_targets
    _ORIGINAL_FIT_TABULAR_OPTIMIZER = workflows_tabular.fit_tabular_optimizer
    workflows_tabular._clean_rows = clean_rows
    workflows_tabular._encode_targets = encode_targets
    workflows_tabular.fit_tabular_optimizer = fit_tabular_optimizer

    from bochan.api.model_registry import MODEL_REGISTRY

    registry = MODEL_REGISTRY.raw()
    adaptive_path = (
        "bochan.serving.webapp.target_missing_policy",
        "adaptive_multitask_gp",
    )
    registry["normal"]["regression"]["multitask"] = adaptive_path
    registry["normal"]["multi_objective"]["multitask"] = adaptive_path
    _INSTALLED = True


def model_variant(model: Any) -> tuple[str | None, str | None]:
    """Return the fitted multitask variant and effective model key."""

    variant = getattr(model, "web_multitask_variant", None)
    effective = getattr(model, "web_effective_model_type", None)
    if variant or effective:
        return variant, effective
    class_name = type(model).__name__.lower()
    if "kronecker" in class_name:
        return "kronecker", "kronecker"
    if "widemultitask" in class_name:
        return "wide_multitask", "multitask"
    return None, None


__all__ = [
    "adaptive_multitask_gp",
    "clean_rows",
    "current_target_missing_report",
    "encode_targets",
    "fit_tabular_optimizer",
    "install_workflow_adapters",
    "model_variant",
    "target_missing_run",
]
