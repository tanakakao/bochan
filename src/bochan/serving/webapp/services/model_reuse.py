"""Request-scoped fitted-model reuse for the React Web workbench."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Any

from .candidate_runtime import apply_web_candidate_runtime_defaults

_WEB_REUSE_MODEL_KEY = "web_reuse_model_run_id"
_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_model_reuse_state",
    default=None,
)
_SIGNATURE_LOCK = RLock()
_MODEL_SIGNATURES: OrderedDict[str, str] = OrderedDict()
_MAX_SIGNATURES = 12


def _plain(value: Any) -> Any:
    """Convert Pydantic and container values into stable JSON-compatible data."""

    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=False))
    if hasattr(value, "dict"):
        return _plain(value.dict())
    if isinstance(value, dict):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _target_model_settings(model_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only target fields that affect encoding or model construction.

    Goal, threshold / target value, optimization role, direction, selected
    classification class, and level-set weight are candidate-time semantics. The
    fitted classifier / regressor does not change when those values change. An
    ordinal ``class_order`` does change encoded training labels and therefore
    remains part of the fitted-model signature.
    """

    raw_settings = list(model_kwargs.pop("web_target_settings", []) or [])
    model_kwargs.pop("web_target_roles", None)
    settings: list[dict[str, Any]] = []
    for raw in raw_settings:
        value = _plain(raw)
        if not isinstance(value, dict):
            continue
        settings.append(
            {
                "target": value.get("target"),
                "task_type": value.get("task_type"),
                "class_order": list(value.get("class_order") or []),
            }
        )
    return settings


def _strip_candidate_only_composition_settings(model_kwargs: dict[str, Any]) -> None:
    """Remove composition rules that affect proposals but not fitted features."""

    raw = model_kwargs.get("web_composition")
    if raw is None:
        return
    value = _plain(raw)
    if not isinstance(value, dict):
        return
    # Element constraints are enforced by CandidateService after the composition
    # transformer has been fitted. Representation, elements, normalization,
    # bounds, coordinate bounds, and the remaining settings stay in the model
    # signature because they may alter fitted features or the input transform.
    value.pop("element_constraints", None)
    model_kwargs["web_composition"] = value


def _model_search_space(request: Any) -> list[dict[str, Any]]:
    """Keep feature encoding and normalization bounds, not proposal-only fields."""

    settings: list[dict[str, Any]] = []
    for raw in list(getattr(request, "search_space", []) or []):
        value = _plain(raw)
        if not isinstance(value, dict):
            continue
        settings.append(
            {
                "name": value.get("name"),
                "type": value.get("type"),
                "lower": value.get("lower"),
                "upper": value.get("upper"),
                "categories": list(value.get("categories") or []),
            }
        )
    return settings


def model_reuse_signature(request: Any) -> str:
    """Return a fingerprint containing only settings that affect model fitting."""

    model_kwargs = dict(getattr(request, "model_kwargs", {}) or {})
    model_kwargs.pop(_WEB_REUSE_MODEL_KEY, None)
    target_model_settings = _target_model_settings(model_kwargs)
    _strip_candidate_only_composition_settings(model_kwargs)
    payload = {
        "dataset_id": getattr(request, "dataset_id", None),
        "feature_columns": list(getattr(request, "feature_columns", []) or []),
        "target_columns": list(getattr(request, "target_columns", []) or []),
        "target_model_settings": target_model_settings,
        "model_type": getattr(request, "model_type", None),
        "model_kwargs": model_kwargs,
        "fit_maxiter": getattr(request, "fit_maxiter", None),
        "normalize": getattr(request, "normalize", None),
        "outcome_transform": getattr(request, "outcome_transform", None),
        "input_perturbation": getattr(request, "input_perturbation", None),
        "n_w": getattr(request, "n_w", None),
        "perturbation_std": getattr(request, "perturbation_std", None),
        "search_space": _model_search_space(request),
        "drop_missing": getattr(request, "drop_missing", None),
    }
    encoded = json.dumps(
        _plain(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_model_reuse_request(request: Any) -> tuple[Any, str | None]:
    """Prepare an execution copy before normal Web model/candidate configuration."""

    model_kwargs = dict(getattr(request, "model_kwargs", {}) or {})
    raw_source = model_kwargs.pop(_WEB_REUSE_MODEL_KEY, None)
    source_run_id = str(raw_source).strip() if raw_source is not None else ""
    source_run_id = source_run_id or None

    if hasattr(request, "model_copy"):
        cleaned = request.model_copy(update={"model_kwargs": model_kwargs})
    else:
        values = dict(vars(request))
        values["model_kwargs"] = model_kwargs
        cleaned = type("WebRequest", (), values)()
    cleaned = apply_web_candidate_runtime_defaults(cleaned)
    return cleaned, source_run_id


@contextmanager
def model_reuse_run(request: Any, source_run_id: str | None) -> Iterator[dict[str, Any]]:
    """Activate reuse metadata for one Web request."""

    state: dict[str, Any] = {
        "requested": source_run_id is not None,
        "source_run_id": source_run_id,
        "model_signature": model_reuse_signature(request),
        "model_reused": False,
        "fit_skipped": False,
    }
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def current_model_reuse_state() -> dict[str, Any] | None:
    """Return the mutable state active in the current request context."""

    return _STATE.get()


def register_model_signature(run_id: str, signature: str) -> None:
    """Associate a run id with an explicit fitted-model signature."""

    if not run_id or not signature:
        return
    with _SIGNATURE_LOCK:
        _MODEL_SIGNATURES.pop(run_id, None)
        _MODEL_SIGNATURES[run_id] = signature
        while len(_MODEL_SIGNATURES) > _MAX_SIGNATURES:
            _MODEL_SIGNATURES.popitem(last=False)


def get_registered_model_signature(run_id: str) -> str | None:
    """Return a fitted-model signature without removing the LRU entry."""

    with _SIGNATURE_LOCK:
        signature = _MODEL_SIGNATURES.get(run_id)
        if signature is not None:
            _MODEL_SIGNATURES.move_to_end(run_id)
        return signature


def register_fitted_model(run_id: str) -> None:
    """Associate the current request's model fingerprint with one fitted run."""

    state = _STATE.get()
    signature = str((state or {}).get("model_signature") or "")
    register_model_signature(run_id, signature)


def mark_model_reused(source_run_id: str) -> None:
    """Record that candidate generation reused an existing fitted model."""

    state = _STATE.get()
    if state is None:
        return
    state["source_run_id"] = source_run_id
    state["model_reused"] = True
    state["fit_skipped"] = True


def _refresh_hybrid_wrapper(source_model: Any, model_config: Any) -> Any:
    """Rebuild only the objective-view wrapper around already-fitted submodels."""

    from bochan.api.modeling.build import (
        _build_wrapper_from_submodels,
        _resolve_output_configs,
    )
    from bochan.models.hybrid import HybridMultiOutputModel

    if not isinstance(source_model, HybridMultiOutputModel):
        raise RuntimeError(
            "The reusable fitted model does not expose the expected Hybrid wrapper."
        )
    multi_output_config = getattr(model_config, "multi_output_config", None)
    if multi_output_config is None:
        raise RuntimeError(
            "Current Hybrid model configuration has no multi_output_config."
        )

    submodels = list(source_model.models)
    output_configs, output_names, inline_spec_kwargs, _ = _resolve_output_configs(
        model_config,
        len(submodels),
    )
    refreshed = _build_wrapper_from_submodels(
        submodels,
        output_configs,
        multi_output_config,
        output_names=output_names,
        output_spec_kwargs=inline_spec_kwargs,
    )

    # Partial-target training wraps the same Hybrid specs with retained wide
    # observation metadata. Preserve that contract while changing only the
    # candidate-time OutputSpec view.
    if all(
        hasattr(source_model, name)
        for name in ("train_X_wide", "train_Y_wide", "observed_mask_wide")
    ):
        from bochan.models.hybrid.partial_observation import (
            PartiallyObservedHybridMultiOutputModel,
        )

        refreshed = PartiallyObservedHybridMultiOutputModel(
            specs=list(refreshed.specs),
            train_X_wide=source_model.train_X_wide,
            train_Y_wide=source_model.train_Y_wide,
            observed_mask_wide=source_model.observed_mask_wide,
        )

    refreshed.train(bool(getattr(source_model, "training", False)))
    return refreshed


def _clone_bayesian_optimizer_for_reuse(
    source_bo: Any,
    *,
    model_config: Any,
    hybrid_model: bool,
) -> Any:
    """Clone mutable optimizer state and refresh candidate-time model semantics."""

    reused_bo = copy.copy(source_bo)
    if hasattr(source_bo, "history"):
        reused_bo.history = list(source_bo.history)
    reused_bo.model_config = model_config

    source_bundle = getattr(source_bo, "bundle", None)
    reused_bundle = None
    if source_bundle is not None:
        reused_bundle = copy.copy(source_bundle)
        reused_bundle.metadata = dict(getattr(source_bundle, "metadata", {}) or {})
        reused_bundle.model_config = model_config
        reused_bo.bundle = reused_bundle

    source_model = getattr(source_bo, "model", None)
    if source_model is None:
        raise RuntimeError("The selected fitted optimizer has no model to reuse.")
    reused_model = (
        _refresh_hybrid_wrapper(source_model, model_config)
        if hybrid_model
        else source_model
    )
    reused_bo.model = reused_model
    if reused_bundle is not None:
        reused_bundle.model = reused_model
    return reused_bo


def _clone_candidate_service(
    source_optimizer: Any,
    reused_optimizer: Any,
    *,
    composition_config: dict[str, Any] | None,
) -> None:
    """Clone request-local candidate state and apply current composition rules."""

    source_service = getattr(source_optimizer, "candidates", None)
    if source_service is None:
        return
    reused_service = copy.copy(source_service)
    reused_service.composition = reused_optimizer.composition
    if composition_config is not None:
        reused_service.element_constraints = reused_service.element_resolver.normalize(
            composition_config.get("element_constraints") or []
        )
        reused_service.projector().validate()
    reused_optimizer.candidates = reused_service


def reuse_fitted_tabular_optimizer(
    *,
    source_run_id: str,
    current_run_id: str,
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    target_metadata: dict[str, dict[str, Any]],
    model_config: Any,
    hybrid_model: bool,
    composition_config: dict[str, Any] | None = None,
) -> Any:
    """Reuse fitted predictors while rebuilding current candidate-time semantics."""

    from ..services.visualization_sessions import (
        attach_fitted_tabular_optimizer,
        get_visualization_session,
    )

    state = _STATE.get()
    current_signature = str((state or {}).get("model_signature") or "")
    with _SIGNATURE_LOCK:
        source_signature = _MODEL_SIGNATURES.get(source_run_id)
    if source_signature is None:
        raise ValueError(
            "The selected fitted model is no longer available in this FastAPI process. "
            "Run model training again."
        )
    if source_signature != current_signature:
        raise ValueError(
            "The fitted model cannot be reused because data, target task, feature type, "
            "model, input-transform, missing-value, bounds, or fit settings have changed."
        )

    source = get_visualization_session(source_run_id)
    source_tabular = source.tabular_optimizer
    reused = copy.copy(source_tabular)
    reused.__dict__.pop("candidate", None)
    reused.dataset = copy.copy(source_tabular.dataset)
    if reused.dataset is None:
        raise RuntimeError("The selected fitted model has no retained tabular dataset.")

    observed = getattr(source_tabular, "web_observed_target_tensor", None)
    if observed is not None:
        reused.dataset.Y = observed.detach().clone()
    elif reused.dataset.Y is not None:
        reused.dataset.Y = reused.dataset.Y.detach().clone()
    if reused.dataset.X is not None:
        reused.dataset.X = reused.dataset.X.detach().clone()
    if reused.dataset.bounds is not None:
        reused.dataset.bounds = reused.dataset.bounds.detach().clone()

    reused.model_config = model_config
    reused.bo = _clone_bayesian_optimizer_for_reuse(
        source_tabular.bo,
        model_config=model_config,
        hybrid_model=hybrid_model,
    )
    _clone_candidate_service(
        source_tabular,
        reused,
        composition_config=composition_config,
    )

    reused.web_model_reused = True
    reused.web_model_reuse_source_run_id = source_run_id
    attach_fitted_tabular_optimizer(
        current_run_id,
        tabular_optimizer=reused,
        data=data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        target_metadata=target_metadata,
        hybrid_model=hybrid_model,
    )
    mark_model_reused(source_run_id)
    register_fitted_model(current_run_id)
    return reused


__all__ = [
    "current_model_reuse_state",
    "get_registered_model_signature",
    "mark_model_reused",
    "model_reuse_run",
    "model_reuse_signature",
    "prepare_model_reuse_request",
    "register_fitted_model",
    "register_model_signature",
    "reuse_fitted_tabular_optimizer",
]
