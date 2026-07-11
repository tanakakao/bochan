"""FastAPI integration for the high-level bochan API.

This module is optional. Importing ``bochan.api`` does not require FastAPI;
only importing ``bochan.api.fastapi`` requires the optional API dependencies.

Run:
    uvicorn bochan.api.fastapi:app --reload
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
from threading import RLock
from typing import Any

try:
    from fastapi import APIRouter, FastAPI, HTTPException
    from pydantic import BaseModel, Field

    try:  # Pydantic v2
        from pydantic import ConfigDict
    except Exception:  # pragma: no cover - pydantic v1 fallback
        ConfigDict = None  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - optional dependency import guard
    raise ImportError(
        "bochan.api.fastapi requires optional API dependencies. "
        "Install with: pip install 'bochan[api]'"
    ) from exc

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    CandidateRepairConfig,
    DataContext,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
    MultiObjectiveConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OptimizeConfig,
    OutcomeConstraintConfig,
    OutputConfig,
)


class APIBaseModel(BaseModel):
    """Base request model with pydantic v1/v2 support."""

    if ConfigDict is not None:  # pydantic v2
        model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
    else:  # pragma: no cover - pydantic v1 fallback
        class Config:
            allow_population_by_field_name = True
            arbitrary_types_allowed = True


class TensorOptions(APIBaseModel):
    """Tensor conversion options for JSON payloads."""

    dtype: str = "float64"
    device: str | None = None


class FitSessionRequest(APIBaseModel):
    """Create and fit a stateful BayesianOptimizer session."""

    train_X: Any
    train_Y: Any
    bounds: Any | None = None
    model_config_payload: dict[str, Any] = Field(default_factory=dict, alias="model_config")
    fit_config_payload: dict[str, Any] | None = Field(default=None, alias="fit_config")
    tensor_options: TensorOptions = Field(default_factory=TensorOptions)


class PredictRequest(APIBaseModel):
    """Predict with a fitted session."""

    X: Any
    return_type: str = "mean_variance"
    tensor_options: TensorOptions = Field(default_factory=TensorOptions)
    posterior_kwargs: dict[str, Any] | None = None


class CandidateRequest(APIBaseModel):
    """Generate candidates with a fitted session."""

    acquisition_config_payload: dict[str, Any] = Field(default_factory=dict, alias="acquisition_config")
    optimize_config_payload: dict[str, Any] = Field(default_factory=dict, alias="optimize_config")
    data_context_payload: dict[str, Any] | None = Field(default=None, alias="data_context")
    bounds: Any | None = None
    tensor_options: TensorOptions = Field(default_factory=TensorOptions)


class TellRequest(APIBaseModel):
    """Add observations to a fitted session."""

    new_X: Any
    new_Y: Any
    refit: bool = True
    fit_config_payload: dict[str, Any] | None = Field(default=None, alias="fit_config")
    tensor_options: TensorOptions = Field(default_factory=TensorOptions)


class SuggestRequest(APIBaseModel):
    """Stateless fit-and-suggest request."""

    train_X: Any
    train_Y: Any
    bounds: Any
    model_config_payload: dict[str, Any] = Field(default_factory=dict, alias="model_config")
    fit_config_payload: dict[str, Any] | None = Field(default=None, alias="fit_config")
    acquisition_config_payload: dict[str, Any] = Field(default_factory=dict, alias="acquisition_config")
    optimize_config_payload: dict[str, Any] = Field(default_factory=dict, alias="optimize_config")
    data_context_payload: dict[str, Any] | None = Field(default=None, alias="data_context")
    tensor_options: TensorOptions = Field(default_factory=TensorOptions)


class SessionInfo(APIBaseModel):
    session_id: str
    task_type: str
    model_type: str
    input_type: str
    metadata: dict[str, Any]


class CandidateResponse(APIBaseModel):
    candidates: Any
    acq_value: Any
    session_id: str | None = None


class PredictResponse(APIBaseModel):
    task_type: str | None = None
    prediction_space: str | None = None
    variance_kind: str | None = None
    posterior: Any | None = None
    mean: Any | None = None
    variance: Any | None = None


class SessionStore:
    """Thread-safe in-memory store for fitted BayesianOptimizer sessions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, BayesianOptimizer] = {}

    def create(self, optimizer: BayesianOptimizer) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = optimizer
        return session_id

    def get(self, session_id: str) -> BayesianOptimizer:
        with self._lock:
            optimizer = self._sessions.get(session_id)
        if optimizer is None:
            raise KeyError(session_id)
        return optimizer

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)


SESSION_STORE = SessionStore()


def _torch_dtype(dtype: str) -> Any:
    import torch

    normalized = str(dtype).replace("torch.", "").lower()
    if normalized in {"float64", "double"}:
        return torch.double
    if normalized in {"float32", "float"}:
        return torch.float
    if normalized in {"int64", "long"}:
        return torch.long
    raise ValueError(f"Unsupported tensor dtype: {dtype!r}.")


def _to_tensor(value: Any, options: TensorOptions | None = None, *, dtype: str | None = None) -> Any:
    if value is None:
        return None

    import torch

    options = options or TensorOptions()
    tensor_dtype = _torch_dtype(dtype or options.dtype)
    device = options.device
    if isinstance(value, torch.Tensor):
        out = value.to(dtype=tensor_dtype)
        return out.to(device) if device is not None else out
    return torch.tensor(value, dtype=tensor_dtype, device=device)


def _to_long_tensor(value: Any, options: TensorOptions | None = None) -> Any:
    return _to_tensor(value, options, dtype="int64")


def _to_python(value: Any) -> Any:
    """Convert tensors, numpy arrays, and dataclasses into JSON-friendly objects."""
    if value is None:
        return None

    try:
        import torch

        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            if detached.ndim == 0:
                return detached.item()
            return detached.tolist()
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if is_dataclass(value):
        return _to_python(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_fixed_features(value: Any) -> dict[int, float] | None:
    if value is None:
        return None
    return {int(k): float(v) for k, v in dict(value).items()}


def _normalize_fixed_features_list(value: Any) -> list[dict[int, float]] | None:
    if value is None:
        return None
    return [_normalize_fixed_features(item) or {} for item in value]


def _normalize_linear_constraints(value: Any, options: TensorOptions) -> Any:
    """Convert JSON constraints to BoTorch-supported linear constraints.

    Accepted formats:
        [{"indices": [0, 1], "coefficients": [1.0, 1.0], "rhs": 1.0}]
        [[[0, 1], [1.0, 1.0], 1.0]]
    """
    if value is None:
        return None
    constraints = []
    for item in value:
        if isinstance(item, dict):
            indices = item["indices"]
            coefficients = item["coefficients"]
            rhs = item["rhs"]
        else:
            indices, coefficients, rhs = item
        constraints.append(
            (
                _to_long_tensor(indices, options),
                _to_tensor(coefficients, options),
                float(rhs),
            )
        )
    return constraints


def _config_dict(data: dict[str, Any] | None) -> dict[str, Any]:
    return dict(data or {})


def _build_input_transform_config(data: Any | None, options: TensorOptions | None = None) -> InputTransformConfig | None:
    if data is None or isinstance(data, InputTransformConfig):
        return data
    raw = dict(data)
    if raw.get("bounds") is not None:
        raw["bounds"] = _to_tensor(raw["bounds"], options)
    return InputTransformConfig(**raw)


def _build_fit_config(data: Any | None) -> FitConfig | None:
    if data is None or isinstance(data, FitConfig):
        return data
    return FitConfig(**dict(data))


def _build_output_config(data: Any, options: TensorOptions | None = None) -> Any:
    if isinstance(data, (str, ModelConfig, OutputConfig)):
        return data
    raw = dict(data)
    if raw.get("input_transform_config") is not None:
        raw["input_transform_config"] = _build_input_transform_config(raw["input_transform_config"], options)
    if raw.get("fit_config") is not None:
        raw["fit_config"] = _build_fit_config(raw["fit_config"])
    return OutputConfig(**raw)


def _build_multi_output_config(data: Any | None, options: TensorOptions | None = None) -> MultiOutputConfig | None:
    if data is None or isinstance(data, MultiOutputConfig):
        return data
    raw = dict(data)
    if raw.get("output_configs") is not None:
        raw["output_configs"] = [_build_output_config(item, options) for item in raw["output_configs"]]
    if raw.get("output_fit_configs") is not None:
        output_fit_configs = raw["output_fit_configs"]
        if isinstance(output_fit_configs, dict):
            raw["output_fit_configs"] = _build_fit_config(output_fit_configs)
        else:
            raw["output_fit_configs"] = [
                _build_fit_config(item) if item is not None else None
                for item in output_fit_configs
            ]
    return MultiOutputConfig(**raw)


def _build_model_config(data: dict[str, Any] | None, options: TensorOptions | None = None) -> ModelConfig:
    raw = _config_dict(data)
    if raw.get("input_transform_config") is not None:
        raw["input_transform_config"] = _build_input_transform_config(raw["input_transform_config"], options)
    if raw.get("multi_output_config") is not None:
        raw["multi_output_config"] = _build_multi_output_config(raw["multi_output_config"], options)
    return ModelConfig(**raw)


def _build_objective_config(data: Any | None) -> ObjectiveConfig | None:
    if data is None or isinstance(data, ObjectiveConfig):
        return data
    return ObjectiveConfig(**dict(data))


def _build_outcome_constraint_config(data: Any | None) -> OutcomeConstraintConfig | None:
    if data is None or isinstance(data, OutcomeConstraintConfig):
        return data
    return OutcomeConstraintConfig(**dict(data))


def _build_multi_objective_config(
    data: Any | None,
    options: TensorOptions,
) -> MultiObjectiveConfig | None:
    if data is None or isinstance(data, MultiObjectiveConfig):
        return data
    raw = dict(data)
    for key in ["ref_point", "Y_baseline", "objective_thresholds", "scalarization_weights"]:
        if raw.get(key) is not None:
            raw[key] = _to_tensor(raw[key], options)
    return MultiObjectiveConfig(**raw)


def _build_data_context(data: dict[str, Any] | None, options: TensorOptions) -> DataContext | None:
    if data is None:
        return None
    raw = dict(data)
    for key in [
        "bounds",
        "X_baseline",
        "X_pending",
        "Y_baseline",
        "best_f",
        "ref_point",
        "objective_thresholds",
        "mc_points",
    ]:
        if raw.get(key) is not None:
            raw[key] = _to_tensor(raw[key], options)
    if raw.get("multi_objective") is not None:
        raw["multi_objective"] = _build_multi_objective_config(raw["multi_objective"], options)
    return DataContext(**raw)


def _build_repair_config(data: Any | None, options: TensorOptions) -> CandidateRepairConfig | None:
    if data is None or isinstance(data, CandidateRepairConfig):
        return data
    raw = dict(data)
    if raw.get("bounds") is not None:
        raw["bounds"] = _to_tensor(raw["bounds"], options)
    if raw.get("steps") is not None:
        raw["steps"] = _to_tensor(raw["steps"], options)
    if raw.get("equality_constraints") is not None:
        raw["equality_constraints"] = _normalize_linear_constraints(raw["equality_constraints"], options)
    if raw.get("inequality_constraints") is not None:
        raw["inequality_constraints"] = _normalize_linear_constraints(raw["inequality_constraints"], options)
    if raw.get("fixed_features") is not None:
        raw["fixed_features"] = _normalize_fixed_features(raw["fixed_features"])
    return CandidateRepairConfig(**raw)


def _build_optimize_config(data: dict[str, Any] | None, options: TensorOptions) -> OptimizeConfig:
    raw = _config_dict(data)
    if raw.get("repair_config") is not None:
        raw["repair_config"] = _build_repair_config(raw["repair_config"], options)
    if raw.get("fixed_features") is not None:
        raw["fixed_features"] = _normalize_fixed_features(raw["fixed_features"])
    if raw.get("fixed_features_list") is not None:
        raw["fixed_features_list"] = _normalize_fixed_features_list(raw["fixed_features_list"])
    if raw.get("equality_constraints") is not None:
        raw["equality_constraints"] = _normalize_linear_constraints(raw["equality_constraints"], options)
    if raw.get("inequality_constraints") is not None:
        raw["inequality_constraints"] = _normalize_linear_constraints(raw["inequality_constraints"], options)
    return OptimizeConfig(**raw)


def _build_acquisition_config(data: dict[str, Any] | None) -> AcquisitionConfig:
    raw = _config_dict(data)
    if "name" not in raw:
        raise ValueError("acquisition_config.name is required.")
    if raw.get("objective_config") is not None:
        raw["objective_config"] = _build_objective_config(raw["objective_config"])
    if raw.get("outcome_constraint_config") is not None:
        raw["outcome_constraint_config"] = _build_outcome_constraint_config(raw["outcome_constraint_config"])
    return AcquisitionConfig(**raw)


def _make_session_info(session_id: str, bo: BayesianOptimizer) -> SessionInfo:
    bundle = bo.bundle
    if bundle is None:
        raise RuntimeError("Session has no fitted bundle.")
    return SessionInfo(
        session_id=session_id,
        task_type=str(bundle.task_type),
        model_type=str(bundle.model_type),
        input_type=str(bundle.input_type),
        metadata=_to_python(bundle.metadata),
    )


def _get_session(session_store: SessionStore, session_id: str) -> BayesianOptimizer:
    try:
        return session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}") from exc


def create_router(store: SessionStore | None = None) -> APIRouter:
    """Create a FastAPI router for bochan optimization sessions."""
    router = APIRouter(prefix="/bochan", tags=["bochan"])
    session_store = store or SESSION_STORE

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "sessions": len(session_store.list_ids())}

    @router.get("/acquisitions")
    def list_acquisitions() -> dict[str, Any]:
        from .acquisition_registry import available_acqf_names

        return {"names": available_acqf_names()}

    @router.get("/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"session_ids": session_store.list_ids()}

    @router.post("/sessions", response_model=SessionInfo)
    def create_session(request: FitSessionRequest) -> SessionInfo:
        try:
            train_X = _to_tensor(request.train_X, request.tensor_options)
            train_Y = _to_tensor(request.train_Y, request.tensor_options)
            bounds = _to_tensor(request.bounds, request.tensor_options) if request.bounds is not None else None
            model_config = _build_model_config(request.model_config_payload, request.tensor_options)
            fit_config = _build_fit_config(request.fit_config_payload)
            bo = BayesianOptimizer(
                model_config=model_config,
                fit_config=fit_config,
                bounds=bounds,
            )
            bo.fit(train_X, train_Y)
            session_id = session_store.create(bo)
            return _make_session_info(session_id, bo)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        deleted = session_store.delete(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
        return {"deleted": True, "session_id": session_id}

    @router.post("/sessions/{session_id}/predict", response_model=PredictResponse)
    def predict(session_id: str, request: PredictRequest) -> PredictResponse:
        bo = _get_session(session_store, session_id)
        try:
            X = _to_tensor(request.X, request.tensor_options)
            result = bo.predict(
                X,
                return_type=request.return_type,
                return_result=True,
                posterior_kwargs=request.posterior_kwargs,
            )
            mean = _to_python(result.mean)
            variance = _to_python(result.variance)
            posterior_payload = None
            if request.return_type == "posterior":
                posterior_payload = {
                    "type": type(result.posterior).__name__,
                    "mean": mean,
                    "variance": variance,
                }
            return PredictResponse(
                task_type=result.task_type,
                prediction_space=result.prediction_space,
                variance_kind=result.variance_kind,
                posterior=posterior_payload,
                mean=mean,
                variance=variance,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/candidate", response_model=CandidateResponse)
    def candidate(session_id: str, request: CandidateRequest) -> CandidateResponse:
        bo = _get_session(session_store, session_id)
        try:
            acq_config = _build_acquisition_config(request.acquisition_config_payload)
            opt_config = _build_optimize_config(request.optimize_config_payload, request.tensor_options)
            data_context = _build_data_context(request.data_context_payload, request.tensor_options)
            bounds = _to_tensor(request.bounds, request.tensor_options) if request.bounds is not None else None
            candidates, acq_value = bo.candidate(
                acq_config=acq_config,
                opt_config=opt_config,
                data_context=data_context,
                bounds=bounds,
            )
            return CandidateResponse(
                session_id=session_id,
                candidates=_to_python(candidates),
                acq_value=_to_python(acq_value),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/sessions/{session_id}/ask", response_model=CandidateResponse)
    def ask(session_id: str, request: CandidateRequest) -> CandidateResponse:
        return candidate(session_id, request)

    @router.post("/sessions/{session_id}/tell", response_model=SessionInfo)
    def tell(session_id: str, request: TellRequest) -> SessionInfo:
        bo = _get_session(session_store, session_id)
        try:
            new_X = _to_tensor(request.new_X, request.tensor_options)
            new_Y = _to_tensor(request.new_Y, request.tensor_options)
            fit_config = _build_fit_config(request.fit_config_payload)
            bo.tell(new_X, new_Y, refit=request.refit, fit_config=fit_config)
            return _make_session_info(session_id, bo)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/suggest", response_model=CandidateResponse)
    def suggest(request: SuggestRequest) -> CandidateResponse:
        try:
            train_X = _to_tensor(request.train_X, request.tensor_options)
            train_Y = _to_tensor(request.train_Y, request.tensor_options)
            bounds = _to_tensor(request.bounds, request.tensor_options)
            model_config = _build_model_config(request.model_config_payload, request.tensor_options)
            fit_config = _build_fit_config(request.fit_config_payload)
            acq_config = _build_acquisition_config(request.acquisition_config_payload)
            opt_config = _build_optimize_config(request.optimize_config_payload, request.tensor_options)
            data_context = _build_data_context(request.data_context_payload, request.tensor_options)

            bo = BayesianOptimizer(
                model_config=model_config,
                fit_config=fit_config,
                bounds=bounds,
            )
            bo.fit(train_X, train_Y)
            candidates, acq_value = bo.candidate(
                acq_config=acq_config,
                opt_config=opt_config,
                data_context=data_context,
            )
            return CandidateResponse(
                candidates=_to_python(candidates),
                acq_value=_to_python(acq_value),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


def create_app() -> FastAPI:
    """Create the default FastAPI application."""
    app = FastAPI(title="bochan API", version="0.1.0")
    app.include_router(create_router())
    return app


app = create_app()


__all__ = [
    "APIBaseModel",
    "CandidateRequest",
    "CandidateResponse",
    "FitSessionRequest",
    "PredictRequest",
    "PredictResponse",
    "SESSION_STORE",
    "SessionInfo",
    "SessionStore",
    "SuggestRequest",
    "TensorOptions",
    "TellRequest",
    "app",
    "create_app",
    "create_router",
]
