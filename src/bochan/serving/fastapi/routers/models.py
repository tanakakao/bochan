"""Model fit / lifecycle endpoints."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from bochan.api import BayesianOptimizer
from bochan.llm import plan_configs

from ..converters import (
    model_metadata,
    to_acquisition_config,
    to_data_context,
    to_fit_config,
    to_model_config,
    to_optimize_config,
    to_serializable,
    to_tensor,
)
from ..dependencies import OptimizerStore, get_optimizer_store
from ..schemas import (
    AutoCandidateRequest,
    FitModelRequest,
    LLMPlanRequest,
    ModelDeleteResponse,
    ModelFitResponse,
    ModelListResponse,
    RefitModelRequest,
    TellRequest,
)

OPTIMIZER_STORE_DEP = Depends(get_optimizer_store)

router = APIRouter(prefix="/models", tags=["models"])


def _model_fit_response(model_id: str, optimizer: BayesianOptimizer) -> ModelFitResponse:
    train_X = getattr(optimizer, "train_X", None)
    n_train = int(train_X.shape[-2]) if hasattr(train_X, "shape") else None
    bundle = optimizer.bundle
    if bundle is None:
        raise RuntimeError("Optimizer has no fitted bundle.")
    return ModelFitResponse(
        model_id=model_id,
        task_type=str(bundle.task_type),
        model_type=str(bundle.model_type),
        n_train=n_train,
        metadata=model_metadata(optimizer),
    )


def _schema_to_dict(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _request_config(request: Any, name: str) -> dict[str, Any] | None:
    value = getattr(request, "bo_model_config", None) if name == "model_config" else getattr(request, name, None)
    return None if value is None else dict(value)


def _plan_from_request(request: LLMPlanRequest | AutoCandidateRequest, train_X: Any, train_Y: Any, bounds: Any) -> dict[str, Any]:
    return plan_configs(
        goal=request.goal,
        llm_config=_schema_to_dict(request.llm_config),
        llm_context=_schema_to_dict(request.llm_context),
        train_X=train_X,
        train_Y=train_Y,
        bounds=bounds,
        mode=getattr(request, "mode", "full"),
        planner_response=getattr(request, "planner_response", None),
        existing_model_config=_request_config(request, "model_config"),
        existing_fit_config=_request_config(request, "fit_config"),
        existing_acquisition_config=_request_config(request, "acquisition_config"),
        existing_optimize_config=_request_config(request, "optimize_config"),
    )


def _planned_config(plan: dict[str, Any], request: Any, name: str) -> dict[str, Any]:
    explicit = _request_config(request, name)
    if explicit is not None:
        return explicit
    value = plan.get(name)
    if value is None:
        raise ValueError(f"LLM plan did not include {name}.")
    return dict(value)


def _inject_llm_options(opt_config: object, request: AutoCandidateRequest) -> object:
    updates: dict[str, Any] = {"goal": request.goal}
    if request.llm_config is not None:
        updates["llm_config"] = _schema_to_dict(request.llm_config)
    if request.llm_context is not None:
        updates["llm_context"] = _schema_to_dict(request.llm_context)

    optimizer_kwargs = dict(getattr(opt_config, "optimizer_kwargs", {}) or {})
    for key, value in updates.items():
        optimizer_kwargs.setdefault(key, value)
    return replace(opt_config, optimizer_kwargs=optimizer_kwargs)


@router.post("", response_model=ModelFitResponse)
def fit_model(
    request: FitModelRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> ModelFitResponse:
    try:
        options = request.tensor_options
        train_X = to_tensor(request.train_X, options)
        train_Y = to_tensor(request.train_Y, options)
        bounds = to_tensor(request.bounds, options) if request.bounds is not None else None
        model_config = to_model_config(request.bo_model_config, options)
        fit_config = to_fit_config(request.fit_config)
        data_context = to_data_context(request.data_context, options) if request.data_context is not None else None

        optimizer = BayesianOptimizer(
            model_config=model_config,
            fit_config=fit_config,
            bounds=bounds,
            data_context=data_context,
        )
        optimizer.fit(train_X, train_Y)
        model_id = store.add(optimizer)
        return _model_fit_response(model_id, optimizer)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/plan")
def plan_model_config(request: LLMPlanRequest) -> dict[str, Any]:
    """Infer model / fit / acquisition / optimize configs without fitting a model."""

    try:
        options = request.tensor_options
        train_X = to_tensor(request.train_X, options) if request.train_X is not None else None
        train_Y = to_tensor(request.train_Y, options) if request.train_Y is not None else None
        bounds = to_tensor(request.bounds, options) if request.bounds is not None else None
        plan = _plan_from_request(request, train_X, train_Y, bounds)
        return {"plan": plan}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/auto-candidates")
def auto_candidates(
    request: AutoCandidateRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> dict[str, Any]:
    """Infer configs, fit a model, and generate candidates in one request."""

    try:
        options = request.tensor_options
        train_X = to_tensor(request.train_X, options)
        train_Y = to_tensor(request.train_Y, options)
        bounds = to_tensor(request.bounds, options) if request.bounds is not None else None
        plan = _plan_from_request(request, train_X, train_Y, bounds)

        model_config = to_model_config(_planned_config(plan, request, "model_config"), options)
        fit_config = to_fit_config(_planned_config(plan, request, "fit_config"))
        data_context = to_data_context(request.data_context, options) if request.data_context is not None else None

        optimizer = BayesianOptimizer(
            model_config=model_config,
            fit_config=fit_config,
            bounds=bounds,
            data_context=data_context,
        )
        optimizer.fit(train_X, train_Y)
        model_id = store.add(optimizer)

        acq_config = to_acquisition_config(_planned_config(plan, request, "acquisition_config"), options)
        opt_config = to_optimize_config(_planned_config(plan, request, "optimize_config"), options)
        opt_config = _inject_llm_options(opt_config, request)
        candidates, acq_value = optimizer.candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
            bounds=bounds,
        )
        model_response = _model_fit_response(model_id, optimizer)
        return {
            "model": model_response.model_dump(),
            "candidate": {
                "model_id": model_id,
                "candidates": to_serializable(candidates),
                "acq_value": to_serializable(acq_value),
            },
            "plan": plan,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=ModelListResponse)
def list_models(store: OptimizerStore = OPTIMIZER_STORE_DEP) -> ModelListResponse:
    return ModelListResponse(model_ids=store.list_ids())


@router.post("/{model_id}/refit", response_model=ModelFitResponse)
def refit_model(
    model_id: str,
    request: RefitModelRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> ModelFitResponse:
    try:
        optimizer = store.get(model_id)
        fit_config = to_fit_config(request.fit_config) if request.fit_config is not None else None
        optimizer.refit(fit_config=fit_config)
        return _model_fit_response(model_id, optimizer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/tell", response_model=ModelFitResponse)
def tell_model(
    model_id: str,
    request: TellRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> ModelFitResponse:
    try:
        optimizer = store.get(model_id)
        options = request.tensor_options
        new_X = to_tensor(request.new_X, options)
        new_Y = to_tensor(request.new_Y, options)
        fit_config = to_fit_config(request.fit_config) if request.fit_config is not None else None
        optimizer.tell(new_X, new_Y, refit=request.refit, fit_config=fit_config)
        return _model_fit_response(model_id, optimizer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{model_id}", response_model=ModelDeleteResponse)
def delete_model(
    model_id: str,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> ModelDeleteResponse:
    try:
        store.delete(model_id)
        return ModelDeleteResponse(model_id=model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
