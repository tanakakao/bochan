"""Material model-axis discovery, validation, and execution endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from bochan.api import BayesianOptimizer, material_task_fixed_features
from bochan.models.regression.gaussian.materials import (
    list_material_families,
    material_model_axes_capabilities,
)

from ..converters import model_metadata, to_data_context, to_fit_config, to_tensor
from ..dependencies import OptimizerStore, get_optimizer_store
from ..schemas.material_model_axes import (
    MaterialExplicitTaskRequest,
    MaterialModelAxesCapabilitiesResponse,
    MaterialModelAxesCatalogResponse,
    MaterialModelAxesRequest,
    MaterialModelAxesResponse,
    MaterialModelFitRequest,
    MaterialTaskFixedFeaturesRequest,
    MaterialTaskFixedFeaturesResponse,
)
from ..schemas.responses import ModelFitResponse
from ..services.material_models import (
    bind_material_model_spec,
    to_material_api_spec,
    to_material_model_config,
)

OPTIMIZER_STORE_DEP = Depends(get_optimizer_store)

router = APIRouter(prefix="/materials/models", tags=["materials"])


def _normalized_response(request: MaterialModelAxesRequest) -> MaterialModelAxesResponse:
    spec = to_material_api_spec(request)
    payload = spec.as_dict()
    task = None
    if spec.axes.task_mode == "explicit":
        task = MaterialExplicitTaskRequest(
            task_feature=spec.task_feature,
            all_tasks=None if spec.all_tasks is None else list(spec.all_tasks),
            output_tasks=None if spec.output_tasks is None else list(spec.output_tasks),
        )
    return MaterialModelAxesResponse(
        family=payload["family"],
        domain=payload["domain"],
        kind=payload["kind"],
        input_mode=payload["input_mode"],
        output_mode=payload["output_mode"],
        task_mode=payload["task_mode"],
        fidelity_mode=payload["fidelity_mode"],
        route=payload["route"],
        implemented=payload["implemented"],
        cat_dims=list(request.cat_dims or []),
        task=task,
    )


@router.get("/capabilities", response_model=MaterialModelAxesCatalogResponse)
def get_material_model_axes_catalog() -> MaterialModelAxesCatalogResponse:
    """Return model-axis capabilities for all registered material families."""

    families = [
        MaterialModelAxesCapabilitiesResponse.model_validate(
            material_model_axes_capabilities(family)
        )
        for family in list_material_families()
    ]
    return MaterialModelAxesCatalogResponse(families=families)


@router.get(
    "/capabilities/{family}",
    response_model=MaterialModelAxesCapabilitiesResponse,
)
def get_material_model_axes_capabilities(
    family: str,
) -> MaterialModelAxesCapabilitiesResponse:
    """Return model-axis capabilities for one registered material family."""

    try:
        payload = material_model_axes_capabilities(family)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MaterialModelAxesCapabilitiesResponse.model_validate(payload)


@router.post("/validate", response_model=MaterialModelAxesResponse)
def validate_material_model_axes(
    request: MaterialModelAxesRequest,
) -> MaterialModelAxesResponse:
    """Normalize and validate a JSON material model-axis request."""

    try:
        response = _normalized_response(request)
        if response.input_mode == "mixed" and not response.cat_dims:
            raise ValueError("cat_dims is required when input_mode='mixed'.")
        if response.input_mode == "continuous" and response.cat_dims:
            raise ValueError("cat_dims must be omitted when input_mode='continuous'.")
        if response.task_mode == "none" and request.task is not None:
            raise ValueError("task is only valid when task_mode='explicit'.")
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return response


@router.post("/fit", response_model=ModelFitResponse)
def fit_material_model(
    request: MaterialModelFitRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> ModelFitResponse:
    """Fit a material-axis surrogate and register it in the shared model store."""

    try:
        options = request.tensor_options
        train_X = to_tensor(request.train_X, options)
        train_Y = to_tensor(request.train_Y, options)
        train_Yvar = (
            to_tensor(request.train_Yvar, options)
            if request.train_Yvar is not None
            else None
        )
        bounds = (
            to_tensor(request.bounds, options)
            if request.bounds is not None
            else None
        )
        data_context = (
            to_data_context(request.data_context, options)
            if request.data_context is not None
            else None
        )
        model_config, spec = to_material_model_config(request.model)
        optimizer = BayesianOptimizer(
            model_config=model_config,
            fit_config=to_fit_config(request.fit_config),
            bounds=bounds,
            data_context=data_context,
        )
        optimizer.fit(train_X, train_Y, train_Yvar)
        bind_material_model_spec(optimizer, spec, input_dim=int(train_X.shape[-1]))
        model_id = store.add(optimizer)
        bundle = optimizer.bundle
        if bundle is None:
            raise RuntimeError("Optimizer has no fitted bundle.")
        metadata = model_metadata(optimizer)
        metadata["material_model_axes"] = spec.as_dict()
        return ModelFitResponse(
            model_id=model_id,
            task_type=str(bundle.task_type),
            model_type=str(bundle.model_type),
            n_train=int(train_X.shape[-2]),
            metadata=metadata,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/task-fixed-features",
    response_model=MaterialTaskFixedFeaturesResponse,
)
def get_material_task_fixed_features(
    request: MaterialTaskFixedFeaturesRequest,
) -> MaterialTaskFixedFeaturesResponse:
    """Resolve fixed features for BO over one explicit material task."""

    try:
        spec = to_material_api_spec(request.model)
        fixed_features = material_task_fixed_features(
            spec,
            request.target_task,
            input_dim=request.input_dim,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MaterialTaskFixedFeaturesResponse(fixed_features=fixed_features)


__all__ = ["router"]
