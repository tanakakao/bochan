"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..converters import to_serializable, to_tensor
from ..dependencies import OptimizerStore, get_optimizer_store
from ..schemas import PredictRequest, PredictResponse

OPTIMIZER_STORE_DEP = Depends(get_optimizer_store)

router = APIRouter(prefix="/models", tags=["predictions"])


@router.post("/{model_id}/predict", response_model=PredictResponse)
def predict(
    model_id: str,
    request: PredictRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> PredictResponse:
    try:
        optimizer = store.get(model_id)
        X = to_tensor(request.X, request.tensor_options)
        result = optimizer.predict(
            X,
            return_type=request.return_type,
            return_result=True,
            posterior_kwargs=request.posterior_kwargs,
        )

        common = {
            "model_id": model_id,
            "task_type": result.task_type,
            "prediction_space": result.prediction_space,
            "variance_kind": result.variance_kind,
        }
        mean = to_serializable(result.mean)
        variance = to_serializable(result.variance)

        if request.return_type == "posterior":
            summary = {
                "type": type(result.posterior).__name__,
                "mean": mean,
                "variance": variance,
            }
            return PredictResponse(
                **common,
                posterior=summary,
                mean=mean,
                variance=variance,
                value=summary,
            )
        if request.return_type == "mean_variance":
            return PredictResponse(**common, mean=mean, variance=variance)
        if request.return_type == "mean":
            return PredictResponse(**common, mean=mean, value=mean)
        if request.return_type == "variance":
            return PredictResponse(**common, variance=variance, value=variance)
        raise ValueError(f"Unsupported return_type: {request.return_type!r}.")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
