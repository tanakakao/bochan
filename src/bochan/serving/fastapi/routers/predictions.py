"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..converters import to_serializable, to_tensor
from ..dependencies import InMemoryOptimizerStore, get_optimizer_store
from ..schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/models", tags=["predictions"])


@router.post("/{model_id}/predict", response_model=PredictResponse)
def predict(
    model_id: str,
    request: PredictRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> PredictResponse:
    try:
        optimizer = store.get(model_id)
        X = to_tensor(request.X)
        result = optimizer.predict(
            X,
            return_type=request.return_type,
            posterior_kwargs=request.posterior_kwargs,
        )
        if request.return_type == "mean_variance":
            mean, variance = result
            return PredictResponse(
                model_id=model_id,
                mean=to_serializable(mean),
                variance=to_serializable(variance),
            )
        if request.return_type == "mean":
            return PredictResponse(model_id=model_id, mean=to_serializable(result), value=to_serializable(result))
        if request.return_type == "variance":
            return PredictResponse(model_id=model_id, variance=to_serializable(result), value=to_serializable(result))
        return PredictResponse(model_id=model_id, value=to_serializable(result))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
