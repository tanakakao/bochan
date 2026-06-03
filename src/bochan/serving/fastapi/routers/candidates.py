"""Candidate generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..converters import to_acquisition_config, to_data_context, to_optimize_config, to_serializable, to_tensor
from ..dependencies import InMemoryOptimizerStore, get_optimizer_store
from ..schemas import CandidateRequest, CandidateResponse

router = APIRouter(prefix="/models", tags=["candidates"])


@router.post("/{model_id}/candidates", response_model=CandidateResponse)
def generate_candidates(
    model_id: str,
    request: CandidateRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> CandidateResponse:
    try:
        optimizer = store.get(model_id)
        acq_config = to_acquisition_config(request.acq_config)
        opt_config = to_optimize_config(request.opt_config)
        data_context = to_data_context(request.data_context) if request.data_context is not None else None
        bounds = to_tensor(request.bounds) if request.bounds is not None else None
        candidates, acq_value = optimizer.candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
            bounds=bounds,
        )
        return CandidateResponse(
            model_id=model_id,
            candidates=to_serializable(candidates),
            acq_value=to_serializable(acq_value),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
