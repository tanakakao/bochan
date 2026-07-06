"""Candidate generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..converters import to_acquisition_config, to_data_context, to_optimize_config, to_serializable, to_tensor
from ..dependencies import InMemoryOptimizerStore, get_optimizer_store
from ..schemas import CandidateRequest, CandidateResponse, CompareCandidatesRequest, CompareCandidatesResponse

router = APIRouter(prefix="/models", tags=["candidates"])


def _candidate_response(model_id: str, candidates: object, acq_value: object) -> CandidateResponse:
    return CandidateResponse(
        model_id=model_id,
        candidates=to_serializable(candidates),
        acq_value=to_serializable(acq_value),
    )


@router.post("/{model_id}/candidates", response_model=CandidateResponse)
def generate_candidates(
    model_id: str,
    request: CandidateRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> CandidateResponse:
    try:
        optimizer = store.get(model_id)
        options = request.tensor_options
        acq_config = to_acquisition_config(request.acq_config, options)
        opt_config = to_optimize_config(request.opt_config, options)
        data_context = to_data_context(request.data_context, options) if request.data_context is not None else None
        bounds = to_tensor(request.bounds, options) if request.bounds is not None else None
        candidates, acq_value = optimizer.candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
            bounds=bounds,
        )
        return _candidate_response(model_id, candidates, acq_value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/ask", response_model=CandidateResponse)
def ask_candidates(
    model_id: str,
    request: CandidateRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> CandidateResponse:
    try:
        optimizer = store.get(model_id)
        options = request.tensor_options
        acq_config = to_acquisition_config(request.acq_config, options)
        opt_config = to_optimize_config(request.opt_config, options)
        data_context = to_data_context(request.data_context, options) if request.data_context is not None else None
        bounds = to_tensor(request.bounds, options) if request.bounds is not None else None
        candidates, acq_value = optimizer.ask(
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
            bounds=bounds,
        )
        return _candidate_response(model_id, candidates, acq_value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/candidates/compare", response_model=CompareCandidatesResponse)
def compare_candidates(
    model_id: str,
    request: CompareCandidatesRequest,
    store: InMemoryOptimizerStore = Depends(get_optimizer_store),
) -> CompareCandidatesResponse:
    try:
        optimizer = store.get(model_id)
        options = request.tensor_options
        acq_configs = [to_acquisition_config(config, options) for config in request.acq_configs]
        opt_config = to_optimize_config(request.opt_config, options)
        data_context = to_data_context(request.data_context, options) if request.data_context is not None else None
        bounds = to_tensor(request.bounds, options) if request.bounds is not None else None
        results = optimizer.compare_acquisitions(
            acq_configs=acq_configs,
            opt_config=opt_config,
            data_context=data_context,
            bounds=bounds,
        )
        payload = {
            name: _candidate_response(model_id, result.candidates, result.acq_value)
            for name, result in results.items()
        }
        return CompareCandidatesResponse(model_id=model_id, results=payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
