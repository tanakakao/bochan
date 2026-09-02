"""Candidate generation HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from bochan.api.acquisition.diagnostics import candidate_acquisition_diagnostics

from ..converters import to_serializable
from ..dependencies import OptimizerStore, get_optimizer_store
from ..schemas import (
    AcquisitionDiagnosticsResponse,
    CandidateRequest,
    CandidateResponse,
    CompareCandidatesRequest,
    CompareCandidatesResponse,
)
from ..services.candidates import (
    compare_candidate_results,
    generate_candidate_result,
)

OPTIMIZER_STORE_DEP = Depends(get_optimizer_store)

router = APIRouter(prefix="/models", tags=["candidates"])


def _candidate_response(
    model_id: str,
    candidates: object,
    acq_value: object,
    *,
    diagnostics: object | None = None,
) -> CandidateResponse:
    return CandidateResponse(
        model_id=model_id,
        candidates=to_serializable(candidates),
        acq_value=to_serializable(acq_value),
        diagnostics=(
            None if diagnostics is None else to_serializable(diagnostics)
        ),
    )


def _optimizer_or_404(store: OptimizerStore, model_id: str):
    try:
        return store.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/{model_id}/acquisition/diagnostics",
    response_model=AcquisitionDiagnosticsResponse,
)
def acquisition_diagnostics(
    model_id: str,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> AcquisitionDiagnosticsResponse:
    """Return the latest acquisition diagnostics and current observation report."""

    optimizer = _optimizer_or_404(store, model_id)
    diagnostics = getattr(optimizer, "last_acquisition_diagnostics", None)
    observations = getattr(optimizer, "observations", None)
    observation_report = None
    if observations is not None:
        try:
            observation_report = observations.report()
        except Exception:
            observation_report = None
    return AcquisitionDiagnosticsResponse(
        model_id=model_id,
        diagnostics=(
            None if diagnostics is None else to_serializable(diagnostics)
        ),
        observation_report=(
            None
            if observation_report is None
            else to_serializable(observation_report)
        ),
    )


@router.post("/{model_id}/candidates", response_model=CandidateResponse)
def generate_candidates(
    model_id: str,
    request: CandidateRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> CandidateResponse:
    optimizer = _optimizer_or_404(store, model_id)
    try:
        result = generate_candidate_result(
            optimizer,
            request,
            return_result=True,
        )
        return _candidate_response(
            model_id,
            result.candidates,
            result.acq_value,
            diagnostics=candidate_acquisition_diagnostics(result),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{model_id}/ask", response_model=CandidateResponse)
def ask_candidates(
    model_id: str,
    request: CandidateRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> CandidateResponse:
    optimizer = _optimizer_or_404(store, model_id)
    try:
        result = generate_candidate_result(
            optimizer,
            request,
            use_ask=True,
            return_result=True,
        )
        return _candidate_response(
            model_id,
            result.candidates,
            result.acq_value,
            diagnostics=candidate_acquisition_diagnostics(result),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/{model_id}/candidates/compare",
    response_model=CompareCandidatesResponse,
)
def compare_candidates(
    model_id: str,
    request: CompareCandidatesRequest,
    store: OptimizerStore = OPTIMIZER_STORE_DEP,
) -> CompareCandidatesResponse:
    optimizer = _optimizer_or_404(store, model_id)
    try:
        results = compare_candidate_results(optimizer, request)
        payload = {
            name: _candidate_response(
                model_id,
                result.candidates,
                result.acq_value,
                diagnostics=candidate_acquisition_diagnostics(result),
            )
            for name, result in results.items()
        }
        return CompareCandidatesResponse(model_id=model_id, results=payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
