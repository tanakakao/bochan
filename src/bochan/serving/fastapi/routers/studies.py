"""Stateful BochanStudy endpoints for ask/tell optimization workflows."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from bochan.api import BochanStudy

from ..converters import to_data_context, to_optimize_config, to_serializable, to_tensor
from ..dependencies import StudyStore, get_study_store
from ..schemas.study import (
    StudyAskRequest,
    StudyAskResponse,
    StudyBestResponse,
    StudyCreateRequest,
    StudyDeleteResponse,
    StudyFailedRequest,
    StudyHistoryResponse,
    StudyListResponse,
    StudyObservationRequest,
    StudyParetoRequest,
    StudyParetoResponse,
    StudyRestoreRequest,
    StudySnapshotResponse,
    StudySummaryResponse,
    StudyTellRequest,
    StudyTrialIdsRequest,
    StudyTrialsResponse,
)
from ..study_service import (
    acquisition_config,
    build_study,
    history_records,
    pareto_records,
    restore_trials,
    summary,
)

STUDY_STORE_DEP = Depends(get_study_store)
router = APIRouter(prefix="/studies", tags=["studies"])


def _handle(operation):
    try:
        return operation()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("", response_model=StudySummaryResponse)
def create_study(
    request: StudyCreateRequest,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudySummaryResponse:
    def operation() -> StudySummaryResponse:
        study = build_study(request)
        study_id = store.add(study)
        return summary(study_id, study)

    return _handle(operation)


@router.post("/restore", response_model=StudySummaryResponse)
def restore_study(
    request: StudyRestoreRequest,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudySummaryResponse:
    def operation() -> StudySummaryResponse:
        snapshot = dict(request.snapshot)
        metadata = dict(snapshot.get("metadata") or {})
        metadata.update(request.metadata)
        study = build_study(request, metadata=metadata)
        restore_trials(study, snapshot)
        study_id = store.add(study)
        return summary(study_id, study)

    return _handle(operation)


@router.get("", response_model=StudyListResponse)
def list_studies(store: StudyStore = STUDY_STORE_DEP) -> StudyListResponse:
    return StudyListResponse(study_ids=store.list_ids())


@router.get("/{study_id}", response_model=StudySummaryResponse)
def get_study(study_id: str, store: StudyStore = STUDY_STORE_DEP) -> StudySummaryResponse:
    return _handle(lambda: store.call(study_id, lambda study: summary(study_id, study)))


@router.post("/{study_id}/observations", response_model=StudySummaryResponse)
def add_observations(
    study_id: str,
    request: StudyObservationRequest,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudySummaryResponse:
    def operation(study: BochanStudy) -> StudySummaryResponse:
        study.add_observations(
            to_tensor(request.X, request.tensor_options),
            to_tensor(request.Y, request.tensor_options),
            metadata=request.metadata,
        )
        return summary(study_id, study)

    return _handle(lambda: store.call(study_id, operation))


@router.post("/{study_id}/ask", response_model=StudyAskResponse)
def ask_study(
    study_id: str,
    request: StudyAskRequest,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudyAskResponse:
    def operation(study: BochanStudy) -> StudyAskResponse:
        batch = study.ask(
            q=request.q,
            acq_config=acquisition_config(request.acq_config, request.tensor_options),
            opt_config=(
                to_optimize_config(request.opt_config, request.tensor_options)
                if request.opt_config is not None
                else None
            ),
            data_context=(
                to_data_context(request.data_context, request.tensor_options)
                if request.data_context is not None
                else None
            ),
            mark_running=request.mark_running,
            return_batch=True,
            fit=request.fit,
        )
        return StudyAskResponse(
            study_id=study_id,
            trial_ids=batch.trial_ids,
            candidates=to_serializable(batch.candidates),
            acq_value=to_serializable(batch.acq_value),
        )

    return _handle(lambda: store.call(study_id, operation))


@router.post("/{study_id}/tell", response_model=StudySummaryResponse)
def tell_study(
    study_id: str,
    request: StudyTellRequest,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudySummaryResponse:
    def operation(study: BochanStudy) -> StudySummaryResponse:
        study.tell(
            trial_ids=request.trial_ids,
            values=to_tensor(request.values, request.tensor_options),
            state=request.state,
            metadata=request.metadata,
        )
        if request.check_early_stop:
            study.check_early_stop(trial_ids=request.trial_ids, update=True)
        return summary(study_id, study)

    return _handle(lambda: store.call(study_id, operation))


@router.post("/{study_id}/trials/running", response_model=StudySummaryResponse)
def mark_running(
    study_id: str,
    request: StudyTrialIdsRequest,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudySummaryResponse:
    def operation(study: BochanStudy) -> StudySummaryResponse:
        study.mark_running(request.trial_ids)
        return summary(study_id, study)

    return _handle(lambda: store.call(study_id, operation))


@router.post("/{study_id}/trials/failed", response_model=StudySummaryResponse)
def mark_failed(
    study_id: str,
    request: StudyFailedRequest,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudySummaryResponse:
    def operation(study: BochanStudy) -> StudySummaryResponse:
        study.mark_failed(request.trial_ids, reason=request.reason)
        return summary(study_id, study)

    return _handle(lambda: store.call(study_id, operation))


@router.get("/{study_id}/trials", response_model=StudyTrialsResponse)
def get_trials(study_id: str, store: StudyStore = STUDY_STORE_DEP) -> StudyTrialsResponse:
    def operation(study: BochanStudy) -> StudyTrialsResponse:
        return StudyTrialsResponse(
            study_id=study_id,
            trials=to_serializable([trial.to_dict() for trial in study.trials]),
        )

    return _handle(lambda: store.call(study_id, operation))


@router.get("/{study_id}/best", response_model=StudyBestResponse)
def get_best(
    study_id: str,
    output_index: int = 0,
    direction: Literal["maximize", "minimize"] | None = None,
    param_names: Annotated[list[str] | None, Query()] = None,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudyBestResponse:
    def operation(study: BochanStudy) -> StudyBestResponse:
        result = study.best_result(
            output_index=output_index,
            direction=direction,
            param_names=param_names,
        )
        result["x"] = study.get_best_x(output_index=output_index, direction=direction)
        return StudyBestResponse(study_id=study_id, result=to_serializable(result))

    return _handle(lambda: store.call(study_id, operation))


@router.get("/{study_id}/history", response_model=StudyHistoryResponse)
def get_history(
    study_id: str,
    output_index: int = 0,
    direction: Literal["maximize", "minimize"] | None = None,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudyHistoryResponse:
    def operation(study: BochanStudy) -> StudyHistoryResponse:
        resolved, records = history_records(
            study,
            output_index=output_index,
            direction=direction,
        )
        return StudyHistoryResponse(
            study_id=study_id,
            output_index=output_index,
            direction=resolved,
            records=records,
        )

    return _handle(lambda: store.call(study_id, operation))


@router.post("/{study_id}/pareto", response_model=StudyParetoResponse)
def get_pareto(
    study_id: str,
    request: StudyParetoRequest,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudyParetoResponse:
    def operation(study: BochanStudy) -> StudyParetoResponse:
        indices, directions, pareto, records = pareto_records(
            study,
            output_indices=request.output_indices,
            directions=request.directions,
        )
        return StudyParetoResponse(
            study_id=study_id,
            output_indices=indices,
            directions=directions,
            pareto_trials=to_serializable([trial.to_dict() for trial in pareto]),
            trials=records,
        )

    return _handle(lambda: store.call(study_id, operation))


@router.get("/{study_id}/snapshot", response_model=StudySnapshotResponse)
def get_snapshot(
    study_id: str,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudySnapshotResponse:
    def operation(study: BochanStudy) -> StudySnapshotResponse:
        return StudySnapshotResponse(
            study_id=study_id,
            snapshot=to_serializable(study.to_snapshot()),
        )

    return _handle(lambda: store.call(study_id, operation))


@router.delete("/{study_id}", response_model=StudyDeleteResponse)
def delete_study(
    study_id: str,
    store: StudyStore = STUDY_STORE_DEP,
) -> StudyDeleteResponse:
    def operation() -> StudyDeleteResponse:
        store.delete(study_id)
        return StudyDeleteResponse(study_id=study_id)

    return _handle(operation)
