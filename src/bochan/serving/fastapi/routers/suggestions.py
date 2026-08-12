"""Stateless candidate suggestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from bochan.api import BayesianOptimizer

from ..converters import (
    to_data_context,
    to_fit_config,
    to_optimize_config,
    to_serializable,
    to_tensor,
)
from ..schemas import CandidateResponse, SuggestRequest
from ..tabular_compat import (
    bind_category_metadata,
    to_acquisition_config,
    to_model_config_with_metadata,
    to_target_tensor,
)

router = APIRouter(tags=["suggestions"])


@router.post("/suggest", response_model=CandidateResponse)
def suggest(request: SuggestRequest) -> CandidateResponse:
    """Fit a temporary model and return candidates without storing state."""

    try:
        options = request.tensor_options
        train_X = to_tensor(request.train_X, options)
        bounds = to_tensor(request.bounds, options)
        model_config, category_metadata = to_model_config_with_metadata(
            request.bo_model_config,
            options,
        )
        train_Y = to_target_tensor(
            request.train_Y,
            options,
            metadata=category_metadata,
        )
        fit_config = to_fit_config(request.fit_config)
        data_context = (
            to_data_context(request.data_context, options)
            if request.data_context is not None
            else None
        )
        opt_config = to_optimize_config(request.optimize_config, options)

        optimizer = BayesianOptimizer(
            model_config=model_config,
            fit_config=fit_config,
            bounds=bounds,
            data_context=data_context,
        )
        optimizer.fit(train_X, train_Y)
        bind_category_metadata(optimizer, category_metadata)
        acq_config = to_acquisition_config(
            request.acquisition_config,
            options,
            optimizer=optimizer,
        )
        candidates, acq_value = optimizer.candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
            bounds=bounds,
        )
        return CandidateResponse(
            model_id="stateless",
            candidates=to_serializable(candidates),
            acq_value=to_serializable(acq_value),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
