"""FastAPI application service for tabular ALIGNN models."""

from __future__ import annotations

import os
from tempfile import NamedTemporaryFile
from typing import Any

from bochan.structure import ALIGNNGraphBuilder, StructureAdapter
from bochan.tabular import TabularBayesianOptimizer

from ..converters import to_serializable
from ..schemas.alignn_tabular import (
    ALIGNNTabularCandidateRequest,
    ALIGNNTabularFitModelRequest,
    CrystalStructureRequest,
)
from ..schemas.tabular import (
    TabularCandidateResponse,
    TabularModelFitResponse,
    TabularPredictRequest,
    TabularPredictResponse,
)
from .tabular import (
    _candidate_direct_kwargs,
    _candidate_optimize_config,
    _experiment_failure_config,
    _frame_records,
    _schema_dict,
    build_fit_response,
    to_dataframe,
)


def _inline_file_structure(
    payload: CrystalStructureRequest,
    *,
    adapter: StructureAdapter,
) -> Any:
    """Parse inline CIF/POSCAR text without accepting client-controlled paths."""

    if payload.content is None:
        raise ValueError(f"format={payload.format!r} requires content.")
    suffix = ".cif" if payload.format == "cif" else ".vasp"
    path: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            encoding="utf-8",
            newline="\n",
            delete=False,
        ) as handle:
            handle.write(payload.content)
            path = handle.name
        return adapter.from_file(path, file_format=payload.format)
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def structure_catalog_from_request(
    request: ALIGNNTabularFitModelRequest,
) -> dict[str, Any]:
    """Convert HTTP structure payloads to the canonical in-memory catalog."""

    adapter = StructureAdapter()
    catalog: dict[str, Any] = {}
    for structure_id, payload in request.structure_catalog.items():
        if payload.format == "mapping":
            catalog[structure_id] = payload.as_mapping()
        else:
            catalog[structure_id] = _inline_file_structure(payload, adapter=adapter)
    return catalog


def graph_builder_from_request(
    request: ALIGNNTabularFitModelRequest,
) -> ALIGNNGraphBuilder:
    """Build the Phase-2 ALIGNN graph adapter from JSON-safe graph settings."""

    config = request.structure_graph_config.model_dump()
    return ALIGNNGraphBuilder(**config)


def _normalize_structure_column(frame: Any, structure_col: str) -> Any:
    """Normalize HTTP structure IDs to JSON object-key strings."""

    if structure_col not in frame.columns:
        raise ValueError(f"Missing structure column {structure_col!r}.")
    frame = frame.copy()
    frame[structure_col] = frame[structure_col].map(str)
    return frame


def fit_alignn_tabular_optimizer(
    request: ALIGNNTabularFitModelRequest,
) -> TabularBayesianOptimizer:
    """Fit one ALIGNN tabular optimizer from JSON data and crystal structures."""

    frame = _normalize_structure_column(to_dataframe(request.data), request.structure_col)
    cv_config = _schema_dict(request.cv_config)
    if cv_config and cv_config.get("splitter") == "stratified_kfold":
        cv_config["splitter"] = "stratified"
    if (
        cv_config
        and cv_config.get("splitter") != "loo"
        and int(cv_config["n_splits"]) > len(frame)
    ):
        raise ValueError("n_splits must not exceed the number of data rows.")

    structure_catalog = structure_catalog_from_request(request)
    graph_builder = graph_builder_from_request(request)
    optimizer = TabularBayesianOptimizer(
        model_config=_schema_dict(request.bo_model_config),
        fit_config=_schema_dict(request.fit_config),
        input_cols=request.input_cols,
        target_cols=request.target_cols,
        categorical_cols=request.categorical_cols,
        target_categorical_cols=request.target_categorical_cols,
        bounds=request.bounds,
        structure_col=request.structure_col,
        structure_catalog=structure_catalog,
        structure_graph_builder=graph_builder,
        dtype=request.dtype,
        device=request.device,
        dropna=request.dropna,
        missing_strategy=request.missing_strategy,
        target_missing_strategy=request.target_missing_strategy,
        experiment_status_col=request.experiment_status_col,
        failure_config=_experiment_failure_config(request),
        continuous_impute_strategy=request.continuous_impute_strategy,
        categorical_impute_strategy=request.categorical_impute_strategy,
        impute_targets=request.impute_targets,
        impute_random_state=request.impute_random_state,
        impute_max_iter=request.impute_max_iter,
        multiple_impute_sample_posterior=request.multiple_impute_sample_posterior,
        encode_categories=request.encode_categories,
        category_maps=request.category_maps,
        target_category_maps=request.target_category_maps,
        return_original_categories=request.return_original_categories,
        cross_validation=request.cross_validation,
        cv_config=cv_config,
    )
    optimizer.fit(frame)
    return optimizer


def build_alignn_fit_response(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
) -> TabularModelFitResponse:
    """Serialize the fitted model together with its structure/encoder contract."""

    response = build_fit_response(model_id, optimizer)
    bundle = optimizer.bo.bundle
    if bundle is None:
        return response
    model = bundle.model
    material_encoder = getattr(model, "material_encoder", None)
    trainable_layers = getattr(model, "trainable_encoder_layers", None)
    model_type = str(bundle.model_type)
    if model_type == "alignn_gp":
        training_mode = "frozen"
    elif trainable_layers == "all":
        training_mode = "full"
    else:
        training_mode = "partial"
    initialization = getattr(material_encoder, "initialization", None)
    graph_builder = getattr(optimizer.structure, "graph_builder", None)
    graph_config = getattr(graph_builder, "config", None)
    metadata = dict(response.metadata)
    metadata["alignn"] = to_serializable(
        {
            "encoder_training": training_mode,
            "encoder_initialization": initialization,
            "checkpoint_configured": initialization == "checkpoint",
            "structure_col": optimizer.structure.column,
            "structure_ids": list(optimizer.structure.structure_ids),
            "num_structures": optimizer.structure.num_structures,
            "process_dim": getattr(model, "process_dim", None),
            "graph_config": graph_config,
        }
    )
    response.metadata = metadata
    return response


def alignn_predict_response(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
    request: TabularPredictRequest,
) -> TabularPredictResponse:
    """Predict with the fitted structure-ID mapping preserved over JSON."""

    frame = _normalize_structure_column(
        to_dataframe(request.data),
        str(optimizer.structure.column),
    )
    value = optimizer.predict(
        frame,
        return_type=request.return_type,
        include_input=request.include_input,
        posterior_kwargs=request.posterior_kwargs,
    )
    if hasattr(value, "columns") and hasattr(value, "to_json"):
        columns, records = _frame_records(value)
        return TabularPredictResponse(
            model_id=model_id,
            columns=columns,
            records=records,
        )
    return TabularPredictResponse(
        model_id=model_id,
        value=to_serializable(value),
    )


def alignn_candidate_response(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
    request: ALIGNNTabularCandidateRequest,
    *,
    use_ask: bool = False,
) -> TabularCandidateResponse:
    """Generate structure-enumerated candidates and serialize the result."""

    method = optimizer.ask if use_ask else optimizer.candidate
    direct_kwargs = _candidate_direct_kwargs(request)
    direct_kwargs["structure_ids"] = request.structure_ids
    candidates, acq_value = method(
        acq_config=request.acq_config,
        opt_config=_candidate_optimize_config(request),
        bounds=request.bounds,
        return_dataframe=True,
        **direct_kwargs,
    )
    columns, records = _frame_records(candidates)
    return TabularCandidateResponse(
        model_id=model_id,
        columns=columns,
        candidates=records,
        acq_value=to_serializable(acq_value),
    )


__all__ = [
    "alignn_candidate_response",
    "alignn_predict_response",
    "build_alignn_fit_response",
    "fit_alignn_tabular_optimizer",
    "graph_builder_from_request",
    "structure_catalog_from_request",
]
