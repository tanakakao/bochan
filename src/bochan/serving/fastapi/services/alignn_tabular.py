"""FastAPI application service for tabular ALIGNN models."""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
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

_CHECKPOINT_ROOT_ENV = "BOCHAN_ALIGNN_CHECKPOINT_ROOT"
_ALIGNN_CORRELATED_MULTITASK_MODEL_TYPES = frozenset(
    {"alignn_multitask", "alignn_multitask_dkl"}
)
_ALIGNN_FROZEN_MODEL_TYPES = frozenset({"alignn_gp", "alignn_multitask"})


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
            with suppress(FileNotFoundError):
                os.unlink(path)


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


def _model_config_from_request(request: ALIGNNTabularFitModelRequest) -> dict[str, Any]:
    """Resolve API-safe model config, including allowlisted checkpoint IDs."""

    config = _schema_dict(request.bo_model_config)
    model_kwargs = dict(config.get("model_kwargs") or {})
    checkpoint_id = model_kwargs.get("checkpoint")
    if checkpoint_id is None:
        return config

    root_value = os.environ.get(_CHECKPOINT_ROOT_ENV)
    if not root_value:
        raise ValueError(
            "ALIGNN checkpoint loading over FastAPI is disabled. Configure "
            f"{_CHECKPOINT_ROOT_ENV} on the server and pass only a checkpoint filename."
        )
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(
            f"Server checkpoint root from {_CHECKPOINT_ROOT_ENV} is not a directory."
        )
    checkpoint_path = (root / str(checkpoint_id)).resolve()
    if checkpoint_path.parent != root:
        raise ValueError("Checkpoint identifier resolved outside the configured checkpoint root.")
    if not checkpoint_path.is_file():
        raise ValueError("Unknown ALIGNN checkpoint identifier.")

    model_kwargs["checkpoint"] = str(checkpoint_path)
    config["model_kwargs"] = model_kwargs
    return config


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
        model_config=_model_config_from_request(request),
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


def _alignn_output_models(model: Any) -> list[Any]:
    """Return the single ALIGNN model or independent ModelListGP submodels."""

    models = getattr(model, "models", None)
    if models is not None:
        resolved = list(models)
        if resolved and all(hasattr(item, "material_encoder") for item in resolved):
            return resolved
    return [model]


def _encoder_training_mode(model_type: str, model: Any) -> str:
    if model_type.lower() in _ALIGNN_FROZEN_MODEL_TYPES:
        return "frozen"
    trainable_layers = getattr(model, "trainable_encoder_layers", None)
    return "full" if trainable_layers == "all" else "partial"


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
    model_type = str(bundle.model_type).lower()
    correlated = model_type in _ALIGNN_CORRELATED_MULTITASK_MODEL_TYPES
    representative_model = model if correlated else _alignn_output_models(model)[0]
    representative_encoder = getattr(representative_model, "material_encoder", None)
    training_mode = _encoder_training_mode(model_type, representative_model)
    initialization = getattr(representative_encoder, "initialization", None)
    graph_builder = getattr(optimizer.structure, "graph_builder", None)
    graph_config = getattr(graph_builder, "config", None)

    dataset = optimizer.dataset
    feature_names = list(getattr(dataset, "feature_names", None) or [])
    target_names = list(getattr(dataset, "target_names", None) or [])
    process_cat_dims = [int(index) for index in (bundle.cat_dims or [])]
    process_categorical_cols = [
        feature_names[index]
        for index in process_cat_dims
        if 0 <= index < len(feature_names)
    ]
    process_cat_dim_set = set(process_cat_dims)
    continuous_process_cols = [
        name
        for index, name in enumerate(feature_names)
        if index != 0 and index not in process_cat_dim_set
    ]
    category_maps = dict(getattr(dataset, "category_maps", None) or {})
    process_category_maps = {
        column: category_maps[column]
        for column in process_categorical_cols
        if column in category_maps
    }

    if correlated:
        num_outputs = int(getattr(model, "num_outputs", len(target_names) or 1))
        output_models = [model] * num_outputs
        output_dependency = "correlated"
    else:
        output_models = _alignn_output_models(model)
        num_outputs = len(output_models)
        output_dependency = "independent" if num_outputs > 1 else "single"

    output_metadata = []
    for index, output_model in enumerate(output_models):
        encoder = getattr(output_model, "material_encoder", None)
        output_metadata.append(
            {
                "name": target_names[index] if index < len(target_names) else f"y{index}",
                "model_cls": output_model.__class__.__name__,
                "encoder_training": _encoder_training_mode(model_type, output_model),
                "encoder_initialization": getattr(encoder, "initialization", None),
                "process_dim": getattr(output_model, "process_dim", None),
                "structure_feature_cache_enabled": getattr(
                    output_model,
                    "structure_feature_cache_enabled",
                    False,
                ),
                "shared_model": correlated,
            }
        )

    task_kernel = None
    task_covar_module = None
    if correlated:
        covar_module = getattr(getattr(model, "deepkernel", None), "covar_module", None)
        task_kernel = getattr(covar_module, "__class__", type(None)).__name__
        task_covar = getattr(model, "task_covar_module", None)
        if task_covar is not None:
            task_covar_module = task_covar.__class__.__name__

    metadata = dict(response.metadata)
    metadata["alignn"] = to_serializable(
        {
            "encoder_training": training_mode,
            "encoder_initialization": initialization,
            "checkpoint_configured": initialization == "checkpoint",
            "structure_col": optimizer.structure.column,
            "structure_ids": list(optimizer.structure.structure_ids),
            "num_structures": optimizer.structure.num_structures,
            "input_type": bundle.input_type,
            "process_dim": getattr(representative_model, "process_dim", None),
            "continuous_process_cols": continuous_process_cols,
            "categorical_process_cols": process_categorical_cols,
            "categorical_process_dims": process_cat_dims,
            "category_maps": process_category_maps,
            "graph_config": graph_config,
            "multi_output": num_outputs > 1,
            "num_outputs": num_outputs,
            "output_names": target_names,
            "output_models": output_metadata,
            "output_dependency": output_dependency,
            "shared_encoder": correlated,
            "task_kernel": task_kernel,
            "task_covar_module": task_covar_module,
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


def _register_pending_candidates(
    optimizer: TabularBayesianOptimizer,
    candidates: Any,
) -> None:
    """Append generated candidate rows to the canonical observation state as pending."""

    import torch

    X_pending, _ = optimizer._prediction_input(candidates)
    if X_pending.ndim == 1:
        X_pending = X_pending.unsqueeze(0)
    target_dim = int(optimizer.dataset.Y.shape[-1])
    Y_pending = torch.full(
        (int(X_pending.shape[0]), target_dim),
        float("nan"),
        dtype=optimizer.dataset.Y.dtype,
        device=optimizer.dataset.Y.device,
    )
    optimizer.bo.tell(
        X_pending,
        Y_pending,
        status=["pending"] * int(X_pending.shape[0]),
        refit=False,
    )


def alignn_candidate_response(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
    request: ALIGNNTabularCandidateRequest,
    *,
    use_ask: bool = False,
) -> TabularCandidateResponse:
    """Generate structure-enumerated candidates and serialize the result."""

    direct_kwargs = _candidate_direct_kwargs(request)
    direct_kwargs["structure_ids"] = request.structure_ids
    candidates, acq_value = optimizer.candidate(
        acq_config=request.acq_config,
        opt_config=_candidate_optimize_config(request),
        bounds=request.bounds,
        return_dataframe=True,
        **direct_kwargs,
    )
    if use_ask:
        _register_pending_candidates(optimizer, candidates)
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
