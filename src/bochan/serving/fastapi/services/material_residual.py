"""FastAPI application service for CHGNet/M3GNet/MACE residual GP models."""

from __future__ import annotations

from typing import Any

from bochan.tabular import TabularBayesianOptimizer

from ..converters import to_serializable
from ..schemas.material_residual import (
    MaterialResidualTabularCandidateRequest,
    MaterialResidualTabularFitModelRequest,
)
from ..schemas.tabular import (
    TabularCandidateResponse,
    TabularModelFitResponse,
    TabularPredictRequest,
    TabularPredictResponse,
)
from .chgnet_tabular import (
    _model_config_from_request as _chgnet_model_config_from_request,
    _normalize_structure_column,
    structure_catalog_from_request,
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

_MULTITASK_RESIDUAL_TYPES = frozenset(
    {
        "chgnet_multitask_residual_gp",
        "chgnet_mixed_multitask_residual_gp",
        "m3gnet_multitask_residual_gp",
        "m3gnet_mixed_multitask_residual_gp",
        "mace_multitask_residual_gp",
        "mace_mixed_multitask_residual_gp",
    }
)
_INDEPENDENT_RESIDUAL_TYPES = frozenset(
    {
        "chgnet_multioutput_residual_gp",
        "chgnet_mixed_multioutput_residual_gp",
        "m3gnet_multioutput_residual_gp",
        "m3gnet_mixed_multioutput_residual_gp",
        "mace_multioutput_residual_gp",
        "mace_mixed_multioutput_residual_gp",
    }
)


def _family(model_type: str) -> str:
    return model_type.split("_", 1)[0]


def _model_config_from_request(
    request: MaterialResidualTabularFitModelRequest,
) -> dict[str, Any]:
    """Resolve one API-safe residual model configuration."""

    model_type = str(request.bo_model_config.model_type).lower()
    if _family(model_type) == "chgnet":
        return _chgnet_model_config_from_request(request)  # type: ignore[arg-type]
    return _schema_dict(request.bo_model_config)


def fit_material_residual_tabular_optimizer(
    request: MaterialResidualTabularFitModelRequest,
) -> TabularBayesianOptimizer:
    """Fit one residual GP optimizer from JSON data and a structure catalog."""

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

    optimizer = TabularBayesianOptimizer(
        model_config=_model_config_from_request(request),
        fit_config=_schema_dict(request.fit_config),
        input_cols=request.input_cols,
        target_cols=request.target_cols,
        target_variance_cols=request.target_variance_cols,
        categorical_cols=request.categorical_cols,
        target_categorical_cols=request.target_categorical_cols,
        bounds=request.bounds,
        structure_col=request.structure_col,
        structure_catalog=structure_catalog_from_request(request),  # type: ignore[arg-type]
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


def _independent_output_metadata(bundle: Any, target_names: list[str]) -> tuple[list[dict[str, Any]], int]:
    """Describe ModelList outputs and locate the unique pretrained residual output."""

    sub_bundles = list((getattr(bundle, "metadata", {}) or {}).get("sub_bundles", []))
    if len(sub_bundles) != len(target_names):
        raise RuntimeError(
            "Independent residual bundle metadata must expose one sub-bundle per target."
        )

    rows: list[dict[str, Any]] = []
    residual_indices: list[int] = []
    for index, sub_bundle in enumerate(sub_bundles):
        model = sub_bundle.model
        is_residual = hasattr(model, "residual_model") and hasattr(model, "predictor")
        if is_residual:
            residual_indices.append(index)
        encoder = getattr(model, "material_encoder", None)
        residual_model = getattr(model, "residual_model", None)
        if encoder is None and residual_model is not None:
            encoder = getattr(residual_model, "material_encoder", None)
        rows.append(
            {
                "index": index,
                "name": target_names[index],
                "role": "pretrained_residual" if is_residual else "ordinary_gp",
                "model_type": str(sub_bundle.model_type),
                "model_cls": model.__class__.__name__,
                "encoder_cls": None if encoder is None else encoder.__class__.__name__,
                "residual_model_cls": (
                    None if residual_model is None else residual_model.__class__.__name__
                ),
            }
        )
    if len(residual_indices) != 1:
        raise RuntimeError(
            "Independent residual ModelList must contain exactly one residual output model."
        )
    return rows, residual_indices[0]


def build_material_residual_fit_response(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
) -> TabularModelFitResponse:
    """Serialize a fitted residual model and its pretrained-baseline contract."""

    response = build_fit_response(model_id, optimizer)
    bundle = optimizer.bo.bundle
    if bundle is None:
        return response

    model = bundle.model
    model_type = str(bundle.model_type).lower()
    family = _family(model_type)
    multitask = model_type in _MULTITASK_RESIDUAL_TYPES
    independent = model_type in _INDEPENDENT_RESIDUAL_TYPES

    dataset = optimizer.dataset
    feature_names = list(getattr(dataset, "feature_names", None) or [])
    target_names = [str(name) for name in (getattr(dataset, "target_names", None) or [])]
    process_cat_dims = [int(index) for index in (bundle.cat_dims or [])]
    process_cat_dim_set = set(process_cat_dims)
    categorical_process_cols = [
        feature_names[index]
        for index in process_cat_dims
        if 0 <= index < len(feature_names)
    ]
    continuous_process_cols = [
        name
        for index, name in enumerate(feature_names)
        if index != 0 and index not in process_cat_dim_set
    ]

    output_models: list[dict[str, Any]] = []
    if independent:
        output_models, pretrained_output_index = _independent_output_metadata(
            bundle,
            target_names,
        )
        residual_submodel = list(bundle.metadata["sub_bundles"])[pretrained_output_index].model
        residual_model = getattr(residual_submodel, "residual_model", None)
        predictor = getattr(residual_submodel, "predictor", None)
        encoder = getattr(residual_submodel, "material_encoder", None)
        if encoder is None and residual_model is not None:
            encoder = getattr(residual_model, "material_encoder", None)
        output_dependency = "independent"
        num_outputs = len(output_models)
    else:
        residual_model = getattr(model, "residual_model", None)
        encoder = getattr(model, "material_encoder", None)
        if encoder is None and residual_model is not None:
            encoder = getattr(residual_model, "material_encoder", None)
        predictor = getattr(model, "predictor", None)
        model_kwargs = dict(getattr(bundle.model_config, "model_kwargs", None) or {})
        pretrained_output_index = model_kwargs.get("pretrained_output_index") if multitask else None
        output_dependency = "correlated" if multitask else "scalar"
        num_outputs = int(getattr(model, "num_outputs", len(target_names) or 1))

    metadata = dict(response.metadata)
    metadata["material_residual"] = to_serializable(
        {
            "family": family,
            "model_type": model_type,
            "residual_gp": True,
            "baseline_deterministic": True,
            "baseline_process_dependent": False,
            "baseline_property": "energy" if family in {"chgnet", "mace"} else "pretrained_scalar",
            "pretrained_output_index": pretrained_output_index,
            "output_dependency": output_dependency,
            "num_outputs": num_outputs,
            "output_names": target_names,
            "output_models": output_models,
            "structure_col": optimizer.structure.column,
            "structure_ids": list(optimizer.structure.structure_ids),
            "num_structures": optimizer.structure.num_structures,
            "continuous_process_cols": continuous_process_cols,
            "categorical_process_cols": categorical_process_cols,
            "categorical_process_dims": process_cat_dims,
            "encoder_frozen": True,
            "encoder_cls": None if encoder is None else encoder.__class__.__name__,
            "predictor_cls": None if predictor is None else predictor.__class__.__name__,
            "residual_model_cls": (
                None if residual_model is None else residual_model.__class__.__name__
            ),
        }
    )
    response.metadata = metadata
    return response


def material_residual_predict_response(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
    request: TabularPredictRequest,
) -> TabularPredictResponse:
    """Predict corrected properties with the fitted structure-ID mapping."""

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
        return TabularPredictResponse(model_id=model_id, columns=columns, records=records)
    return TabularPredictResponse(model_id=model_id, value=to_serializable(value))


def _register_pending_candidates(
    optimizer: TabularBayesianOptimizer,
    candidates: Any,
) -> None:
    import torch

    X_pending, _ = optimizer._prediction_input(candidates)  # noqa: SLF001
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


def material_residual_candidate_response(
    model_id: str,
    optimizer: TabularBayesianOptimizer,
    request: MaterialResidualTabularCandidateRequest,
    *,
    use_ask: bool = False,
) -> TabularCandidateResponse:
    """Enumerate structure/category choices and optimize continuous corrections."""

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
    "build_material_residual_fit_response",
    "fit_material_residual_tabular_optimizer",
    "material_residual_candidate_response",
    "material_residual_predict_response",
]
