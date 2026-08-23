"""Adapters that run the React Web workflow through TabularBayesianOptimizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any


def relabel_feature_importance_outputs(
    result: Any,
    target_columns: Sequence[str],
) -> Any:
    """Replace positional output names with source target-column names.

    Native wide multitask models do not use ``MultiOutputConfig``. Core
    cross-validation therefore names their outputs ``output_0``, ``output_1``,
    and so on, even though permutation importance is evaluated independently
    for each posterior output. The Web response should use the original target
    columns so the summary table and generated figure identifiers stay aligned.

    The supplied result is mutated in place and returned for convenience. If
    its output count does not match the target-column count, no changes are
    made because a positional mapping would be ambiguous.
    """

    outputs = getattr(result, "outputs", None)
    targets = [str(column) for column in target_columns]
    if not isinstance(outputs, dict) or len(outputs) != len(targets):
        return result
    if len(set(targets)) != len(targets):
        return result

    original_items = list(outputs.items())
    name_map: dict[str, str] = {}
    renamed: dict[str, Any] = {}
    for (original_name, output), target_name in zip(
        original_items,
        targets,
        strict=True,
    ):
        name_map[str(original_name)] = target_name
        if hasattr(output, "output_name"):
            output.output_name = target_name
        renamed[target_name] = output

    result.outputs = renamed
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        metadata["output_name_map"] = name_map
        metadata["output_names_source"] = "target_columns"
    return result


def _category_key_from_label(series: Any, label: Any) -> Any:
    """Recover an original category value from the Web encoder's string label."""

    observed = series.dropna().unique().tolist()
    for value in observed:
        if value == label or str(value) == str(label):
            return value

    try:
        import pandas as pd

        if pd.api.types.is_integer_dtype(series.dtype):
            return int(float(label))
        if pd.api.types.is_float_dtype(series.dtype):
            return float(label)
    except (TypeError, ValueError):
        pass
    return label


def feature_category_maps(
    data: Any,
    encoded_features: dict[str, Any],
) -> dict[str, dict[Any, int]]:
    """Convert Web category maps to original-label maps accepted by the tabular API."""

    maps: dict[str, dict[Any, int]] = {}
    for column, encoded_map in dict(
        encoded_features.get("category_maps") or {}
    ).items():
        if column not in data.columns:
            continue
        series = data[column]
        maps[str(column)] = {
            _category_key_from_label(series, label): int(index)
            for label, index in dict(encoded_map).items()
        }
    return maps


def target_category_maps(
    target_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[Any, int]]:
    """Build explicit target maps so classification and ordinal ranks stay stable."""

    maps: dict[str, dict[Any, int]] = {}
    for target, metadata in target_metadata.items():
        if metadata.get("internal_task") == "regression":
            continue
        classes = list(metadata.get("classes") or [])
        maps[target] = {value: index for index, value in enumerate(classes)}
    return maps


def tabular_bounds(
    encoded_features: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    """Return column-keyed bounds for TabularDataConfig."""

    feature_columns = list(encoded_features["feature_columns"])
    lower, upper = encoded_features["bounds"]
    return {
        column: (float(lower[index]), float(upper[index]))
        for index, column in enumerate(feature_columns)
    }


def categorical_feature_columns(
    encoded_features: dict[str, Any],
) -> list[str]:
    """Resolve encoded categorical positions back to feature names."""

    columns = list(encoded_features["feature_columns"])
    return [
        columns[int(index)]
        for index in encoded_features.get("cat_dims", [])
    ]


def categorical_target_columns(
    target_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    """Return targets that require tabular class/rank encoding."""

    return [
        target
        for target, metadata in target_metadata.items()
        if metadata.get("internal_task") != "regression"
    ]


def _mutable_category_frame(
    data: Any,
    *,
    categorical_columns: list[str],
) -> Any:
    """Cast extension string/category columns before replacing labels with codes."""

    import pandas as pd

    frame = data.copy()
    for column in categorical_columns:
        series = frame.loc[:, column]
        if pd.api.types.is_string_dtype(series.dtype) or isinstance(
            series.dtype,
            pd.CategoricalDtype,
        ):
            frame[column] = series.astype(object)
    return frame


def _unwrap_single_output_crabnet_mixed_model_config(model_config: Any) -> Any:
    """Convert Web's one-output hybrid wrapper into the direct mixed CrabNet model.

    The generic Web workflow builds non-special single-output models through a
    one-output ``MultiOutputConfig``. ``crabnet_mixed_gp`` needs the direct
    tabular model contract so composition/categorical layout can be derived
    before construction. Keep all parent search/transform settings while taking
    the single output's model kwargs and regression task.
    """

    if str(getattr(model_config, "model_type", "")).lower() != "crabnet_mixed_gp":
        return model_config
    if str(getattr(model_config, "task_type", "")).lower() != "hybrid":
        return model_config

    multi_output = getattr(model_config, "multi_output_config", None)
    output_configs = list(getattr(multi_output, "output_configs", None) or [])
    if len(output_configs) != 1:
        raise ValueError(
            "crabnet_mixed_gp currently supports exactly one regression target."
        )
    output = output_configs[0]
    if str(getattr(output, "task_type", "")).lower() != "regression":
        raise ValueError(
            "crabnet_mixed_gp currently supports a continuous regression target only."
        )
    return replace(
        model_config,
        task_type="regression",
        input_type="mixed",
        model_kwargs=dict(getattr(output, "model_kwargs", {}) or {}),
        multi_output_config=None,
    )


def _descriptor_augmented_model_config(
    *,
    model_config: Any,
    encoded_features: dict[str, Any],
    composition_config: dict[str, Any] | None,
) -> Any:
    """Attach a differentiable derived-descriptor transform to a Web model.

    The tabular dataset and candidate optimizer remain in the original decision
    space. The model sees descriptors appended from the current composition at
    every train/predict/acquisition evaluation.
    """

    if composition_config is None or not bool(
        composition_config.get("include_descriptors", False)
    ):
        return model_config

    model_type = str(getattr(model_config, "model_type", "")).lower()
    if model_type in {"crabnet_gp", "crabnet_dkl", "crabnet_mixed_gp"}:
        raise ValueError(
            "CrabNet already derives a learned material representation from the "
            "composition. Web composition descriptors cannot currently be "
            "combined with CrabNet models."
        )
    if model_type in {
        "random_forest",
        "lightgbm_ensemble",
        "ngboost_ensemble",
        "tabpfn",
    }:
        raise ValueError(
            "Web composition descriptor augmentation currently requires a "
            "BoTorch-native surrogate model; external estimator models are not "
            "yet supported."
        )
    if getattr(model_config, "input_transform", None) is not None:
        raise ValueError(
            "Composition descriptor augmentation cannot replace an explicitly "
            "configured model input_transform."
        )

    transform_config = getattr(model_config, "input_transform_config", None)
    if transform_config is not None and bool(
        getattr(transform_config, "perturbation", False)
    ):
        raise ValueError(
            "Web composition descriptors do not yet support input perturbation. "
            "Disable input perturbation so descriptors remain deterministic "
            "functions of each composition candidate."
        )

    import torch

    from .composition.descriptors import (
        build_composition_descriptor_input_transform,
    )

    bounds = torch.as_tensor(
        encoded_features["bounds"],
        dtype=torch.double,
    )
    normalize = bool(
        getattr(transform_config, "normalize", True)
        if transform_config is not None
        else True
    )
    input_transform, descriptor_names, augmented_bounds = (
        build_composition_descriptor_input_transform(
            feature_names=encoded_features["feature_columns"],
            bounds=bounds,
            categorical_idx=encoded_features.get("cat_dims") or None,
            config=composition_config,
            normalize=normalize,
        )
    )
    composition_config["descriptor_feature_names"] = list(descriptor_names)
    composition_config["model_feature_names"] = [
        *list(encoded_features["feature_columns"]),
        *descriptor_names,
    ]
    composition_config["model_bounds_with_descriptors"] = (
        augmented_bounds.detach().cpu().tolist()
    )
    return replace(
        model_config,
        input_transform=input_transform,
        input_transform_config=None,
    )


def fit_tabular_optimizer(
    *,
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    encoded_features: dict[str, Any],
    target_metadata: dict[str, dict[str, Any]],
    model_config: Any,
    fit_config: Any,
    cross_validation: bool = False,
    cv_config: dict[str, Any] | None = None,
    composition_config: dict[str, Any] | None = None,
) -> Any:
    """Fit or reuse the public pandas-friendly optimizer for the Web workflow."""

    from .logging import current_request_id
    from .services.model_reuse import (
        current_model_reuse_state,
        register_fitted_model,
        reuse_fitted_tabular_optimizer,
    )

    model_config = _unwrap_single_output_crabnet_mixed_model_config(model_config)
    model_config = _descriptor_augmented_model_config(
        model_config=model_config,
        encoded_features=encoded_features,
        composition_config=composition_config,
    )

    run_id = current_request_id()
    reuse_state = current_model_reuse_state()
    source_run_id = str((reuse_state or {}).get("source_run_id") or "")
    if source_run_id:
        if not run_id:
            raise RuntimeError(
                "Model reuse requires an active Web request identifier."
            )
        return reuse_fitted_tabular_optimizer(
            source_run_id=source_run_id,
            current_run_id=run_id,
            data=data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            target_metadata=target_metadata,
            model_config=model_config,
            hybrid_model=str(getattr(model_config, "task_type", "")) == "hybrid",
            composition_config=composition_config,
        )

    from bochan.tabular import TabularBayesianOptimizer

    from .targets.missing import current_target_missing_report

    missing_report = current_target_missing_report()
    target_missing_strategy = (
        "keep"
        if bool(missing_report.get("preserve_target_missing"))
        else "drop"
    )

    categorical_features = [
        column
        for column in categorical_feature_columns(encoded_features)
        if column in data.columns
        and (
            composition_config is None
            or column != composition_config["column"]
        )
    ]
    categorical_targets = categorical_target_columns(target_metadata)
    fit_data = _mutable_category_frame(
        data,
        categorical_columns=[*categorical_features, *categorical_targets],
    )

    composition_kwargs: dict[str, Any] = {}
    if composition_config is not None:
        from .composition.support import composition_site

        composition_kwargs = {
            "composition_sites": {
                "composition": composition_site(composition_config),
            },
            "composition_element_constraints": composition_config[
                "element_constraints"
            ],
            "composition_constraint_rerank": True,
        }

    optimizer = TabularBayesianOptimizer(
        model_config=model_config,
        fit_config=fit_config,
        input_cols=feature_columns,
        target_cols=target_columns,
        categorical_cols=categorical_features,
        target_categorical_cols=categorical_targets,
        bounds=tabular_bounds(encoded_features),
        category_maps=feature_category_maps(data, encoded_features),
        target_category_maps=target_category_maps(target_metadata),
        encode_categories=True,
        return_original_categories=True,
        dropna=False,
        target_missing_strategy=target_missing_strategy,
        cross_validation=cross_validation,
        cv_config=cv_config,
        **composition_kwargs,
    )
    optimizer.fit(fit_data)
    if optimizer.dataset is None:
        raise RuntimeError(
            "TabularBayesianOptimizer did not retain its fitted dataset."
        )

    state = current_target_missing_report()
    if state.get("feature_impute_values"):
        optimizer.dataset.impute_values = dict(state["feature_impute_values"])

    cross_validation_result = optimizer.cross_validation_result_
    if cross_validation_result is not None:
        feature_importance = getattr(
            cross_validation_result,
            "feature_importance",
            None,
        )
        if feature_importance is not None:
            relabel_feature_importance_outputs(
                feature_importance,
                target_columns,
            )

    from .services.visualization_sessions import attach_fitted_tabular_optimizer

    if run_id:
        attach_fitted_tabular_optimizer(
            run_id,
            tabular_optimizer=optimizer,
            data=data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            target_metadata=target_metadata,
            hybrid_model=str(getattr(model_config, "task_type", "")) == "hybrid",
        )
        register_fitted_model(run_id)
    return optimizer


def encoded_features_from_tabular(
    optimizer: Any,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Expose tabular dataset metadata in the Web response shape."""

    dataset = optimizer.dataset
    if dataset is None:
        raise RuntimeError("TabularBayesianOptimizer is not fitted.")
    metadata = dict(fallback)
    metadata["feature_columns"] = list(dataset.feature_names)
    metadata["cat_dims"] = list(dataset.cat_dims)
    metadata["category_maps"] = dict(dataset.category_maps or {})
    metadata["inverse_category_maps"] = dict(
        dataset.inverse_category_maps or {}
    )
    if dataset.bounds is not None:
        metadata["bounds"] = dataset.bounds.detach().cpu().tolist()
    return metadata


__all__ = [
    "categorical_feature_columns",
    "categorical_target_columns",
    "encoded_features_from_tabular",
    "feature_category_maps",
    "fit_tabular_optimizer",
    "relabel_feature_importance_outputs",
    "tabular_bounds",
    "target_category_maps",
]
