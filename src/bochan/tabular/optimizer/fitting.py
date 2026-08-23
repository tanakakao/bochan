"""Fit and dataset-conversion lifecycle for the tabular optimizer facade."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from bochan.api import CrossValidationConfig
from bochan.composition import ATOMIC_NUMBERS, ATOMIC_WEIGHTS

from ..config import UNSET, make_fit_config, make_model_config
from ..data import dataframe_to_tensors, numpy_to_tensors
from .configuration import DATA_KEYS, FIT_KEYS, MODEL_KEYS, merge_data_config, resolve_cv_config, take
from .settings import apply_alpha_to_model_config, merge_input_transform_config, validate_noise_alpha

_CRABNET_MODEL_TYPES = frozenset(
    {"crabnet_gp", "crabnet_dkl", "crabnet_mixed_gp", "crabnet_mixed_dkl"}
)
_CRABNET_MIXED_MODEL_TYPES = frozenset({"crabnet_mixed_gp", "crabnet_mixed_dkl"})
_CRABNET_FROZEN_MODEL_TYPES = frozenset({"crabnet_gp", "crabnet_mixed_gp"})
_WEIGHT_BASIS_NAMES = frozenset({"weight", "weight_fraction", "mass_fraction", "wt%"})


def _configure_tabular_crabnet_model(
    owner: Any,
    dataset: Any,
    config: Any,
) -> Any:
    """Derive the canonical CrabNet tensor contract from one composition site."""

    model_type = str(config.model_type).lower()
    if model_type not in _CRABNET_MODEL_TYPES:
        return config
    mixed_model = model_type in _CRABNET_MIXED_MODEL_TYPES
    dkl_model = model_type in {"crabnet_dkl", "crabnet_mixed_dkl"}
    if str(config.task_type) != "regression":
        raise ValueError("Tabular CrabNet models support task_type='regression' only.")
    if config.multi_output_config is not None or dataset.Y.shape[-1] != 1:
        raise ValueError("Tabular CrabNet models currently support single-output Gaussian regression only.")
    expected_input_type = "mixed" if mixed_model else "normal"
    if config.input_type not in (None, expected_input_type):
        raise ValueError(
            f"{model_type} requires input_type={expected_input_type!r}."
        )
    if mixed_model:
        if not dataset.cat_dims:
            fallback = "crabnet_dkl" if dkl_model else "crabnet_gp"
            raise ValueError(
                f"{model_type} requires at least one categorical process column. "
                f"Use {fallback} when all process columns are continuous."
            )
    elif dataset.cat_dims:
        categorical = [dataset.feature_names[index] for index in dataset.cat_dims]
        fallback = "crabnet_mixed_dkl" if dkl_model else "crabnet_mixed_gp"
        raise ValueError(
            f"{model_type} supports continuous process columns only; "
            f"categorical columns were configured: {categorical!r}. "
            f"Use model_type={fallback!r} for mixed process inputs."
        )
    if len(owner.composition.sites) != 1:
        raise ValueError("Tabular CrabNet models require exactly one composition site.")

    site_name, site_config = next(iter(owner.composition.sites.items()))
    if site_config["include_descriptors"]:
        raise ValueError(
            "Tabular CrabNet models do not accept independent composition "
            f"descriptor columns; disable include_descriptors for site {site_name!r}."
        )
    transformer = owner.composition.transformers.get(site_name)
    if transformer is None:
        raise RuntimeError("The CrabNet composition transformer must be fitted before model construction.")

    elements = transformer.fitted_elements
    coordinate_names = list(transformer.representation_feature_names_)
    feature_names = list(dataset.feature_names)
    missing = [name for name in coordinate_names if name not in feature_names]
    if missing:
        raise ValueError(
            f"The CrabNet composition source must be included in input_cols; missing model coordinates: {missing!r}."
        )
    composition_indices = [feature_names.index(name) for name in coordinate_names]
    composition_index_set = set(composition_indices)
    categorical_index_set = set(dataset.cat_dims)
    process_indices = [
        index
        for index in range(dataset.X.shape[-1])
        if index not in composition_index_set and index not in categorical_index_set
    ]

    model_kwargs = dict(config.model_kwargs)
    derived_names = {
        "element_ids",
        "input_transform",
        "composition_indices",
        "method",
        "reference_index",
        "process_bounds",
        "component_weights",
        "normalize_process",
        "category_cardinalities",
    }
    reserved = sorted(derived_names.intersection(model_kwargs))
    if config.input_transform is not None or reserved:
        raise ValueError(
            "Tabular CrabNet models derive composition/process layout from "
            "composition_sites and categorical_cols; do not provide "
            f"input_transform or derived model kwargs explicitly: {reserved!r}."
        )

    encoder_training = model_kwargs.pop("encoder_training", None)
    if model_type in _CRABNET_FROZEN_MODEL_TYPES:
        if encoder_training is not None or "trainable_encoder_layers" in model_kwargs:
            fallback = "crabnet_mixed_dkl" if mixed_model else "crabnet_dkl"
            raise ValueError(
                f"{model_type} always freezes the encoder. Use model_type={fallback!r} "
                "for partial or full encoder training."
            )
    elif encoder_training is not None:
        if "trainable_encoder_layers" in model_kwargs:
            raise ValueError("Specify either encoder_training or trainable_encoder_layers, not both.")
        normalized_training = str(encoder_training).lower()
        if normalized_training == "partial":
            model_kwargs["trainable_encoder_layers"] = 1
        elif normalized_training == "full":
            model_kwargs["trainable_encoder_layers"] = "all"
        else:
            raise ValueError(
                "encoder_training must be 'partial' or 'full'. Use crabnet_gp or "
                "crabnet_mixed_gp for a frozen encoder."
            )

    transform_config = config.input_transform_config
    normalize_process = True
    transform_bounds = dataset.bounds
    if transform_config is not None:
        if transform_config.perturbation:
            raise ValueError("Tabular CrabNet models do not yet support input perturbation.")
        if not mixed_model and transform_config.categorical_idx is not None:
            raise ValueError("Continuous CrabNet models do not support categorical input transforms.")
        normalize_process = bool(transform_config.normalize)
        if transform_config.bounds is not None:
            try:
                transform_bounds = torch.as_tensor(
                    transform_config.bounds,
                    dtype=dataset.X.dtype,
                    device=dataset.X.device,
                )
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "CrabNet InputTransformConfig.bounds must be a 2 x d tensor-like value; "
                    "use tabular bounds for column-addressed mappings."
                ) from error

    process_bounds = None
    if process_indices and normalize_process:
        if transform_bounds is None:
            transform_bounds = torch.stack(
                (
                    dataset.X.min(dim=0).values,
                    dataset.X.max(dim=0).values,
                )
            )
        if transform_bounds.shape != (2, dataset.X.shape[-1]):
            raise ValueError("CrabNet process normalization bounds must have shape [2, d].")
        process_bounds = transform_bounds[:, process_indices]

    reference_index = None
    if str(site_config["representation"]).lower() == "alr":
        reference_element = site_config["reference_element"]
        reference_index = len(elements) - 1 if reference_element is None else elements.index(reference_element)

    component_weights = dataset.X.new_ones(len(elements))
    if str(site_config["normalization"]).lower() in _WEIGHT_BASIS_NAMES:
        component_weights = dataset.X.new_tensor([ATOMIC_WEIGHTS[element] for element in elements])

    element_ids = dataset.X.new_tensor(
        [ATOMIC_NUMBERS[element] for element in elements],
        dtype=torch.long,
    )
    model_kwargs["element_ids"] = element_ids

    if mixed_model:
        if model_type == "crabnet_mixed_dkl":
            from bochan.models.regression.gaussian.deep import CrabNetMixedDKLModel

            model_cls = CrabNetMixedDKLModel
            category_cardinalities: list[int] = []
            for index in dataset.cat_dims:
                values = dataset.X[:, int(index)]
                rounded = values.round()
                if not torch.allclose(values, rounded, rtol=0.0, atol=1e-6) or (rounded < 0).any():
                    raise ValueError(
                        "CrabNet-Mixed DKL requires non-negative integer-coded categorical process columns."
                    )
                category_cardinalities.append(int(rounded.max().item()) + 1)
            model_kwargs["category_cardinalities"] = category_cardinalities
        else:
            from bochan.models.regression.gaussian.deep import CrabNetMixedGPModel

            model_cls = CrabNetMixedGPModel

        model_kwargs.update(
            {
                "composition_indices": composition_indices,
                "method": str(site_config["representation"]),
                "reference_index": reference_index,
                "process_bounds": process_bounds,
                "component_weights": component_weights,
                "normalize_process": normalize_process,
            }
        )
        return replace(
            config,
            model_cls=model_cls,
            input_type="mixed",
            input_transform=None,
            input_transform_config=None,
            model_kwargs=model_kwargs,
        )

    from bochan.models.regression.gaussian.deep import CrabNetInputTransform

    input_transform = CrabNetInputTransform(
        input_dim=dataset.X.shape[-1],
        composition_indices=composition_indices,
        n_components=len(elements),
        method=str(site_config["representation"]),
        reference_index=reference_index,
        process_bounds=process_bounds,
        component_weights=component_weights,
        normalize_process=normalize_process,
    ).to(dataset.X)

    return replace(
        config,
        input_type="normal",
        input_transform=input_transform,
        input_transform_config=None,
        model_kwargs=model_kwargs,
    )


def default_to_dataset(owner: Any, data: Any, y: Any | None = None, *, data_config: Any = None, feature_names: Any = None, target_names: Any = None) -> Any:
    config = data_config or owner.data_config
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None and isinstance(data, pd.DataFrame):
        return dataframe_to_tensors(data, config)
    return numpy_to_tensors(data, y, config, feature_names=feature_names, target_names=target_names)


def to_dataset(owner: Any, data: Any, y: Any | None = None, *, data_config: Any = None, feature_names: Any = None, target_names: Any = None) -> Any:
    config = data_config or owner.data_config
    def converter(value: Any, target: Any = None, **converter_kwargs: Any) -> Any:
        return default_to_dataset(owner, value, target, **converter_kwargs)
    return owner.observation.to_dataset(data, y, config=config, feature_names=feature_names, target_names=target_names, default_converter=converter)


def model_config_for_dataset(owner: Any, dataset: Any) -> Any:
    config = _configure_tabular_crabnet_model(
        owner,
        dataset,
        owner.model_config,
    )
    if config.cat_dims is None and dataset.cat_dims:
        config = replace(config, cat_dims=dataset.cat_dims)
    return apply_alpha_to_model_config(config, train_X=dataset.X, train_Y=dataset.Y, explicit_alpha=owner.alpha)


def sync_visualization_metadata(owner: Any) -> None:
    if owner.dataset is None or owner.bo.bundle is None:
        return
    metadata = dict(getattr(owner.bo.bundle, "metadata", {}) or {})
    metadata["feature_cols"] = list(owner.dataset.feature_names)
    metadata["target_cols"] = list(owner.dataset.target_names)
    if owner.dataset.category_maps:
        labels = dict(metadata.get("labels") or {})
        labels.update(owner.dataset.category_maps)
        metadata["labels"] = labels
    owner.bo.bundle.metadata = metadata


def fit_optimizer(owner: Any, data: Any | None, y: Any | None, *, data_config: Any, alpha: Any, beta: Any, normalize: Any, perturbation: Any, n_w: Any, std: Any, target_missing_strategy: Any, experiment_status_col: Any, failure_config: Any, cross_validation: bool | None, cv_config: Any, kwargs: dict[str, Any]) -> Any:
    data = owner.data if data is None else data
    if data is None:
        raise ValueError("No data was supplied. Pass data to fit(...) or use from_csv(...).")
    if alpha is not UNSET:
        owner.alpha = validate_noise_alpha(alpha)
    model_values = take(kwargs, MODEL_KEYS)
    supplied_model = kwargs.pop("model_config", None)
    transform_config = merge_input_transform_config(model_config=supplied_model or owner.model_config, input_transform_config=model_values.get("input_transform_config", UNSET), normalize=normalize, perturbation=perturbation, n_w=n_w, std=std)
    if transform_config is not UNSET:
        model_values["input_transform_config"] = transform_config
    owner.model_config = make_model_config(supplied_model or owner.model_config, **model_values)
    fit_values = take(kwargs, FIT_KEYS)
    supplied_fit = kwargs.pop("fit_config", None)
    if beta is not UNSET:
        fit_values["beta"] = beta
    owner.fit_config = make_fit_config(supplied_fit or owner.fit_config, **fit_values)
    source_config = merge_data_config(data_config or owner.source_data_config, take(kwargs, DATA_KEYS))
    source_config = owner.observation.resolve_config(source_config, target_missing_strategy=target_missing_strategy, experiment_status_col=experiment_status_col)
    owner.source_data_config = source_config
    fit_data = data
    resolved = source_config
    if owner.composition.enabled:
        fit_data = owner.composition.prepare_frame(data, fit_transformers=True)
        resolved = replace(source_config, input_cols=owner.composition.replace_input_cols(source_config.input_cols), categorical_cols=owner.composition.resolve_categorical_cols(source_config.categorical_cols, default_categorical_cols=source_config.categorical_cols or ()), bounds=owner.composition.expanded_bounds(source_config.bounds, fit_data))
    owner.data_config = resolved
    run_cv = owner.cross_validation if cross_validation is None else bool(cross_validation)
    if owner.observation.uses_observation_conversion(resolved) and run_cv:
        raise ValueError("Cross-validation requires an observation-aware validation protocol.")
    dataset = to_dataset(owner, fit_data, y, data_config=resolved)
    if dataset.Y is None:
        raise ValueError("Target values are required for fit(). Set target_cols or pass y.")
    owner.dataset = dataset
    model_config = model_config_for_dataset(owner, dataset)
    resolved_cv = resolve_cv_config(cv_config) if cv_config is not None else owner.cv_config
    owner.cross_validation_result_ = None
    if run_cv:
        owner.cross_validation_result_ = owner.bo.cross_validate(dataset.X, dataset.Y, model_config=model_config, fit_config=owner.fit_config, cv_config=resolved_cv or CrossValidationConfig())
    owner.bo.fit(dataset.X, dataset.Y, model_config=model_config, fit_config=owner.fit_config)
    if dataset.bounds is not None:
        owner.bo.set_bounds(dataset.bounds)
    owner.observation.attach(owner.bo, dataset, failure_config=owner.observation.resolve_failure_config(failure_config))
    sync_visualization_metadata(owner)
    return owner


__all__ = ["default_to_dataset", "fit_optimizer", "model_config_for_dataset", "sync_visualization_metadata", "to_dataset"]
