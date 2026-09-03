"""Tabular routing for pretrained structure-model residual Gaussian processes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any


_RESIDUAL_MODEL_SPECS: dict[str, tuple[str, bool, bool, str]] = {
    "chgnet_residual_gp": ("chgnet", False, False, "CHGNetResidualGPModel"),
    "chgnet_mixed_residual_gp": ("chgnet", True, False, "CHGNetMixedResidualGPModel"),
    "chgnet_multitask_residual_gp": (
        "chgnet",
        False,
        True,
        "CHGNetMultiTaskResidualGPModel",
    ),
    "chgnet_mixed_multitask_residual_gp": (
        "chgnet",
        True,
        True,
        "CHGNetMixedMultiTaskResidualGPModel",
    ),
    "m3gnet_residual_gp": ("m3gnet", False, False, "M3GNetResidualGPModel"),
    "m3gnet_mixed_residual_gp": ("m3gnet", True, False, "M3GNetMixedResidualGPModel"),
    "m3gnet_multitask_residual_gp": (
        "m3gnet",
        False,
        True,
        "M3GNetMultiTaskResidualGPModel",
    ),
    "m3gnet_mixed_multitask_residual_gp": (
        "m3gnet",
        True,
        True,
        "M3GNetMixedMultiTaskResidualGPModel",
    ),
    "mace_residual_gp": ("mace", False, False, "MACEResidualGPModel"),
    "mace_mixed_residual_gp": ("mace", True, False, "MACEMixedResidualGPModel"),
    "mace_multitask_residual_gp": (
        "mace",
        False,
        True,
        "MACEMultiTaskResidualGPModel",
    ),
    "mace_mixed_multitask_residual_gp": (
        "mace",
        True,
        True,
        "MACEMixedMultiTaskResidualGPModel",
    ),
}


def material_residual_model_types() -> tuple[str, ...]:
    """Return stable public tabular model types for structure residual GPs."""

    return tuple(sorted(_RESIDUAL_MODEL_SPECS))


def _target_names(target_cols: Any) -> list[Any] | None:
    if target_cols is None:
        return None
    if isinstance(target_cols, (str, bytes, int)):
        return [target_cols]
    if isinstance(target_cols, Sequence):
        return list(target_cols)
    return [target_cols]


def _has_mapping_key(mapping: Mapping[Any, Any], key: Any) -> bool:
    return key in mapping or str(key) in mapping


def _mixed_counterpart(model_type: str, *, mixed: bool) -> str:
    if mixed:
        return model_type.replace("_mixed_", "_")
    family, suffix = model_type.split("_", 1)
    return f"{family}_mixed_{suffix}"


def _resolve_model_class(class_name: str) -> type:
    from bochan.models.regression.gaussian.materials import structure

    return getattr(structure, class_name)


def _validate_model_kwargs(
    config: Any,
    *,
    structures: tuple[Any, ...],
    multitask: bool,
    n_targets: int,
) -> dict[str, Any]:
    if config.input_transform is not None:
        raise ValueError(
            "Tabular material residual models derive process normalization internally; "
            "do not pass input_transform explicitly."
        )
    transform_config = config.input_transform_config
    if transform_config is not None:
        if bool(transform_config.perturbation):
            raise ValueError("Tabular material residual models do not support input perturbation.")
        if not bool(transform_config.normalize):
            raise ValueError("Tabular material residual models currently require normalization.")
        if transform_config.bounds is not None:
            raise ValueError(
                "Tabular material residual models derive normalization from tabular bounds; "
                "do not pass InputTransformConfig.bounds."
            )

    model_kwargs = dict(config.model_kwargs)
    existing_structures = model_kwargs.pop("structures", None)
    if existing_structures is not None and existing_structures is not structures:
        raise ValueError(
            "Tabular material residual models derive structures from structure_catalog; "
            "do not override structures in model_kwargs."
        )
    if "encoder_training" in model_kwargs or "trainable_encoder_layers" in model_kwargs:
        raise ValueError(
            "Residual GP models keep the pretrained baseline encoder frozen; "
            "encoder_training and trainable_encoder_layers are not supported."
        )

    if multitask:
        value = model_kwargs.get("pretrained_output_index", 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("pretrained_output_index must be an integer.")
        if value < 0 or value >= n_targets:
            raise ValueError(
                "pretrained_output_index must select one target column: "
                f"0 <= index < {n_targets}."
            )
        model_kwargs["pretrained_output_index"] = value
    elif "pretrained_output_index" in model_kwargs:
        raise ValueError(
            "pretrained_output_index is only valid for correlated multitask residual GP models."
        )

    model_kwargs["structures"] = structures
    return model_kwargs


def configure_tabular_material_residual(owner: Any) -> bool:
    """Configure CHGNet/M3GNet/MACE residual GP structure/process contracts.

    The structure feature remains a discrete bank index at feature 0.  Numeric
    and categorical process variables are learned only by the GP correction;
    the deterministic pretrained baseline is structure-only.
    """

    config = owner.model_config
    model_type = str(config.model_type).lower()
    spec = _RESIDUAL_MODEL_SPECS.get(model_type)
    if spec is None:
        return False

    family, mixed, multitask, class_name = spec
    if str(config.task_type) not in {"regression", "multi_objective"}:
        raise ValueError(
            f"{family.upper()} residual tabular models support regression or multi_objective only."
        )
    if config.multi_output_config is not None:
        raise ValueError(
            "Residual structure models use one scalar or one correlated multitask model; "
            "do not provide multi_output_config."
        )
    if owner.composition.enabled:
        raise ValueError(
            "Residual structure models accept crystal structure plus process variables; "
            "composition_sites cannot be combined with them yet."
        )
    if not owner.structure.enabled:
        raise ValueError(
            f"model_type={model_type!r} requires structure_col and structure_catalog."
        )
    if owner.structure.graph_builder is not None:
        raise ValueError(
            "structure_graph_builder is ALIGNN-specific and must be omitted for residual models."
        )

    target_names = _target_names(owner.source_data_config.target_cols)
    if not target_names:
        raise ValueError("Residual structure models require at least one target column.")
    n_targets = len(target_names)
    if multitask:
        if n_targets < 2:
            fallback = model_type.replace("_multitask", "")
            raise ValueError(
                f"{model_type} requires at least two continuous target columns. "
                f"Use model_type={fallback!r} for a single target."
            )
    elif n_targets != 1:
        raise ValueError(
            f"{model_type} is a scalar residual GP and requires exactly one target column. "
            "Use the corresponding multitask_residual_gp model for correlated wide targets."
        )

    source = owner.source_data_config
    if source.input_cols is None:
        raise ValueError(
            "Tabular residual structure models require explicit input_cols so structure_col "
            "can be placed at model feature index 0."
        )
    input_cols = owner.structure.replace_input_cols(source.input_cols)
    categorical_cols = owner.structure.resolve_categorical_cols(source.categorical_cols)
    process_categorical_cols = [
        column for column in categorical_cols if column != owner.structure.column
    ]
    process_categorical_set = set(process_categorical_cols)
    continuous_process_cols = [
        column
        for column in input_cols
        if column != owner.structure.column and column not in process_categorical_set
    ]

    has_process_categories = bool(process_categorical_cols)
    if mixed != has_process_categories:
        counterpart = _mixed_counterpart(model_type, mixed=mixed)
        if mixed:
            raise ValueError(
                f"{model_type} requires at least one categorical process column. "
                f"Use model_type={counterpart!r} when all process columns are continuous."
            )
        raise ValueError(
            f"{model_type} does not accept categorical process columns. "
            f"Use model_type={counterpart!r} for mixed process inputs."
        )

    if config.input_type not in (None, "mixed" if mixed else "normal"):
        raise ValueError(
            f"{model_type} requires input_type={'mixed' if mixed else 'normal'!r}."
        )
    configured_cat_dims = [] if config.cat_dims is None else list(config.cat_dims)
    if configured_cat_dims:
        raise ValueError(
            "Tabular residual structure models derive cat_dims from categorical process columns; "
            "do not pass cat_dims explicitly."
        )

    if source.bounds is not None and not isinstance(source.bounds, Mapping):
        raise ValueError(
            "Tabular residual structure models require column-addressed bounds when supplied."
        )
    if continuous_process_cols and source.bounds is None:
        raise ValueError(
            "Tabular residual structure models require bounds for every continuous process variable."
        )
    bounds_mapping = source.bounds or {}
    missing_bounds = [
        column for column in continuous_process_cols if not _has_mapping_key(bounds_mapping, column)
    ]
    if missing_bounds:
        raise ValueError(
            "Tabular residual structure models require bounds for every continuous process variable; "
            f"missing bounds for {missing_bounds!r}."
        )

    category_maps = owner.structure.merge_category_maps(source.category_maps)
    bounds = owner.structure.expanded_bounds(source.bounds)
    resolved_source = replace(
        source,
        input_cols=input_cols,
        categorical_cols=categorical_cols,
        category_maps=category_maps,
        bounds=bounds,
    )
    owner.source_data_config = resolved_source
    owner.data_config = resolved_source

    process_cat_dims = [input_cols.index(column) for column in process_categorical_cols]
    structures = owner.structure.structures
    model_kwargs = _validate_model_kwargs(
        config,
        structures=structures,
        multitask=multitask,
        n_targets=n_targets,
    )
    model_cls = _resolve_model_class(class_name)

    owner.model_config = replace(
        config,
        task_type="multi_objective" if multitask else "regression",
        model_cls=model_cls,
        model_factory=None,
        input_type="mixed" if mixed else "normal",
        cat_dims=process_cat_dims,
        input_transform=None,
        input_transform_config=None,
        pass_cat_dims=mixed,
        pass_input_transform=False,
        model_kwargs=model_kwargs,
        multi_output_config=None,
    )
    return True


__all__ = ["configure_tabular_material_residual", "material_residual_model_types"]
