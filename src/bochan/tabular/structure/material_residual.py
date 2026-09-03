"""Tabular routing for pretrained structure-model residual Gaussian processes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

ResidualOutputMode = Literal["scalar", "multitask", "multioutput"]
ResidualModelSpec = tuple[str, bool, ResidualOutputMode, str | None]

_RESIDUAL_MODEL_SPECS: dict[str, ResidualModelSpec] = {
    "chgnet_residual_gp": ("chgnet", False, "scalar", "CHGNetResidualGPModel"),
    "chgnet_mixed_residual_gp": ("chgnet", True, "scalar", "CHGNetMixedResidualGPModel"),
    "chgnet_multitask_residual_gp": (
        "chgnet",
        False,
        "multitask",
        "CHGNetMultiTaskResidualGPModel",
    ),
    "chgnet_mixed_multitask_residual_gp": (
        "chgnet",
        True,
        "multitask",
        "CHGNetMixedMultiTaskResidualGPModel",
    ),
    "chgnet_multioutput_residual_gp": ("chgnet", False, "multioutput", None),
    "chgnet_mixed_multioutput_residual_gp": ("chgnet", True, "multioutput", None),
    "m3gnet_residual_gp": ("m3gnet", False, "scalar", "M3GNetResidualGPModel"),
    "m3gnet_mixed_residual_gp": ("m3gnet", True, "scalar", "M3GNetMixedResidualGPModel"),
    "m3gnet_multitask_residual_gp": (
        "m3gnet",
        False,
        "multitask",
        "M3GNetMultiTaskResidualGPModel",
    ),
    "m3gnet_mixed_multitask_residual_gp": (
        "m3gnet",
        True,
        "multitask",
        "M3GNetMixedMultiTaskResidualGPModel",
    ),
    "m3gnet_multioutput_residual_gp": ("m3gnet", False, "multioutput", None),
    "m3gnet_mixed_multioutput_residual_gp": ("m3gnet", True, "multioutput", None),
    "mace_residual_gp": ("mace", False, "scalar", "MACEResidualGPModel"),
    "mace_mixed_residual_gp": ("mace", True, "scalar", "MACEMixedResidualGPModel"),
    "mace_multitask_residual_gp": (
        "mace",
        False,
        "multitask",
        "MACEMultiTaskResidualGPModel",
    ),
    "mace_mixed_multitask_residual_gp": (
        "mace",
        True,
        "multitask",
        "MACEMixedMultiTaskResidualGPModel",
    ),
    "mace_multioutput_residual_gp": ("mace", False, "multioutput", None),
    "mace_mixed_multioutput_residual_gp": ("mace", True, "multioutput", None),
}

_ORDINARY_MODEL_CLASSES: dict[tuple[str, bool], str] = {
    ("chgnet", False): "CHGNetGPModel",
    ("chgnet", True): "CHGNetMixedGPModel",
    ("m3gnet", False): "M3GNetGPModel",
    ("m3gnet", True): "M3GNetMixedGPModel",
    ("mace", False): "MACEGPModel",
    ("mace", True): "MACEMixedGPModel",
}

_SCALAR_RESIDUAL_MODEL_CLASSES: dict[tuple[str, bool], str] = {
    ("chgnet", False): "CHGNetResidualGPModel",
    ("chgnet", True): "CHGNetMixedResidualGPModel",
    ("m3gnet", False): "M3GNetResidualGPModel",
    ("m3gnet", True): "M3GNetMixedResidualGPModel",
    ("mace", False): "MACEResidualGPModel",
    ("mace", True): "MACEMixedResidualGPModel",
}


def material_residual_model_types() -> tuple[str, ...]:
    """Return stable public tabular model types for structure residual GPs."""

    return tuple(sorted(_RESIDUAL_MODEL_SPECS))


def independent_residual_model_types() -> tuple[str, ...]:
    """Return public model types using independent ModelList output semantics."""

    return tuple(
        sorted(
            model_type
            for model_type, (_, _, mode, _) in _RESIDUAL_MODEL_SPECS.items()
            if mode == "multioutput"
        )
    )


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
    output_mode: ResidualOutputMode,
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

    if output_mode in {"multitask", "multioutput"}:
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
            "pretrained_output_index is only valid for multitask or multioutput residual GP models."
        )

    model_kwargs["structures"] = structures
    return model_kwargs


def _single_output_config(
    config: Any,
    *,
    family: str,
    mixed: bool,
    model_cls: type,
    model_type: str,
    model_kwargs: dict[str, Any],
    process_cat_dims: list[int],
) -> Any:
    return replace(
        config,
        task_type="regression",
        model_type=model_type,
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


def _configure_independent_multioutput(
    config: Any,
    *,
    family: str,
    mixed: bool,
    target_names: list[Any],
    process_cat_dims: list[int],
    model_kwargs: dict[str, Any],
) -> Any:
    """Build a ModelList config with one residual output and ordinary GP peers."""

    from bochan.api import MultiOutputConfig

    baseline_index = int(model_kwargs["pretrained_output_index"])
    shared_kwargs = dict(model_kwargs)
    shared_kwargs.pop("pretrained_output_index", None)

    residual_cls = _resolve_model_class(_SCALAR_RESIDUAL_MODEL_CLASSES[(family, mixed)])
    ordinary_cls = _resolve_model_class(_ORDINARY_MODEL_CLASSES[(family, mixed)])
    residual_type = f"{family}_{'mixed_' if mixed else ''}residual_gp"
    ordinary_type = f"{family}_gp"

    output_configs = []
    for index, _ in enumerate(target_names):
        if index == baseline_index:
            output_configs.append(
                _single_output_config(
                    config,
                    family=family,
                    mixed=mixed,
                    model_cls=residual_cls,
                    model_type=residual_type,
                    model_kwargs=dict(shared_kwargs),
                    process_cat_dims=process_cat_dims,
                )
            )
        else:
            output_configs.append(
                _single_output_config(
                    config,
                    family=family,
                    mixed=mixed,
                    model_cls=ordinary_cls,
                    model_type=ordinary_type,
                    model_kwargs=dict(shared_kwargs),
                    process_cat_dims=process_cat_dims,
                )
            )

    return replace(
        config,
        task_type="multi_objective",
        model_cls=None,
        model_factory=None,
        input_type="mixed" if mixed else "normal",
        cat_dims=process_cat_dims,
        input_transform=None,
        input_transform_config=None,
        pass_cat_dims=mixed,
        pass_input_transform=False,
        model_kwargs={},
        multi_output_config=MultiOutputConfig(
            output_configs=output_configs,
            output_names=[str(name) for name in target_names],
            use_hybrid=False,
        ),
    )


def configure_tabular_material_residual(owner: Any) -> bool:
    """Configure CHGNet/M3GNet/MACE residual GP structure/process contracts.

    ``multitask`` variants keep all targets in one correlated GP. ``multioutput``
    variants build independent ModelList submodels: the target selected by
    ``pretrained_output_index`` uses a residual GP while the remaining targets
    use ordinary family GP models.
    """

    config = owner.model_config
    model_type = str(config.model_type).lower()
    spec = _RESIDUAL_MODEL_SPECS.get(model_type)
    if spec is None:
        return False

    family, mixed, output_mode, class_name = spec
    if str(config.task_type) not in {"regression", "multi_objective"}:
        raise ValueError(
            f"{family.upper()} residual tabular models support regression or multi_objective only."
        )
    if config.multi_output_config is not None:
        raise ValueError(
            "Residual structure models derive scalar, correlated multitask, or independent "
            "ModelList output structure from model_type; do not provide multi_output_config."
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
    if output_mode in {"multitask", "multioutput"}:
        if n_targets < 2:
            fallback = model_type.replace("_multitask", "").replace("_multioutput", "")
            raise ValueError(
                f"{model_type} requires at least two continuous target columns. "
                f"Use model_type={fallback!r} for a single target."
            )
    elif n_targets != 1:
        raise ValueError(
            f"{model_type} is a scalar residual GP and requires exactly one target column. "
            "Use a multioutput_residual_gp or multitask_residual_gp model for wide targets."
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

    expected_input_type = "mixed" if mixed else "normal"
    if config.input_type not in (None, expected_input_type):
        raise ValueError(f"{model_type} requires input_type={expected_input_type!r}.")
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
        output_mode=output_mode,
        n_targets=n_targets,
    )

    if output_mode == "multioutput":
        owner.model_config = _configure_independent_multioutput(
            config,
            family=family,
            mixed=mixed,
            target_names=target_names,
            process_cat_dims=process_cat_dims,
            model_kwargs=model_kwargs,
        )
        return True

    if class_name is None:
        raise RuntimeError(f"No model class is registered for model_type={model_type!r}.")
    model_cls = _resolve_model_class(class_name)
    owner.model_config = replace(
        config,
        task_type="multi_objective" if output_mode == "multitask" else "regression",
        model_cls=model_cls,
        model_factory=None,
        input_type=expected_input_type,
        cat_dims=process_cat_dims,
        input_transform=None,
        input_transform_config=None,
        pass_cat_dims=mixed,
        pass_input_transform=False,
        model_kwargs=model_kwargs,
        multi_output_config=None,
    )
    return True


__all__ = [
    "configure_tabular_material_residual",
    "independent_residual_model_types",
    "material_residual_model_types",
]
