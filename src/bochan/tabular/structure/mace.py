"""Tabular routing for MACE structure/process GP, DKL, and multitask models."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

_MACE_INDEPENDENT_MODEL_TYPES = frozenset({"mace_gp", "mace_dkl"})
_MACE_CORRELATED_MULTITASK_MODEL_TYPES = frozenset(
    {"mace_multitask", "mace_multitask_dkl"}
)
_MACE_MODEL_TYPES = _MACE_INDEPENDENT_MODEL_TYPES | _MACE_CORRELATED_MULTITASK_MODEL_TYPES
_DERIVED_INDEPENDENT_MACE_MODEL_CLASSES = frozenset(
    {
        "MACEGPModel",
        "MACEDKLModel",
        "MACEMixedGPModel",
        "MACEMixedDKLModel",
    }
)
_DERIVED_CORRELATED_MACE_MODEL_CLASSES = frozenset(
    {
        "MACEMultiTaskGPModel",
        "MACEMultiTaskDKLModel",
        "MACEMixedMultiTaskGPModel",
        "MACEMixedMultiTaskDKLModel",
    }
)
_DERIVED_MACE_MODEL_CLASSES = (
    _DERIVED_INDEPENDENT_MACE_MODEL_CLASSES | _DERIVED_CORRELATED_MACE_MODEL_CLASSES
)


def _target_names(target_cols: Any) -> list[Any] | None:
    if target_cols is None:
        return None
    if isinstance(target_cols, (str, bytes, int)):
        return [target_cols]
    if isinstance(target_cols, Sequence):
        return list(target_cols)
    return [target_cols]


def _target_count(target_cols: Any) -> int | None:
    names = _target_names(target_cols)
    return None if names is None else len(names)


def _has_mapping_key(mapping: Mapping[Any, Any], key: Any) -> bool:
    return key in mapping or str(key) in mapping


def _has_derived_mace_model_cls(model_cls: Any) -> bool:
    if model_cls is None:
        return False
    return (
        getattr(model_cls, "__name__", None) in _DERIVED_MACE_MODEL_CLASSES
        and str(getattr(model_cls, "__module__", "")).startswith(
            "bochan.models.regression.gaussian.deep"
        )
    )


def _derived_multi_output_configs(multi_output_config: Any) -> list[Any] | None:
    """Return previously derived independent MACE output configurations."""

    if multi_output_config is None or multi_output_config.output_configs is None:
        return None
    configs = list(multi_output_config.output_configs)
    if not configs:
        return None
    if not all(
        getattr(getattr(config, "model_cls", None), "__name__", None)
        in _DERIVED_INDEPENDENT_MACE_MODEL_CLASSES
        and str(getattr(getattr(config, "model_cls", None), "__module__", "")).startswith(
            "bochan.models.regression.gaussian.deep"
        )
        and getattr(config, "multi_output_config", None) is None
        for config in configs
    ):
        return None
    return configs


def _clone_independent_output_config(single_output_config: Any, structures: Any) -> Any:
    """Clone trainable model state while sharing the immutable raw structure bank."""

    model_kwargs = dict(single_output_config.model_kwargs)
    model_kwargs.pop("structures", None)
    try:
        model_kwargs = copy.deepcopy(model_kwargs)
        outcome_transform = copy.deepcopy(single_output_config.outcome_transform)
    except Exception as error:
        raise TypeError(
            "Independent MACE multi-output requires injected encoders, projections, "
            "transforms, and other model configuration objects to support deepcopy."
        ) from error
    model_kwargs["structures"] = structures
    return replace(
        single_output_config,
        task_type="regression",
        outcome_transform=outcome_transform,
        model_kwargs=model_kwargs,
        multi_output_config=None,
    )


def _validate_model_config(
    config: Any,
    *,
    structures: tuple[Any, ...],
    process_cat_dims: list[int],
    expected_input_type: str,
) -> tuple[bool, dict[str, Any], Any | None]:
    """Validate shared tabular MACE configuration and return mutable kwargs."""

    is_mixed = bool(process_cat_dims)
    derived_config = _has_derived_mace_model_cls(config.model_cls)
    if not derived_config and config.input_type not in (None, expected_input_type):
        raise ValueError(
            "Tabular MACE with the configured process columns requires "
            f"input_type={expected_input_type!r}."
        )
    configured_cat_dims = [] if config.cat_dims is None else list(config.cat_dims)
    if not derived_config and configured_cat_dims:
        raise ValueError(
            "Tabular MACE derives cat_dims from categorical process columns; "
            "do not pass cat_dims explicitly."
        )
    if config.input_transform is not None:
        raise ValueError(
            "Tabular MACE derives process-only normalization automatically; "
            "do not pass input_transform explicitly."
        )
    transform_config = config.input_transform_config
    if transform_config is not None:
        if bool(transform_config.perturbation):
            raise ValueError("Tabular MACE does not support input perturbation.")
        if not bool(transform_config.normalize):
            raise ValueError("Tabular MACE currently requires process normalization.")
        if transform_config.bounds is not None:
            raise ValueError(
                "Tabular MACE derives process normalization from tabular bounds; "
                "do not pass InputTransformConfig.bounds."
            )

    model_kwargs = dict(config.model_kwargs)
    existing_structures = model_kwargs.pop("structures", None)
    if existing_structures is not None and existing_structures is not structures:
        raise ValueError(
            "Tabular MACE derives structures from structure_catalog; "
            "do not override the derived structure bank in model_kwargs."
        )
    encoder_training = model_kwargs.pop("encoder_training", None)
    return is_mixed, model_kwargs, encoder_training


def _apply_encoder_training_policy(
    model_kwargs: dict[str, Any],
    encoder_training: Any | None,
    *,
    frozen_model_type: str,
    dkl_model_type: str,
    dkl: bool,
) -> None:
    if not dkl:
        if encoder_training is not None or "trainable_encoder_layers" in model_kwargs:
            raise ValueError(
                f"{frozen_model_type} freezes the MACE structure encoder. Use "
                f"model_type={dkl_model_type!r} for partial or full fine-tuning."
            )
        return
    if encoder_training is None:
        return
    if "trainable_encoder_layers" in model_kwargs:
        raise ValueError(
            "Specify either encoder_training or trainable_encoder_layers, not both."
        )
    normalized = str(encoder_training).lower()
    if normalized == "partial":
        model_kwargs["trainable_encoder_layers"] = 1
    elif normalized == "full":
        model_kwargs["trainable_encoder_layers"] = "all"
    else:
        raise ValueError(
            f"encoder_training must be 'partial' or 'full' for {dkl_model_type}."
        )


def _configure_single_model(
    config: Any,
    *,
    model_type: str,
    structures: tuple[Any, ...],
    process_cat_dims: list[int],
    expected_input_type: str,
) -> Any:
    """Resolve one independent single-output MACE model configuration."""

    if model_type not in _MACE_INDEPENDENT_MODEL_TYPES:
        raise ValueError(f"Unsupported independent MACE model_type={model_type!r}.")
    is_mixed, model_kwargs, encoder_training = _validate_model_config(
        config,
        structures=structures,
        process_cat_dims=process_cat_dims,
        expected_input_type=expected_input_type,
    )
    dkl = model_type == "mace_dkl"
    _apply_encoder_training_policy(
        model_kwargs,
        encoder_training,
        frozen_model_type="mace_gp",
        dkl_model_type="mace_dkl",
        dkl=dkl,
    )

    if dkl:
        if is_mixed:
            from bochan.models.regression.gaussian.deep import MACEMixedDKLModel

            model_cls = MACEMixedDKLModel
        else:
            from bochan.models.regression.gaussian.deep import MACEDKLModel

            model_cls = MACEDKLModel
    elif is_mixed:
        from bochan.models.regression.gaussian.deep import MACEMixedGPModel

        model_cls = MACEMixedGPModel
    else:
        from bochan.models.regression.gaussian.deep import MACEGPModel

        model_cls = MACEGPModel

    model_kwargs["structures"] = structures
    return replace(
        config,
        task_type="regression",
        model_cls=model_cls,
        model_factory=None,
        input_type=expected_input_type,
        cat_dims=process_cat_dims,
        input_transform=None,
        input_transform_config=None,
        pass_cat_dims=is_mixed,
        pass_input_transform=False,
        model_kwargs=model_kwargs,
        multi_output_config=None,
    )


def _configure_correlated_model(
    config: Any,
    *,
    model_type: str,
    structures: tuple[Any, ...],
    process_cat_dims: list[int],
    expected_input_type: str,
) -> Any:
    """Resolve one shared-backbone correlated MACE multitask configuration."""

    if model_type not in _MACE_CORRELATED_MULTITASK_MODEL_TYPES:
        raise ValueError(f"Unsupported correlated MACE model_type={model_type!r}.")
    is_mixed, model_kwargs, encoder_training = _validate_model_config(
        config,
        structures=structures,
        process_cat_dims=process_cat_dims,
        expected_input_type=expected_input_type,
    )
    dkl = model_type == "mace_multitask_dkl"
    _apply_encoder_training_policy(
        model_kwargs,
        encoder_training,
        frozen_model_type="mace_multitask",
        dkl_model_type="mace_multitask_dkl",
        dkl=dkl,
    )

    if dkl:
        if is_mixed:
            from bochan.models.regression.gaussian.deep import MACEMixedMultiTaskDKLModel

            model_cls = MACEMixedMultiTaskDKLModel
        else:
            from bochan.models.regression.gaussian.deep import MACEMultiTaskDKLModel

            model_cls = MACEMultiTaskDKLModel
    elif is_mixed:
        from bochan.models.regression.gaussian.deep import MACEMixedMultiTaskGPModel

        model_cls = MACEMixedMultiTaskGPModel
    else:
        from bochan.models.regression.gaussian.deep import MACEMultiTaskGPModel

        model_cls = MACEMultiTaskGPModel

    model_kwargs["structures"] = structures
    return replace(
        config,
        task_type="multi_objective",
        model_cls=model_cls,
        model_factory=None,
        input_type=expected_input_type,
        cat_dims=process_cat_dims,
        input_transform=None,
        input_transform_config=None,
        pass_cat_dims=is_mixed,
        pass_input_transform=False,
        model_kwargs=model_kwargs,
        multi_output_config=None,
    )


def configure_tabular_mace(owner: Any) -> bool:
    """Configure canonical MACE structure/process and output-dependency contracts."""

    model_type = str(owner.model_config.model_type).lower()
    if model_type not in _MACE_MODEL_TYPES:
        return False

    config = owner.model_config
    correlated = model_type in _MACE_CORRELATED_MULTITASK_MODEL_TYPES
    derived_output_configs = _derived_multi_output_configs(config.multi_output_config)
    derived_multi_output = derived_output_configs is not None
    if str(config.task_type) not in {"regression", "multi_objective"}:
        raise ValueError(
            "Tabular MACE models support regression or multi_objective regression only."
        )
    if correlated and config.multi_output_config is not None:
        raise ValueError(
            "Correlated MACE multitask models keep wide targets in one model; "
            "do not provide multi_output_config."
        )
    if not correlated and config.multi_output_config is not None and not derived_multi_output:
        raise ValueError(
            "Tabular MACE derives independent multi-output structure automatically "
            "from target_cols; do not provide multi_output_config explicitly."
        )

    target_names = _target_names(owner.source_data_config.target_cols)
    target_count = _target_count(owner.source_data_config.target_cols)
    if target_count == 0:
        raise ValueError("Tabular MACE requires at least one target column.")
    if correlated and target_count == 1:
        fallback = "mace_dkl" if model_type.endswith("_dkl") else "mace_gp"
        raise ValueError(
            f"{model_type} requires at least two continuous target columns. "
            f"Use model_type={fallback!r} for a single target."
        )
    if owner.composition.enabled:
        raise ValueError(
            "Tabular MACE accepts crystal structure plus process variables; "
            "composition_sites cannot be combined with MACE yet."
        )
    if not owner.structure.enabled:
        raise ValueError(
            f"model_type={model_type!r} requires structure_col and structure_catalog."
        )
    if owner.structure.graph_builder is not None:
        raise ValueError(
            "structure_graph_builder is ALIGNN-specific and must be omitted for MACE."
        )

    source = owner.source_data_config
    if source.input_cols is None:
        raise ValueError(
            "Tabular MACE requires explicit input_cols so structure_col can be placed "
            "at model feature index 0."
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

    if source.bounds is not None and not isinstance(source.bounds, Mapping):
        raise ValueError(
            "Tabular MACE requires column-addressed bounds when bounds are supplied."
        )
    if continuous_process_cols and source.bounds is None:
        raise ValueError(
            "Tabular MACE requires column-addressed bounds for every continuous process variable."
        )
    bounds_mapping = source.bounds or {}
    missing_bounds = [
        column
        for column in continuous_process_cols
        if not _has_mapping_key(bounds_mapping, column)
    ]
    if missing_bounds:
        raise ValueError(
            "Tabular MACE requires bounds for every continuous process variable; "
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
    expected_input_type = "mixed" if process_cat_dims else "normal"
    structures = owner.structure.structures

    if correlated:
        owner.model_config = _configure_correlated_model(
            replace(config, multi_output_config=None),
            model_type=model_type,
            structures=structures,
            process_cat_dims=process_cat_dims,
            expected_input_type=expected_input_type,
        )
    elif target_count is not None and target_count > 1:
        if target_names is None:
            raise RuntimeError("MACE multi-output target names could not be resolved.")
        if derived_output_configs is not None:
            if len(derived_output_configs) != target_count:
                raise ValueError(
                    "The fitted MACE multi-output config no longer matches target_cols; "
                    f"expected {target_count} outputs, got {len(derived_output_configs)}."
                )
            output_configs = [
                _configure_single_model(
                    replace(output_config, task_type="regression", multi_output_config=None),
                    model_type=model_type,
                    structures=structures,
                    process_cat_dims=process_cat_dims,
                    expected_input_type=expected_input_type,
                )
                for output_config in derived_output_configs
            ]
        else:
            single_output_config = _configure_single_model(
                replace(config, task_type="regression", multi_output_config=None),
                model_type=model_type,
                structures=structures,
                process_cat_dims=process_cat_dims,
                expected_input_type=expected_input_type,
            )
            output_configs = [
                _clone_independent_output_config(single_output_config, structures)
                for _ in target_names
            ]

        from bochan.api import MultiOutputConfig

        owner.model_config = replace(
            config,
            task_type="multi_objective",
            model_cls=None,
            model_factory=None,
            input_type=expected_input_type,
            cat_dims=process_cat_dims,
            input_transform=None,
            input_transform_config=None,
            pass_cat_dims=bool(process_cat_dims),
            pass_input_transform=False,
            model_kwargs={},
            multi_output_config=MultiOutputConfig(
                output_configs=output_configs,
                output_names=[str(name) for name in target_names],
                use_hybrid=False,
            ),
        )
    else:
        if derived_output_configs is not None:
            base_config = derived_output_configs[0]
        else:
            base_config = replace(config, task_type="regression", multi_output_config=None)
        owner.model_config = _configure_single_model(
            base_config,
            model_type=model_type,
            structures=structures,
            process_cat_dims=process_cat_dims,
            expected_input_type=expected_input_type,
        )

    from bochan.api import FitConfig
    from bochan.fit import fit_deepkernel_mll

    if owner.fit_config is None:
        owner.fit_config = FitConfig(fit_func=fit_deepkernel_mll)
    elif owner.fit_config.fit_func is None:
        owner.fit_config = replace(owner.fit_config, fit_func=fit_deepkernel_mll)
    return True


def _dataset_output_names(dataset: Any, n_outputs: int) -> list[Any]:
    """Resolve target metadata against the authoritative fitted target width."""

    names = list(getattr(dataset, "target_names", None) or [])
    if not names:
        names = [f"y{index}" for index in range(n_outputs)]
        dataset.target_names = names
        return names
    if len(names) != n_outputs:
        raise ValueError(
            "MACE target metadata must match the fitted target tensor width: "
            f"{len(names)} names for {n_outputs} outputs."
        )
    return names


def configure_mace_outputs_from_dataset(owner: Any, dataset: Any) -> None:
    """Reconcile MACE output routing with the authoritative ``dataset.Y`` width."""

    model_type = str(owner.model_config.model_type).lower()
    if model_type not in _MACE_MODEL_TYPES:
        return
    Y = getattr(dataset, "Y", None)
    if Y is None:
        return
    if Y.ndim == 1:
        n_outputs = 1
    elif Y.ndim == 2:
        n_outputs = int(Y.shape[-1])
    else:
        raise ValueError(
            "Tabular MACE targets must have shape [n] or [n, m]; "
            f"got {tuple(Y.shape)}."
        )
    if n_outputs < 1:
        raise ValueError("Tabular MACE requires at least one target output.")

    target_names = _dataset_output_names(dataset, n_outputs)
    config = owner.model_config
    correlated = model_type in _MACE_CORRELATED_MULTITASK_MODEL_TYPES
    process_cat_dims = [int(index) for index in (config.cat_dims or [])]
    expected_input_type = "mixed" if process_cat_dims else "normal"
    structures = owner.structure.structures

    if correlated:
        if config.multi_output_config is not None:
            raise ValueError(
                "Correlated MACE multitask models keep wide targets in one model; "
                "do not provide multi_output_config."
            )
        if n_outputs < 2:
            fallback = "mace_dkl" if model_type.endswith("_dkl") else "mace_gp"
            raise ValueError(
                f"{model_type} requires at least two continuous target columns. "
                f"Use model_type={fallback!r} for a single target."
            )
        owner.model_config = _configure_correlated_model(
            replace(config, task_type="multi_objective", multi_output_config=None),
            model_type=model_type,
            structures=structures,
            process_cat_dims=process_cat_dims,
            expected_input_type=expected_input_type,
        )
        return

    derived_output_configs = _derived_multi_output_configs(config.multi_output_config)
    if config.multi_output_config is not None and derived_output_configs is None:
        raise ValueError(
            "Tabular MACE derives independent multi-output structure automatically; "
            "do not provide multi_output_config explicitly."
        )

    if n_outputs > 1:
        if derived_output_configs is not None:
            if len(derived_output_configs) != n_outputs:
                raise ValueError(
                    "The derived MACE output configuration does not match the fitted "
                    f"target tensor width: {len(derived_output_configs)} != {n_outputs}."
                )
            output_configs = [
                _configure_single_model(
                    replace(output_config, task_type="regression", multi_output_config=None),
                    model_type=model_type,
                    structures=structures,
                    process_cat_dims=process_cat_dims,
                    expected_input_type=expected_input_type,
                )
                for output_config in derived_output_configs
            ]
        else:
            single_output_config = _configure_single_model(
                replace(config, task_type="regression", multi_output_config=None),
                model_type=model_type,
                structures=structures,
                process_cat_dims=process_cat_dims,
                expected_input_type=expected_input_type,
            )
            output_configs = [
                _clone_independent_output_config(single_output_config, structures)
                for _ in range(n_outputs)
            ]

        from bochan.api import MultiOutputConfig

        owner.model_config = replace(
            config,
            task_type="multi_objective",
            model_cls=None,
            model_factory=None,
            input_type=expected_input_type,
            cat_dims=process_cat_dims,
            input_transform=None,
            input_transform_config=None,
            pass_cat_dims=bool(process_cat_dims),
            pass_input_transform=False,
            model_kwargs={},
            multi_output_config=MultiOutputConfig(
                output_configs=output_configs,
                output_names=[str(name) for name in target_names],
                use_hybrid=False,
            ),
        )
        return

    if derived_output_configs is not None:
        if len(derived_output_configs) != 1:
            raise ValueError(
                "The derived MACE output configuration does not match the fitted "
                f"single target: {len(derived_output_configs)} outputs configured."
            )
        base_config = derived_output_configs[0]
    else:
        base_config = replace(config, task_type="regression", multi_output_config=None)
    owner.model_config = _configure_single_model(
        base_config,
        model_type=model_type,
        structures=structures,
        process_cat_dims=process_cat_dims,
        expected_input_type=expected_input_type,
    )


__all__ = [
    "_MACE_CORRELATED_MULTITASK_MODEL_TYPES",
    "_MACE_MODEL_TYPES",
    "configure_mace_outputs_from_dataset",
    "configure_tabular_mace",
]
