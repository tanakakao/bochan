"""Tabular configuration for ALIGNN-GP and ALIGNN-DKL models."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

_ALIGNN_MODEL_TYPES = frozenset({"alignn_gp", "alignn_dkl"})
_DERIVED_ALIGNN_MODEL_CLASSES = frozenset(
    {
        "ALIGNNGPModel",
        "ALIGNNDKLModel",
        "ALIGNNMixedGPModel",
        "ALIGNNMixedDKLModel",
    }
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


def _has_derived_alignn_model_cls(model_cls: Any) -> bool:
    """Return whether model_cls was produced by this tabular ALIGNN adapter."""

    if model_cls is None:
        return False
    return (
        getattr(model_cls, "__name__", None) in _DERIVED_ALIGNN_MODEL_CLASSES
        and str(getattr(model_cls, "__module__", "")).startswith(
            "bochan.models.regression.gaussian.deep"
        )
    )


def _derived_multi_output_configs(multi_output_config: Any) -> list[Any] | None:
    """Return previously derived ALIGNN output configs for idempotent reconfiguration."""

    if multi_output_config is None or multi_output_config.output_configs is None:
        return None
    configs = list(multi_output_config.output_configs)
    if not configs:
        return None
    if not all(
        _has_derived_alignn_model_cls(getattr(config, "model_cls", None))
        and getattr(config, "multi_output_config", None) is None
        for config in configs
    ):
        return None
    return configs


def _clone_independent_output_config(single_output_config: Any, structure_graphs: Any) -> Any:
    """Clone trainable model components while sharing the immutable structure graph bank."""

    model_kwargs = dict(single_output_config.model_kwargs)
    model_kwargs.pop("structure_graphs", None)
    try:
        model_kwargs = copy.deepcopy(model_kwargs)
        outcome_transform = copy.deepcopy(single_output_config.outcome_transform)
    except Exception as error:
        raise TypeError(
            "Independent ALIGNN multi-output requires injected encoders, projections, "
            "transforms, and other model configuration objects to support deepcopy."
        ) from error
    model_kwargs["structure_graphs"] = structure_graphs
    return replace(
        single_output_config,
        task_type="regression",
        outcome_transform=outcome_transform,
        model_kwargs=model_kwargs,
        multi_output_config=None,
    )


def _configure_single_alignn_model(
    config: Any,
    *,
    model_type: str,
    structure_graphs: Any,
    process_cat_dims: list[int],
    expected_input_type: str,
) -> Any:
    """Resolve one canonical single-output ALIGNN submodel config."""

    is_mixed = bool(process_cat_dims)
    derived_config = _has_derived_alignn_model_cls(config.model_cls)
    if not derived_config and config.input_type not in (None, expected_input_type):
        raise ValueError(
            f"Tabular ALIGNN with the configured process columns requires "
            f"input_type={expected_input_type!r}."
        )
    configured_cat_dims = [] if config.cat_dims is None else list(config.cat_dims)
    if not derived_config and configured_cat_dims:
        raise ValueError(
            "Tabular ALIGNN derives cat_dims from categorical process columns; "
            "do not pass cat_dims explicitly."
        )
    if config.input_transform is not None:
        raise ValueError(
            "Tabular ALIGNN derives process-only normalization automatically; "
            "do not pass input_transform explicitly."
        )
    transform_config = config.input_transform_config
    if transform_config is not None:
        if bool(transform_config.perturbation):
            raise ValueError("Tabular ALIGNN does not support input perturbation.")
        if not bool(transform_config.normalize):
            raise ValueError("Tabular ALIGNN currently requires process normalization.")
        if transform_config.bounds is not None:
            raise ValueError(
                "Tabular ALIGNN derives process normalization from tabular bounds; "
                "do not pass InputTransformConfig.bounds."
            )

    model_kwargs = dict(config.model_kwargs)
    existing_structure_graphs = model_kwargs.pop("structure_graphs", None)
    if existing_structure_graphs is not None and existing_structure_graphs is not structure_graphs:
        raise ValueError(
            "Tabular ALIGNN derives structure_graphs from structure_catalog; "
            "do not override the derived graph bank in model_kwargs."
        )
    encoder_training = model_kwargs.pop("encoder_training", None)

    if model_type == "alignn_gp":
        if encoder_training is not None or "trainable_encoder_layers" in model_kwargs:
            raise ValueError(
                "alignn_gp freezes the structure encoder. Use model_type='alignn_dkl' "
                "for partial or full ALIGNN fine-tuning."
            )
        if is_mixed:
            from bochan.models.regression.gaussian.deep import ALIGNNMixedGPModel

            model_cls = ALIGNNMixedGPModel
        else:
            from bochan.models.regression.gaussian.deep import ALIGNNGPModel

            model_cls = ALIGNNGPModel
    else:
        if is_mixed:
            from bochan.models.regression.gaussian.deep import ALIGNNMixedDKLModel

            model_cls = ALIGNNMixedDKLModel
        else:
            from bochan.models.regression.gaussian.deep import ALIGNNDKLModel

            model_cls = ALIGNNDKLModel
        if encoder_training is not None:
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
                    "encoder_training must be 'partial' or 'full' for alignn_dkl."
                )

    model_kwargs["structure_graphs"] = structure_graphs
    return replace(
        config,
        task_type="regression",
        model_cls=model_cls,
        input_type=expected_input_type,
        cat_dims=process_cat_dims,
        input_transform=None,
        input_transform_config=None,
        pass_cat_dims=is_mixed,
        pass_input_transform=False,
        model_kwargs=model_kwargs,
        multi_output_config=None,
    )


def configure_tabular_alignn(owner: Any) -> None:
    """Configure the canonical structure/process and independent-output contract.

    ``structure_col`` is always model feature 0. Categorical process columns are
    excluded from the ALIGNN/process feature extractor and represented by the
    mixed GP categorical kernel. Multiple continuous target columns are modeled
    as independent ALIGNN GP/DKL submodels inside BoTorch ``ModelListGP`` so the
    standard bochan multi-objective acquisition stack can be reused unchanged.
    """

    model_type = str(owner.model_config.model_type).lower()
    if model_type not in _ALIGNN_MODEL_TYPES:
        if owner.structure.enabled:
            raise ValueError(
                "structure_col/structure_catalog are currently supported only with "
                "model_type='alignn_gp' or 'alignn_dkl'."
            )
        return

    config = owner.model_config
    derived_output_configs = _derived_multi_output_configs(config.multi_output_config)
    derived_multi_output = derived_output_configs is not None
    if str(config.task_type) not in {"regression", "multi_objective"}:
        raise ValueError(
            "Tabular ALIGNN models support regression or multi_objective regression only."
        )
    if config.multi_output_config is not None and not derived_multi_output:
        raise ValueError(
            "Tabular ALIGNN derives independent multi-output structure automatically "
            "from target_cols; do not provide multi_output_config explicitly."
        )

    target_names = _target_names(owner.source_data_config.target_cols)
    target_count = _target_count(owner.source_data_config.target_cols)
    if target_count == 0:
        raise ValueError("Tabular ALIGNN requires at least one target column.")
    if owner.composition.enabled:
        raise ValueError(
            "Tabular ALIGNN accepts crystal structure plus process variables; "
            "composition_sites cannot be combined with ALIGNN yet."
        )
    if not owner.structure.enabled:
        raise ValueError(
            f"model_type={model_type!r} requires structure_col and structure_catalog."
        )

    source = owner.source_data_config
    if source.input_cols is None:
        raise ValueError(
            "Tabular ALIGNN requires explicit input_cols so structure_col can be placed "
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
            "Tabular ALIGNN requires column-addressed bounds when bounds are supplied."
        )
    if continuous_process_cols and source.bounds is None:
        raise ValueError(
            "Tabular ALIGNN requires column-addressed bounds for every continuous process variable."
        )
    bounds_mapping = source.bounds or {}
    missing_bounds = [
        column
        for column in continuous_process_cols
        if not _has_mapping_key(bounds_mapping, column)
    ]
    if missing_bounds:
        raise ValueError(
            "Tabular ALIGNN requires bounds for every continuous process variable; "
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
    structure_graphs = owner.structure.structure_graphs

    if target_count is not None and target_count > 1:
        if target_names is None:
            raise RuntimeError("ALIGNN multi-output target names could not be resolved.")
        if derived_output_configs is not None:
            if len(derived_output_configs) != target_count:
                raise ValueError(
                    "The fitted ALIGNN multi-output config no longer matches target_cols; "
                    f"expected {target_count} outputs, got {len(derived_output_configs)}."
                )
            output_configs = [
                _configure_single_alignn_model(
                    replace(output_config, task_type="regression", multi_output_config=None),
                    model_type=model_type,
                    structure_graphs=structure_graphs,
                    process_cat_dims=process_cat_dims,
                    expected_input_type=expected_input_type,
                )
                for output_config in derived_output_configs
            ]
        else:
            single_output_config = _configure_single_alignn_model(
                replace(config, task_type="regression", multi_output_config=None),
                model_type=model_type,
                structure_graphs=structure_graphs,
                process_cat_dims=process_cat_dims,
                expected_input_type=expected_input_type,
            )
            output_configs = [
                _clone_independent_output_config(single_output_config, structure_graphs)
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
        owner.model_config = _configure_single_alignn_model(
            base_config,
            model_type=model_type,
            structure_graphs=structure_graphs,
            process_cat_dims=process_cat_dims,
            expected_input_type=expected_input_type,
        )

    from bochan.api import FitConfig
    from bochan.fit import fit_deepkernel_mll

    if owner.fit_config is None:
        owner.fit_config = FitConfig(fit_func=fit_deepkernel_mll)
    elif owner.fit_config.fit_func is None:
        owner.fit_config = replace(owner.fit_config, fit_func=fit_deepkernel_mll)


__all__ = ["configure_tabular_alignn"]
