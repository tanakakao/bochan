"""Tabular configuration for ALIGNN-GP and ALIGNN-DKL models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

_ALIGNN_MODEL_TYPES = frozenset({"alignn_gp", "alignn_dkl"})


def _target_count(target_cols: Any) -> int | None:
    if target_cols is None:
        return None
    if isinstance(target_cols, (str, bytes, int)):
        return 1
    if isinstance(target_cols, Sequence):
        return len(target_cols)
    return 1


def _has_mapping_key(mapping: Mapping[Any, Any], key: Any) -> bool:
    return key in mapping or str(key) in mapping


def configure_tabular_alignn(owner: Any) -> None:
    """Configure the canonical structure/process tensor contract.

    ``structure_col`` is always moved to model feature 0 and encoded with the
    structure catalog's deterministic ID -> graph-bank index mapping. Other
    categorical columns remain normal tabular categorical process variables.
    They are excluded from the ALIGNN/process feature extractor and passed to
    the mixed GP categorical kernel through ``cat_dims``.

    The operation is intentionally idempotent so the same contract can be
    reapplied after public ``fit(..., data_config=..., model_config=...)``
    overrides without allowing the structure-ID/category-map order to drift
    away from the cached graph-bank order.
    """

    model_type = str(owner.model_config.model_type).lower()
    if model_type not in _ALIGNN_MODEL_TYPES:
        if owner.structure.enabled:
            raise ValueError(
                "structure_col/structure_catalog are currently supported only with "
                "model_type='alignn_gp' or 'alignn_dkl'."
            )
        return

    if str(owner.model_config.task_type) != "regression":
        raise ValueError("Tabular ALIGNN models currently support regression only.")
    if owner.model_config.multi_output_config is not None:
        raise ValueError("Tabular ALIGNN currently does not support multi_output_config.")
    target_count = _target_count(owner.source_data_config.target_cols)
    if target_count is not None and target_count != 1:
        raise ValueError("Tabular ALIGNN models currently require exactly one target column.")
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

    config = owner.model_config
    process_cat_dims = [input_cols.index(column) for column in process_categorical_cols]
    is_mixed = bool(process_cat_dims)
    expected_input_type = "mixed" if is_mixed else "normal"
    if config.input_type not in (None, expected_input_type):
        raise ValueError(
            f"Tabular ALIGNN with the configured process columns requires "
            f"input_type={expected_input_type!r}."
        )
    if config.cat_dims not in (None, [], ()):
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
    structure_graphs = owner.structure.structure_graphs
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
    owner.model_config = replace(
        config,
        model_cls=model_cls,
        input_type=expected_input_type,
        cat_dims=process_cat_dims,
        input_transform=None,
        input_transform_config=None,
        pass_cat_dims=is_mixed,
        pass_input_transform=False,
        model_kwargs=model_kwargs,
    )

    from bochan.api import FitConfig
    from bochan.fit import fit_deepkernel_mll

    if owner.fit_config is None:
        owner.fit_config = FitConfig(fit_func=fit_deepkernel_mll)
    elif owner.fit_config.fit_func is None:
        owner.fit_config = replace(owner.fit_config, fit_func=fit_deepkernel_mll)


__all__ = ["configure_tabular_alignn"]
