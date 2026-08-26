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
    """Configure the canonical structure-index/process tensor contract.

    The tabular data layer uses its existing category mapping machinery to map
    structure IDs to deterministic integer indices. The model layer still sees
    ``input_type='normal'`` with ``cat_dims=[]`` because the first coordinate is
    handled by ALIGNN itself and must be enumerated through fixed features.
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
        raise ValueError("Tabular ALIGNN Phase 3 does not support multi_output_config.")
    target_count = _target_count(owner.source_data_config.target_cols)
    if target_count is not None and target_count != 1:
        raise ValueError("Tabular ALIGNN models currently require exactly one target column.")
    if owner.composition.enabled:
        raise ValueError(
            "Tabular ALIGNN Phase 3 accepts crystal structure plus process variables; "
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
    other_categorical = [
        column for column in categorical_cols if column != owner.structure.column
    ]
    if other_categorical:
        raise ValueError(
            "Tabular ALIGNN Phase 3 supports continuous process variables only; "
            f"categorical process columns were configured: {other_categorical!r}."
        )
    if source.bounds is None or not isinstance(source.bounds, Mapping):
        raise ValueError(
            "Tabular ALIGNN requires column-addressed bounds for every continuous process variable."
        )
    process_columns = [
        column for column in input_cols if column != owner.structure.column
    ]
    missing_bounds = [
        column for column in process_columns if not _has_mapping_key(source.bounds, column)
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
    if config.input_type not in (None, "normal"):
        raise ValueError("Tabular ALIGNN requires input_type='normal'.")
    if config.cat_dims not in (None, [], ()):
        raise ValueError(
            "Tabular ALIGNN derives its structure selector from structure_col; "
            "do not pass cat_dims explicitly."
        )
    if config.input_transform is not None:
        raise ValueError(
            "Tabular ALIGNN derives process-only normalization automatically; "
            "do not pass input_transform explicitly in Phase 3."
        )
    transform_config = config.input_transform_config
    if transform_config is not None:
        if bool(transform_config.perturbation):
            raise ValueError("Tabular ALIGNN Phase 3 does not support input perturbation.")
        if not bool(transform_config.normalize):
            raise ValueError(
                "Tabular ALIGNN Phase 3 currently requires process normalization."
            )
        if transform_config.bounds is not None:
            raise ValueError(
                "Tabular ALIGNN derives process normalization from tabular bounds; "
                "do not pass InputTransformConfig.bounds in Phase 3."
            )

    model_kwargs = dict(config.model_kwargs)
    supplied_reserved = sorted({"structure_graphs"}.intersection(model_kwargs))
    if supplied_reserved:
        raise ValueError(
            "Tabular ALIGNN derives structure_graphs from structure_catalog; "
            f"do not pass derived model kwargs explicitly: {supplied_reserved!r}."
        )
    encoder_training = model_kwargs.pop("encoder_training", None)

    if model_type == "alignn_gp":
        if encoder_training is not None or "trainable_encoder_layers" in model_kwargs:
            raise ValueError(
                "alignn_gp freezes the structure encoder. Use model_type='alignn_dkl' "
                "for partial or full ALIGNN fine-tuning."
            )
        from bochan.models.regression.gaussian.deep import ALIGNNGPModel

        model_cls = ALIGNNGPModel
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

    model_kwargs["structure_graphs"] = owner.structure.structure_graphs
    owner.model_config = replace(
        config,
        model_cls=model_cls,
        input_type="normal",
        cat_dims=[],
        input_transform=None,
        input_transform_config=None,
        pass_cat_dims=False,
        pass_input_transform=False,
        model_kwargs=model_kwargs,
    )

    if owner.fit_config.fit_func is None:
        from bochan.fit import fit_deepkernel_mll

        owner.fit_config = replace(owner.fit_config, fit_func=fit_deepkernel_mll)


__all__ = ["configure_tabular_alignn"]
