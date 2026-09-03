"""Tabular routing for cross-family multiple pretrained material baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from bochan.models.regression.gaussian.materials.common import (
    MaterialBaselinePlan,
    MaterialBaselineSpec,
    MultipleBaselineModelListGP,
    ResidualMaterialGPModel,
)

from .material_residual import (
    _ORDINARY_MODEL_CLASSES,
    _SCALAR_RESIDUAL_MODEL_CLASSES,
    _has_mapping_key,
    _resolve_model_class,
    _single_output_config,
    _target_names,
)

_MULTI_BASELINE_MODEL_TYPES = frozenset(
    {
        "material_multi_baseline_residual_gp",
        "material_mixed_multi_baseline_residual_gp",
    }
)
_SUPPORTED_FAMILIES = frozenset({"chgnet", "m3gnet", "mace"})


def multiple_baseline_residual_model_types() -> tuple[str, ...]:
    """Return public model types for cross-family independent residual outputs."""

    return tuple(sorted(_MULTI_BASELINE_MODEL_TYPES))


def _resolve_baseline_routes(
    raw_routes: Any,
    *,
    output_names: list[str],
) -> tuple[MaterialBaselinePlan, dict[int, dict[str, Any]]]:
    if isinstance(raw_routes, (str, bytes)) or not isinstance(raw_routes, Sequence):
        raise ValueError("baseline_routes must be a non-empty sequence of route mappings.")
    routes = list(raw_routes)
    if not routes:
        raise ValueError("baseline_routes must contain at least one enabled baseline route.")

    specs: list[MaterialBaselineSpec] = []
    route_kwargs_by_spec: list[tuple[MaterialBaselineSpec, dict[str, Any]]] = []
    for raw in routes:
        if not isinstance(raw, Mapping):
            raise TypeError("Each baseline_routes entry must be a mapping.")
        spec = raw.get("spec")
        if not isinstance(spec, MaterialBaselineSpec):
            raise TypeError("baseline_routes[].spec must be a MaterialBaselineSpec.")
        model_kwargs = raw.get("model_kwargs", {})
        if not isinstance(model_kwargs, Mapping):
            raise TypeError("baseline_routes[].model_kwargs must be a mapping.")
        specs.append(spec)
        route_kwargs_by_spec.append((spec, dict(model_kwargs)))

    plan = MaterialBaselinePlan.resolve(
        output_names=output_names,
        baseline_specs=specs,
    )
    kwargs_by_index: dict[int, dict[str, Any]] = {}
    for assignment in plan.assignments:
        for spec, model_kwargs in route_kwargs_by_spec:
            if spec == assignment.spec:
                kwargs_by_index[assignment.output_index] = dict(model_kwargs)
                break
    return plan, kwargs_by_index


def _validate_family_kwargs(
    family: str,
    model_kwargs: Mapping[str, Any] | None,
    *,
    structures: tuple[Any, ...],
) -> dict[str, Any]:
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported material baseline family: {family!r}.")
    resolved = dict(model_kwargs or {})
    forbidden = {
        "encoder",
        "adapter",
        "structures",
        "structure_graphs",
        "graph_converter",
        "batch_builder",
        "baseline_spec",
    }
    injected = sorted(forbidden.intersection(resolved))
    if injected:
        raise ValueError(
            "Multiple-baseline routing derives encoder/structure/baseline objects internally; "
            f"do not provide {injected!r}."
        )
    if "encoder_training" in resolved or "trainable_encoder_layers" in resolved:
        raise ValueError(
            "Residual GP baselines remain frozen; encoder training controls are not supported."
        )
    resolved["structures"] = structures
    return resolved


def _build_multiple_baseline_wrapper(
    *,
    submodels: Sequence[Any],
    output_configs: Sequence[Any],
    config: Any,
) -> MultipleBaselineModelListGP:
    """Attach resolved baseline specs and build the validated ModelList wrapper."""

    del output_configs
    output_names = list(config.wrapper_kwargs["output_names"])
    baseline_specs = list(config.wrapper_kwargs["baseline_specs"])
    plan = MaterialBaselinePlan.resolve(
        output_names=output_names,
        baseline_specs=baseline_specs,
    )
    for assignment in plan.assignments:
        model = submodels[assignment.output_index]
        if not isinstance(model, ResidualMaterialGPModel):
            raise TypeError(
                f"Output {assignment.output_name!r} is baseline-assigned but did not build "
                "a ResidualMaterialGPModel."
            )
        model.baseline_spec = assignment.spec
    return MultipleBaselineModelListGP(
        *submodels,
        output_names=output_names,
        baseline_specs=baseline_specs,
    )


def configure_tabular_multiple_baseline(owner: Any) -> bool:
    """Configure independent outputs with zero, one, or many pretrained baselines."""

    config = owner.model_config
    model_type = str(config.model_type).lower()
    if model_type not in _MULTI_BASELINE_MODEL_TYPES:
        return False

    mixed = model_type == "material_mixed_multi_baseline_residual_gp"
    if str(config.task_type) not in {"regression", "multi_objective"}:
        raise ValueError("Multiple material baselines support regression or multi_objective only.")
    if config.multi_output_config is not None:
        raise ValueError(
            "Multiple-baseline output structure is derived from baseline_routes; "
            "do not provide multi_output_config explicitly."
        )
    if owner.composition.enabled:
        raise ValueError("Multiple structure baselines cannot be combined with composition_sites.")
    if not owner.structure.enabled:
        raise ValueError(
            f"model_type={model_type!r} requires structure_col and structure_catalog."
        )
    if owner.structure.graph_builder is not None:
        raise ValueError("structure_graph_builder is ALIGNN-specific and must be omitted.")

    target_names_raw = _target_names(owner.source_data_config.target_cols)
    if not target_names_raw or len(target_names_raw) < 2:
        raise ValueError("Multiple-baseline residual routing requires at least two target columns.")
    target_names = [str(name) for name in target_names_raw]

    source = owner.source_data_config
    if source.input_cols is None:
        raise ValueError(
            "Multiple-baseline structure models require explicit input_cols so structure_col "
            "can be placed at feature index 0."
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
        counterpart = (
            "material_multi_baseline_residual_gp"
            if mixed
            else "material_mixed_multi_baseline_residual_gp"
        )
        if mixed:
            raise ValueError(
                f"{model_type} requires categorical process columns. "
                f"Use model_type={counterpart!r} for continuous-only process inputs."
            )
        raise ValueError(
            f"{model_type} does not accept categorical process columns. "
            f"Use model_type={counterpart!r} for mixed process inputs."
        )

    expected_input_type = "mixed" if mixed else "normal"
    if config.input_type not in (None, expected_input_type):
        raise ValueError(f"{model_type} requires input_type={expected_input_type!r}.")
    if config.cat_dims:
        raise ValueError("cat_dims are derived from categorical process columns; omit them.")
    if config.input_transform is not None:
        raise ValueError("Multiple-baseline routing derives process normalization internally.")
    if config.input_transform_config is not None:
        tf = config.input_transform_config
        if bool(tf.perturbation) or not bool(tf.normalize) or tf.bounds is not None:
            raise ValueError(
                "Multiple-baseline routing requires default normalization without perturbation."
            )

    if source.bounds is not None and not isinstance(source.bounds, Mapping):
        raise ValueError("Multiple-baseline routing requires column-addressed bounds.")
    if continuous_process_cols and source.bounds is None:
        raise ValueError("Bounds are required for every continuous process variable.")
    bounds_mapping = source.bounds or {}
    missing_bounds = [
        column for column in continuous_process_cols if not _has_mapping_key(bounds_mapping, column)
    ]
    if missing_bounds:
        raise ValueError(f"Missing bounds for continuous process columns: {missing_bounds!r}.")

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

    model_kwargs = dict(config.model_kwargs)
    raw_routes = model_kwargs.pop("baseline_routes", None)
    ordinary_family = model_kwargs.pop("ordinary_family", None)
    ordinary_model_kwargs = model_kwargs.pop("ordinary_model_kwargs", {})
    if model_kwargs:
        raise ValueError(
            "Multiple-baseline model_kwargs only accepts baseline_routes, ordinary_family, "
            f"and ordinary_model_kwargs; unexpected keys: {sorted(model_kwargs)!r}."
        )

    plan, baseline_kwargs_by_index = _resolve_baseline_routes(
        raw_routes,
        output_names=target_names,
    )
    if plan.ordinary_output_indices:
        if not isinstance(ordinary_family, str) or ordinary_family.casefold() not in _SUPPORTED_FAMILIES:
            raise ValueError(
                "ordinary_family must be one of 'chgnet', 'm3gnet', or 'mace' when "
                "one or more outputs do not have a pretrained baseline."
            )
        ordinary_family = ordinary_family.casefold()
    elif ordinary_family is not None:
        if not isinstance(ordinary_family, str) or ordinary_family.casefold() not in _SUPPORTED_FAMILIES:
            raise ValueError("ordinary_family must be one of 'chgnet', 'm3gnet', or 'mace'.")
        ordinary_family = ordinary_family.casefold()

    if not isinstance(ordinary_model_kwargs, Mapping):
        raise TypeError("ordinary_model_kwargs must be a mapping.")

    structures = owner.structure.structures
    process_cat_dims = [input_cols.index(column) for column in process_categorical_cols]
    output_configs = []
    for index, output_name in enumerate(target_names):
        assignment = plan.assignment_for_output(index)
        if assignment is not None:
            family = assignment.spec.family
            family_kwargs = _validate_family_kwargs(
                family,
                baseline_kwargs_by_index.get(index),
                structures=structures,
            )
            if assignment.spec.model_name is not None:
                existing_model_name = family_kwargs.get("model_name")
                if existing_model_name is not None and existing_model_name != assignment.spec.model_name:
                    raise ValueError(
                        f"Baseline model_name mismatch for output {output_name!r}: "
                        f"{existing_model_name!r} != {assignment.spec.model_name!r}."
                    )
                family_kwargs["model_name"] = assignment.spec.model_name
            model_cls = _resolve_model_class(_SCALAR_RESIDUAL_MODEL_CLASSES[(family, mixed)])
            model_type_for_output = f"{family}_{'mixed_' if mixed else ''}residual_gp"
        else:
            family = str(ordinary_family)
            family_kwargs = _validate_family_kwargs(
                family,
                ordinary_model_kwargs,
                structures=structures,
            )
            model_cls = _resolve_model_class(_ORDINARY_MODEL_CLASSES[(family, mixed)])
            model_type_for_output = f"{family}_gp"

        output_configs.append(
            _single_output_config(
                config,
                family=family,
                mixed=mixed,
                model_cls=model_cls,
                model_type=model_type_for_output,
                model_kwargs=family_kwargs,
                process_cat_dims=process_cat_dims,
            )
        )

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
        pass_cat_dims=mixed,
        pass_input_transform=False,
        model_kwargs={},
        multi_output_config=MultiOutputConfig(
            output_configs=output_configs,
            output_names=target_names,
            wrapper_factory=_build_multiple_baseline_wrapper,
            wrapper_kwargs={
                "output_names": target_names,
                "baseline_specs": [assignment.spec for assignment in plan.assignments],
            },
            use_hybrid=False,
        ),
    )
    return True


__all__ = [
    "configure_tabular_multiple_baseline",
    "multiple_baseline_residual_model_types",
]
