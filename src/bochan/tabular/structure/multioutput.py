"""Dataset-aware output resolution for tabular ALIGNN models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .fitting import (
    _ALIGNN_CORRELATED_MULTITASK_MODEL_TYPES,
    _ALIGNN_MODEL_TYPES,
    _clone_independent_output_config,
    _configure_correlated_alignn_model,
    _configure_single_alignn_model,
    _derived_multi_output_configs,
)


def _dataset_output_names(dataset: Any, n_outputs: int) -> list[Any]:
    """Resolve target metadata against the authoritative fitted target width."""

    names = list(getattr(dataset, "target_names", None) or [])
    if not names:
        names = [f"y{index}" for index in range(n_outputs)]
        dataset.target_names = names
        return names
    if len(names) != n_outputs:
        raise ValueError(
            "ALIGNN target metadata must match the fitted target tensor width: "
            f"{len(names)} names for {n_outputs} outputs."
        )
    return names


def configure_alignn_outputs_from_dataset(owner: Any, dataset: Any) -> None:
    """Reconcile ALIGNN output config with authoritative ``dataset.Y`` width.

    Independent ``alignn_gp`` / ``alignn_dkl`` models expand to one submodel per
    output. Correlated ``alignn_multitask`` / ``alignn_multitask_dkl`` models keep
    the wide target tensor in one shared-backbone GP and require at least two
    targets. Array-based ``fit(X, y)`` therefore follows the same contract as
    DataFrame input even when target names were omitted at construction time.
    """

    model_type = str(owner.model_config.model_type).lower()
    if model_type not in _ALIGNN_MODEL_TYPES:
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
            "Tabular ALIGNN targets must have shape [n] or [n, m]; "
            f"got {tuple(Y.shape)}."
        )
    if n_outputs < 1:
        raise ValueError("Tabular ALIGNN requires at least one target output.")

    target_names = _dataset_output_names(dataset, n_outputs)
    config = owner.model_config
    correlated = model_type in _ALIGNN_CORRELATED_MULTITASK_MODEL_TYPES
    process_cat_dims = [int(index) for index in (config.cat_dims or [])]
    expected_input_type = "mixed" if process_cat_dims else "normal"
    structure_graphs = owner.structure.structure_graphs

    if correlated:
        if config.multi_output_config is not None:
            raise ValueError(
                "Correlated ALIGNN multitask models keep wide targets in one model; "
                "do not provide multi_output_config."
            )
        if n_outputs < 2:
            fallback = "alignn_dkl" if model_type.endswith("_dkl") else "alignn_gp"
            raise ValueError(
                f"{model_type} requires at least two continuous target columns. "
                f"Use model_type={fallback!r} for a single target."
            )
        owner.model_config = _configure_correlated_alignn_model(
            replace(config, task_type="multi_objective", multi_output_config=None),
            model_type=model_type,
            structure_graphs=structure_graphs,
            process_cat_dims=process_cat_dims,
            expected_input_type=expected_input_type,
        )
        return

    derived_output_configs = _derived_multi_output_configs(config.multi_output_config)
    if config.multi_output_config is not None and derived_output_configs is None:
        raise ValueError(
            "Tabular ALIGNN derives independent multi-output structure automatically; "
            "do not provide multi_output_config explicitly."
        )

    if n_outputs > 1:
        if derived_output_configs is not None:
            if len(derived_output_configs) != n_outputs:
                raise ValueError(
                    "The derived ALIGNN output configuration does not match the fitted "
                    f"target tensor width: {len(derived_output_configs)} != {n_outputs}."
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
                "The derived ALIGNN output configuration does not match the fitted "
                f"single target: {len(derived_output_configs)} outputs configured."
            )
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


__all__ = ["configure_alignn_outputs_from_dataset"]
