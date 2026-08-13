"""Acquisition context helpers shared by public optimization services."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

from ..configs import AcquisitionConfig, ModelBundle, ModelConfig


def _filter_context_fields_for_acqf(config: AcquisitionConfig) -> AcquisitionConfig:
    """Keep only context fields explicitly accepted by the acquisition class.

    Some acquisition classes accept ``**kwargs`` and forward them to BoTorch /
    GPyTorch base classes. Passing automatic context fields such as
    ``X_baseline`` to those classes can fail with errors like
    ``MCAcquisitionFunction.__init__() got an unexpected keyword argument``.

    Explicit ``acqf_kwargs`` are preserved. This helper filters only the
    automatically injected fields from ``DataContext``.
    """
    if config.acqf_cls is None or not config.filter_kwargs_by_signature:
        return config
    try:
        signature = inspect.signature(config.acqf_cls)
    except (TypeError, ValueError):
        return config
    explicit_params = set(signature.parameters)
    filtered_fields = tuple(field for field in config.context_fields if field in explicit_params)
    if filtered_fields == config.context_fields:
        return config
    return replace(config, context_fields=filtered_fields)


def _input_transform_n_w_from_model_config(model_config: ModelConfig | None) -> int | None:
    """ModelConfig.input_transform_config から perturbation 用 n_w を取り出す。"""
    if model_config is None:
        return None
    transform_config = getattr(model_config, "input_transform_config", None)
    if transform_config is None:
        return None
    if not bool(getattr(transform_config, "perturbation", False)):
        return None
    n_w = getattr(transform_config, "n_w", None)
    return None if n_w is None else int(n_w)


def _safe_output_index(output: Any | None) -> int | None:
    if output is None or isinstance(output, str):
        return None
    try:
        return int(output)
    except (TypeError, ValueError):
        return None


def _input_transform_n_w_from_bundle(bundle: ModelBundle | None, output: Any | None = None) -> int | None:
    """ObjectiveConfig.n_w の未指定時に bundle の input_transform 設定から n_w を推定する。"""
    if bundle is None:
        return None

    n_w = _input_transform_n_w_from_model_config(bundle.model_config)
    if n_w is not None:
        return n_w

    sub_bundles = list(bundle.metadata.get("sub_bundles", []) or [])
    if not sub_bundles:
        return None

    output_index = _safe_output_index(output)
    if output_index is not None and 0 <= output_index < len(sub_bundles):
        return _input_transform_n_w_from_model_config(sub_bundles[output_index].model_config)

    inferred_values = [
        value
        for value in (_input_transform_n_w_from_model_config(sub_bundle.model_config) for sub_bundle in sub_bundles)
        if value is not None
    ]
    if inferred_values and len(set(inferred_values)) == 1:
        return inferred_values[0]
    return None


def _resolve_objective_config_n_w_from_input_transform(
    *,
    acq_config: AcquisitionConfig,
    bundle: ModelBundle | None,
) -> AcquisitionConfig:
    """ObjectiveConfig.n_w 未指定なら InputTransformConfig.n_w で補完する。"""
    objective_config = acq_config.objective_config
    if objective_config is None or objective_config.n_w is not None:
        return acq_config
    if "n_w" in objective_config.objective_kwargs:
        return acq_config

    inferred_n_w = _input_transform_n_w_from_bundle(bundle, output=objective_config.output)
    if inferred_n_w is None:
        return acq_config

    return replace(
        acq_config,
        objective_config=replace(objective_config, n_w=inferred_n_w),
    )
