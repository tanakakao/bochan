"""Capability-based, read-only model diagnostic extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

_MAX_DIAGNOSTIC_POINTS = 500


def _class_path(value: Any) -> str:
    """Return a stable fully qualified class name for diagnostic metadata."""

    return f"{type(value).__module__}.{type(value).__name__}"


def _module_summary(module: torch.nn.Module) -> dict[str, Any]:
    """Describe a torch module without returning the live Python object."""

    parameter_count = 0
    trainable_parameter_count = 0
    try:
        for parameter in module.parameters():
            count = int(parameter.numel())
            parameter_count += count
            if parameter.requires_grad:
                trainable_parameter_count += count
    except (AttributeError, RuntimeError, TypeError, ValueError):
        parameter_count = 0
        trainable_parameter_count = 0

    children: list[dict[str, str]] = []
    try:
        children = [
            {"name": str(name), "class": _class_path(child)}
            for name, child in module.named_children()
        ]
    except (AttributeError, RuntimeError, TypeError, ValueError):
        children = []

    return {
        "kind": "module",
        "class": _class_path(module),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "children": children,
    }


def _callable_summary(value: Any) -> dict[str, Any]:
    """Describe an available callable without returning a bound method."""

    return {
        "kind": "callable",
        "name": str(getattr(value, "__name__", type(value).__name__)),
        "available": True,
    }


def _value(value: Any) -> Any:
    """Convert a diagnostic attribute to a compact JSON-safe value."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, torch.nn.Module):
        return _module_summary(value)
    if callable(value):
        return _callable_summary(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_value(item) for item in value]

    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _value(to_list())
        except (RuntimeError, TypeError, ValueError):
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _value(item())
        except (RuntimeError, TypeError, ValueError):
            pass
    return {"kind": "object", "class": _class_path(value)}


def _first_tensor(value: Any) -> torch.Tensor | None:
    """Return the first tensor from a tensor or train-input container."""

    if torch.is_tensor(value):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if torch.is_tensor(item):
                return item
    return None


def _training_inputs(model: Any) -> torch.Tensor | None:
    """Locate raw-space training inputs used by a heteroscedastic model."""

    candidates = [
        model,
        getattr(model, "noise_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("train_inputs_raw", "train_inputs"):
            tensor = _first_tensor(getattr(candidate, attribute, None))
            if tensor is not None:
                return tensor
    return None


def _sample_training_rows(
    X: torch.Tensor,
    *,
    max_points: int = _MAX_DIAGNOSTIC_POINTS,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Select deterministic, order-preserving rows for diagnostic plots."""

    if X.ndim < 2 or X.shape[-1] == 0:
        raise ValueError("Training inputs must have shape [..., n, d] with d > 0.")
    flat = X.detach().reshape(-1, X.shape[-1])
    total = int(flat.shape[0])
    if total <= max_points:
        indices = torch.arange(total, device=flat.device)
    else:
        indices = torch.linspace(
            0,
            total - 1,
            steps=max_points,
            device=flat.device,
        ).round().to(dtype=torch.long).unique(sorted=True)
    return flat.index_select(0, indices), indices, total


def _heteroscedastic_noise_profile(
    model: Any,
    *,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate input-dependent noise on representative raw training rows."""

    predictor = getattr(model, "predict_noise_var", None)
    X = _training_inputs(model)
    if not callable(predictor) or X is None:
        return {}

    try:
        X_display, indices, total = _sample_training_rows(X)
        with torch.no_grad():
            variance = torch.as_tensor(predictor(X_display)).detach()
        displayed = int(X_display.shape[0])
        if displayed == 0 or variance.numel() % displayed != 0:
            return {}
        variance = variance.reshape(displayed, -1).mean(dim=-1).clamp_min(0)
        std = variance.sqrt()
        log_variance = variance.clamp_min(1e-30).log()
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return {}

    dimension = int(X_display.shape[-1])
    names = [
        feature_names[index] if index < len(feature_names) else f"feature_{index}"
        for index in range(dimension)
    ]
    return {
        "sample_index": (indices.detach().cpu() + 1).tolist(),
        "feature_names": names,
        "feature_values": X_display.detach().cpu().tolist(),
        "noise_variance": variance.cpu().tolist(),
        "noise_std": std.cpu().tolist(),
        "noise_log_variance": log_variance.cpu().tolist(),
        "total_count": total,
        "displayed_count": displayed,
        "sampling": "all" if displayed == total else "evenly_spaced",
        "source_space": "raw",
        "interpretation": (
            "Predicted observation-noise variance and standard deviation on raw "
            "training inputs; larger values indicate noisier input regions."
        ),
    }


def _kernel_components(model: Any) -> list[dict[str, Any]]:
    """Walk the kernel module tree and describe available parameters."""
    root = getattr(model, "covar_module", None)
    if root is None:
        return []
    components = []
    for suffix, module in root.named_modules():
        path = "covar_module" + (f".{suffix}" if suffix else "")
        item: dict[str, Any] = {"path": path, "kernel_class": type(module).__name__}
        for name in ("active_dims", "lengthscale", "outputscale", "batch_shape"):
            try:
                value = getattr(module, name)
            except (AttributeError, RuntimeError):
                continue
            item[name] = _value(value)
        if "lengthscale" in item:
            lengthscale = torch.as_tensor(item["lengthscale"])
            item["inverse_lengthscale"] = torch.reciprocal(lengthscale).tolist()
        if len(item) > 2:
            item.update(source_space="raw", is_predictive_importance=False)
            components.append(item)
    return components


def _output_models(model: Any) -> list[tuple[str, Any]]:
    """Return named submodels from a hybrid multi-output wrapper when available."""
    models = getattr(model, "models", None)
    if models is None or isinstance(models, (str, bytes)):
        return []
    try:
        children = list(models)
    except TypeError:
        return []
    if not children:
        return []
    raw_names = getattr(model, "output_names", None)
    names = list(raw_names) if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)) else []
    return [
        (str(names[index]) if index < len(names) else f"output_{index}", child)
        for index, child in enumerate(children)
    ]


def _merge_output_diagnostics(
    values: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Merge per-output diagnostics while preserving the original output names."""
    keys = list(dict.fromkeys(key for _, diagnostics in values for key in diagnostics))
    merged: dict[str, Any] = {}
    for key in keys:
        by_output = {
            output_name: diagnostics[key]
            for output_name, diagnostics in values
            if key in diagnostics
        }
        if len(by_output) == 1:
            merged[key] = next(iter(by_output.values()))
            continue
        if key == "kernel_components":
            components = []
            for output_name, items in by_output.items():
                for item in items if isinstance(items, list) else []:
                    component = dict(item)
                    component["output_name"] = output_name
                    components.append(component)
            merged[key] = components
            continue
        if key == "ard":
            components = []
            template: dict[str, Any] = {}
            for output_name, diagnostic in by_output.items():
                record = diagnostic if isinstance(diagnostic, dict) else {}
                if not template:
                    template = {
                        name: item
                        for name, item in record.items()
                        if name != "components"
                    }
                for item in record.get("components", []):
                    component = dict(item)
                    component["output_name"] = output_name
                    components.append(component)
            merged[key] = {
                **template,
                "components": components,
                "by_output": by_output,
            }
            continue
        merged[key] = {
            "by_output": by_output,
            "output_names": list(by_output),
            "is_predictive_importance": False,
        }
    return merged


def _extract_direct_model_diagnostics(
    model: Any,
    *,
    methods: tuple[str, ...],
    feature_names: tuple[str, ...],
    cat_dims: tuple[int, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Extract diagnostics from one concrete fitted model."""
    if not methods:
        return {}, []
    auto = "auto" in methods
    requested = set(methods) - {"auto"}
    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []
    components = _kernel_components(model)
    if components and (auto or requested & {"ard", "kernel_components"}):
        if auto or "kernel_components" in requested:
            diagnostics["kernel_components"] = components
        if auto or "ard" in requested:
            ard = [item for item in components if "lengthscale" in item]
            if ard:
                diagnostics["ard"] = {
                    "components": ard,
                    "source_space": "raw",
                    "is_predictive_importance": False,
                    "interpretation": "Inverse lengthscales are kernel sensitivity diagnostics, not causal or predictive importance.",
                    "warning": "Transforms or projected wrappers may change the kernel source space.",
                }
    capabilities = {
        "pca": ("components_", "explained_variance_ratio_"),
        "rembo": ("projection_matrix", "projection"),
        "vae": ("encoder", "decoder"),
        "deepkernel": ("feature_extractor",),
        "deepgp": ("layers", "hidden_layers"),
        "rrp": ("support", "support_values", "outlier_indices"),
        "multitask": ("task_covar_module", "task_kernel"),
        "multifidelity": ("fidelity_dims", "fidelity_features"),
        "heteroscedastic": ("noise_model", "predict_noise_var", "predict_noise_logvar"),
        "saas": ("tau", "raw_tau"),
    }
    for name, attrs in capabilities.items():
        if not (auto or name in requested):
            continue
        found = {attr: _value(getattr(model, attr)) for attr in attrs if hasattr(model, attr)}
        if name == "heteroscedastic" and found:
            profile = _heteroscedastic_noise_profile(
                model,
                feature_names=feature_names,
            )
            if profile:
                found["noise_profile"] = profile
        if found:
            found.update(
                source_space="latent" if name in {"pca", "rembo", "vae", "deepkernel", "deepgp"} else "raw",
                is_predictive_importance=False,
                interpretation=f"Read-only {name} model structure; it is not permutation importance.",
                raw_feature_names=list(feature_names),
                categorical_dims=list(cat_dims),
            )
            key = "observation_relevance" if name == "rrp" else name
            diagnostics[key] = found
        elif not auto and name in requested:
            warnings.append(f"Diagnostic {name!r} is not supported by this model interface.")
    return diagnostics, warnings


def extract_model_diagnostics(
    model: Any,
    *,
    methods: tuple[str, ...],
    feature_names: tuple[str, ...],
    cat_dims: tuple[int, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Extract lightweight diagnostics through interface capabilities.

    Hybrid Web models expose their concrete fitted models through ``models``.
    A single child is unwrapped transparently. Multiple children are inspected
    independently and merged under their original output-column names.

    Args:
        model: Fitted model inspected without mutation.
        methods: Requested names; ``auto`` enables all supported extractors.
        feature_names: Raw input names used only for provenance.
        cat_dims: Raw categorical dimensions.

    Returns:
        Diagnostic mapping and non-fatal warnings.
    """
    output_models = _output_models(model)
    if len(output_models) == 1:
        return _extract_direct_model_diagnostics(
            output_models[0][1],
            methods=methods,
            feature_names=feature_names,
            cat_dims=cat_dims,
        )
    if len(output_models) > 1:
        extracted: list[tuple[str, dict[str, Any]]] = []
        warnings: list[str] = []
        for output_name, child in output_models:
            diagnostics, child_warnings = _extract_direct_model_diagnostics(
                child,
                methods=methods,
                feature_names=feature_names,
                cat_dims=cat_dims,
            )
            if diagnostics:
                extracted.append((output_name, diagnostics))
            warnings.extend(f"{output_name}: {warning}" for warning in child_warnings)
        return _merge_output_diagnostics(extracted), warnings
    return _extract_direct_model_diagnostics(
        model,
        methods=methods,
        feature_names=feature_names,
        cat_dims=cat_dims,
    )
