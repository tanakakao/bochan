"""Capability-based, read-only model diagnostic extraction."""

from __future__ import annotations

from typing import Any

import torch


def _value(value: Any) -> Any:
    """Detach a parameter without changing its device or dtype."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


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


def extract_model_diagnostics(
    model: Any,
    *,
    methods: tuple[str, ...],
    feature_names: tuple[str, ...],
    cat_dims: tuple[int, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Extract lightweight diagnostics through interface capabilities.

    Args:
        model: Fitted model inspected without mutation.
        methods: Requested names; ``auto`` enables all supported extractors.
        feature_names: Raw input names used only for provenance.
        cat_dims: Raw categorical dimensions.

    Returns:
        Diagnostic mapping and non-fatal warnings.
    """
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
