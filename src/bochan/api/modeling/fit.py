"""Model fitting for the public API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..configs import FitConfig, ModelBundle, MultiOutputConfig
from ..support.callables import _filter_kwargs_for_callable


def _resolve_output_fit_configs(
    parent_fit_config: FitConfig | None,
    mo_config: MultiOutputConfig,
    n_outputs: int,
    embedded: Sequence[FitConfig | None] | None = None,
) -> list[FitConfig | None]:
    embedded = list(embedded or [None for _ in range(n_outputs)])
    output_fit_configs = mo_config.output_fit_configs
    if output_fit_configs is None:
        return [embedded[i] or parent_fit_config for i in range(n_outputs)]
    if isinstance(output_fit_configs, FitConfig):
        return [embedded[i] or output_fit_configs for i in range(n_outputs)]
    if len(output_fit_configs) != n_outputs:
        raise ValueError(f"Expected {n_outputs} output_fit_configs, got {len(output_fit_configs)}.")
    return [embedded[i] or output_fit_configs[i] for i in range(n_outputs)]


def _get_num_data(model: Any) -> int:
    if hasattr(model, "model") and hasattr(model.model, "train_inputs"):
        return int(model.model.train_inputs[0].shape[-2])
    if hasattr(model, "train_inputs"):
        train_inputs = model.train_inputs
        return int(train_inputs[0].shape[-2] if isinstance(train_inputs, tuple) else train_inputs.shape[-2])
    if hasattr(model, "train_X"):
        return int(model.train_X.shape[-2])
    raise AttributeError(f"Could not infer num_data from {model.__class__.__name__}.")


def _make_default_mll(bundle: ModelBundle, config: FitConfig) -> Any | None:
    model = bundle.model
    task_type = str(bundle.task_type)

    if config.use_model_make_mll and hasattr(model, "make_mll"):
        return model.make_mll(**config.mll_kwargs)
    if config.mll_factory is not None:
        return config.mll_factory(model, **config.mll_kwargs)
    if config.mll_cls is not None:
        if hasattr(model, "likelihood"):
            return config.mll_cls(model.likelihood, model, **config.mll_kwargs)
        return config.mll_cls(model, **config.mll_kwargs)

    if task_type == "ordinal":
        from bochan.fit import make_ordinal_mll

        return make_ordinal_mll(model, **config.mll_kwargs)

    if task_type in {"binary", "multiclass"}:
        from gpytorch.mlls import VariationalELBO

        likelihood = getattr(model, "likelihood", None)
        mll_model = getattr(model, "model", model)
        return VariationalELBO(likelihood=likelihood, model=mll_model, num_data=_get_num_data(model), **config.mll_kwargs)

    if hasattr(model, "likelihood"):
        from gpytorch.mlls import ExactMarginalLogLikelihood

        return ExactMarginalLogLikelihood(model.likelihood, model, **config.mll_kwargs)

    return None


def _fit_kwargs_from_config(config: FitConfig, *, exact: bool = False) -> dict[str, Any]:
    kwargs = dict(config.fit_kwargs)

    if exact:
        optimizer_kwargs = dict(config.optimizer_kwargs)
        if config.maxiter is not None:
            options = dict(optimizer_kwargs.get("options", {}))
            options.setdefault("maxiter", int(config.maxiter))
            optimizer_kwargs["options"] = options
        if optimizer_kwargs:
            kwargs.setdefault("optimizer_kwargs", optimizer_kwargs)
        return kwargs

    if config.num_epochs is not None:
        kwargs.setdefault("num_epochs", int(config.num_epochs))
    if config.lr is not None:
        kwargs.setdefault("lr", float(config.lr))
    if config.batch_size is not None:
        kwargs.setdefault("batch_size", int(config.batch_size))
    kwargs.setdefault("shuffle", bool(config.shuffle))
    kwargs.setdefault("verbose", bool(config.verbose))
    if config.clip_grad_norm is not None:
        kwargs.setdefault("clip_grad_norm", float(config.clip_grad_norm))
    if config.optimizer_kwargs:
        kwargs.setdefault("optimizer_kwargs", dict(config.optimizer_kwargs))
    return kwargs


def _resolve_fit_func(bundle: ModelBundle, config: FitConfig, mll: Any | None) -> tuple[Callable[..., Any], bool]:
    if config.fit_func is not None:
        return config.fit_func, False

    task_type = str(bundle.task_type)
    model_type = str(bundle.model_type).lower()

    if task_type == "binary":
        if model_type == "rrp":
            from bochan.fit import fit_rrp_binary_classifier_mll

            return fit_rrp_binary_classifier_mll, False
        from bochan.fit import fit_binary_classifier_mll

        return fit_binary_classifier_mll, False

    if task_type == "ordinal":
        if model_type == "rrp":
            from bochan.fit import fit_rrp_ordinal_mll

            return fit_rrp_ordinal_mll, False
        from bochan.fit import fit_ordinal_mll

        return fit_ordinal_mll, False

    if task_type == "multiclass":
        from bochan.fit import fit_multiclass_mll

        return fit_multiclass_mll, False

    if "deepgp" in model_type:
        from bochan.fit import fit_deepgp_mll

        return fit_deepgp_mll, False

    if "deepkernel" in model_type:
        from bochan.fit import fit_deepkernel_mll

        return fit_deepkernel_mll, False

    if model_type.startswith(("beta_", "gamma_", "poisson_", "negative_binomial_")):
        from bochan.fit import fit_non_gaussian_mll

        return fit_non_gaussian_mll, False

    if mll is not None:
        from botorch.fit import fit_gpytorch_mll

        return fit_gpytorch_mll, True

    model = bundle.model
    if hasattr(model, "fit") and callable(model.fit):
        return model.fit, False

    raise ValueError(
        "No fit function could be inferred. "
        "Set FitConfig.fit_func explicitly or FitConfig.skip_fit=True."
    )


def _fit_single_bundle(bundle: ModelBundle, config: FitConfig | None = None) -> ModelBundle:
    config = config or FitConfig()
    if config.skip_fit:
        bundle.fit_config = config
        return bundle

    mll = _make_default_mll(bundle, config)
    fit_func, exact = _resolve_fit_func(bundle, config, mll)
    fit_target = mll if mll is not None else bundle.model
    fit_kwargs = _fit_kwargs_from_config(config, exact=exact)
    fit_kwargs = _filter_kwargs_for_callable(fit_func, fit_kwargs)
    fit_result = fit_func(fit_target, **fit_kwargs)

    bundle.fit_config = config
    bundle.mll = mll
    bundle.fit_result = fit_result
    bundle.metadata.update({"mll": None if mll is None else mll.__class__.__name__, "fit_func": getattr(fit_func, "__name__", str(fit_func))})
    return bundle


def fit_model(bundle: ModelBundle, config: FitConfig | None = None) -> ModelBundle:
    mo_config = bundle.model_config.multi_output_config
    if mo_config is not None:
        sub_bundles = bundle.metadata.get("sub_bundles", [])
        embedded_fit_configs = bundle.metadata.get("embedded_fit_configs", [])
        output_fit_configs = _resolve_output_fit_configs(config or FitConfig(), mo_config, len(sub_bundles), embedded=embedded_fit_configs)
        if mo_config.fit_submodels:
            for sub_bundle, sub_fit_config in zip(sub_bundles, output_fit_configs, strict=True):
                _fit_single_bundle(sub_bundle, sub_fit_config)
        bundle.fit_config = config
        bundle.metadata["sub_fit_configs"] = output_fit_configs
        if mo_config.fit_wrapper:
            return _fit_single_bundle(bundle, config)
        return bundle
    return _fit_single_bundle(bundle, config)


