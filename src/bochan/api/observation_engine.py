"""Observation-aware high-level Bayesian optimization engine.

This module extends the regular automatic-default engine through normal source
inheritance.  It does not replace functions or class methods at import time.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .configs import CandidateResult, DataContext, FitConfig, ModelBundle, ModelConfig, OptimizeConfig
from .engine import (
    _filter_context_fields_for_acqf,
    _infer_bounds_from_train_X,
    _resolve_mixed_fixed_features_from_train_X,
    _resolve_mixed_optimizer_callable,
    _resolve_objective_config_n_w_from_input_transform,
    _resolve_optimizer_from_cat_dims,
)
from .engine_defaults import (
    BayesianOptimizer as _DefaultBayesianOptimizer,
    resolve_acquisition_defaults,
    resolve_information_optimizer_defaults,
    resolve_llm_selected_model_config,
    resolve_multi_output_model_config,
)
from .factory import (
    _as_cat_dims,
    _build_single_model,
    _build_wrapper_from_submodels,
    _infer_num_outputs,
    _resolve_output_configs,
    fit_model,
    infer_input_type,
)
from .observation import ExperimentFailureConfig, ObservationData


def _normalize_model_name(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _supports_wide_missing_targets(config: ModelConfig) -> bool:
    """Return whether one model consumes a wide matrix with NaN task cells."""

    name = _normalize_model_name(config.model_type)
    return name == "multitask" or name.endswith("widemultitask")


def _reject_unsupported_correlated_missing(config: ModelConfig, train_Y: Any) -> None:
    import torch

    if not bool(torch.isnan(torch.as_tensor(train_Y)).any()):
        return
    name = _normalize_model_name(config.model_type)
    if name == "kronecker" or name.endswith("kronecker"):
        raise ValueError(
            "Kronecker multi-task models require a complete rectangular target matrix. "
            "Use model_type='multitask' / a '*_wide_multitask' model for partially "
            "observed objectives. Missing targets are not imputed automatically."
        )
    if name == "multifidelity" or name.endswith("multifidelity"):
        raise ValueError(
            "The current multi-fidelity model requires complete target rows. "
            "Partially observed objective cells are not imputed automatically."
        )


def _partial_hybrid_wrapper(
    model: Any,
    *,
    train_X: Any,
    train_Y: Any,
    observed_mask: Any,
) -> Any:
    """Retain the original wide observation table for a Hybrid wrapper."""

    from bochan.models.hybrid.multi_output import HybridMultiOutputModel

    if not isinstance(model, HybridMultiOutputModel):
        return model
    from bochan.models.hybrid.partial_observation import (
        PartiallyObservedHybridMultiOutputModel,
    )

    return PartiallyObservedHybridMultiOutputModel(
        specs=list(model.specs),
        train_X_wide=train_X,
        train_Y_wide=train_Y,
        observed_mask_wide=observed_mask,
    )


def _build_split_partial_bundle(
    *,
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    model_registry: Any = None,
) -> ModelBundle:
    """Build split-output models from only the rows observed for each output."""

    import torch

    mo_config = config.multi_output_config
    if mo_config is None:
        raise RuntimeError("multi_output_config is required for split partial observations.")
    n_outputs = _infer_num_outputs(train_Y)
    output_configs, output_names, inline_spec_kwargs, embedded_fit_configs = (
        _resolve_output_configs(config, n_outputs)
    )
    observed_mask = torch.isfinite(torch.as_tensor(train_Y))
    sub_bundles: list[ModelBundle] = []
    observed_counts: list[int] = []

    for index, output_config in enumerate(output_configs):
        mask = observed_mask[:, index]
        count = int(mask.sum().item())
        if count == 0:
            name = output_names[index] or f"output_{index}"
            raise ValueError(f"{name}: at least one observed target value is required.")
        observed_counts.append(count)
        output_X = train_X[mask]
        output_Y = train_Y[mask, index : index + 1]
        sub_bundles.append(
            _build_single_model(
                train_X=output_X,
                train_Y=output_Y,
                config=output_config,
                model_registry=model_registry,
            )
        )

    model = _build_wrapper_from_submodels(
        [bundle.model for bundle in sub_bundles],
        output_configs,
        mo_config,
        output_names=output_names,
        output_spec_kwargs=inline_spec_kwargs,
    )
    model = _partial_hybrid_wrapper(
        model,
        train_X=train_X,
        train_Y=train_Y,
        observed_mask=observed_mask,
    )

    return ModelBundle(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        input_type=config.input_type or infer_input_type(_as_cat_dims(config.cat_dims)),
        task_type=str(config.task_type),
        model_type=str(config.model_type),
        cat_dims=_as_cat_dims(config.cat_dims),
        metadata={
            "model_cls": model.__class__.__name__,
            "multi_output": True,
            "partial_observation": True,
            "observed_per_output": observed_counts,
            "sub_bundles": sub_bundles,
            "output_configs": output_configs,
            "embedded_fit_configs": embedded_fit_configs,
        },
    )


def _build_partial_objective_bundle(
    *,
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    model_registry: Any = None,
) -> ModelBundle:
    """Build an objective model without replacing missing targets by predictions."""

    import torch

    Y = torch.as_tensor(train_Y)
    has_missing = bool(torch.isnan(Y).any())
    if not has_missing:
        from .factory import build_model

        return build_model(
            train_X=train_X,
            train_Y=train_Y,
            config=config,
            model_registry=model_registry,
        )

    _reject_unsupported_correlated_missing(config, Y)
    if _supports_wide_missing_targets(config):
        from .factory import build_model

        return build_model(
            train_X=train_X,
            train_Y=train_Y,
            config=config,
            model_registry=model_registry,
        )

    if config.multi_output_config is not None:
        return _build_split_partial_bundle(
            train_X=train_X,
            train_Y=Y,
            config=config,
            model_registry=model_registry,
        )

    if int(Y.shape[-1]) != 1:
        raise ValueError(
            "Partially observed multi-output data requires model_type='multitask' "
            "or a MultiOutputConfig so each output can be fitted from its observed rows."
        )
    finite = torch.isfinite(Y[:, 0])
    if not bool(finite.any()):
        raise ValueError("The objective has no observed values.")
    return _build_single_model(
        train_X=train_X[finite],
        train_Y=Y[finite],
        config=config,
        model_registry=model_registry,
    )


def _default_failure_model_config(objective_config: ModelConfig) -> ModelConfig:
    """Create a binary GP config sharing only meaningful input-side settings."""

    return ModelConfig(
        task_type="binary",
        model_type="base",
        input_type=objective_config.input_type,
        cat_dims=objective_config.cat_dims,
        input_transform_config=objective_config.input_transform_config,
        outcome_transform=False,
    )


class BayesianOptimizer(_DefaultBayesianOptimizer):
    """Bayesian optimizer with explicit observation and experiment-failure states."""

    observations: ObservationData | None = None
    failure_config: ExperimentFailureConfig | None = None
    failure_bundle: ModelBundle | None = None
    failure_model: Any | None = None

    def fit(
        self,
        train_X: Any | None = None,
        train_Y: Any | None = None,
        *,
        observation_data: ObservationData | None = None,
        observed_mask: Any | None = None,
        failed_mask: Any | None = None,
        pending_mask: Any | None = None,
        failure_config: ExperimentFailureConfig | None = None,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> "BayesianOptimizer":
        """Fit objective models and, when requested, an independent success model."""

        if observation_data is None:
            if train_X is None or train_Y is None:
                raise ValueError("Provide observation_data or both train_X and train_Y.")
            observation_data = ObservationData(
                X=train_X,
                Y=train_Y,
                observed_mask=observed_mask,
                failed_mask=failed_mask,
                pending_mask=pending_mask,
            )
        elif any(value is not None for value in (observed_mask, failed_mask, pending_mask)):
            raise ValueError(
                "Pass masks inside ObservationData when observation_data is supplied."
            )

        objective_X, objective_Y = observation_data.objective_training_data()
        base_model_config = model_config or self.model_config
        base_fit_config = fit_config or self.fit_config
        base_model_config, base_fit_config, llm_plan = resolve_llm_selected_model_config(
            base_model_config,
            objective_X,
            objective_Y,
            bounds=self.bounds,
            fit_config=base_fit_config,
        )
        resolved_model_config = resolve_multi_output_model_config(
            base_model_config,
            objective_Y,
        )

        self.observations = observation_data
        self.failure_config = failure_config
        self.model_config = self._merge_llm_settings_into_model_config(resolved_model_config)
        self.fit_config = base_fit_config
        self.train_X = objective_X
        self.train_Y = objective_Y
        if self.bounds is None:
            self.bounds = _infer_bounds_from_train_X(observation_data.X)
        if self.data_context is not None:
            self._resolve_data_context(self.data_context)

        self.bundle = _build_partial_objective_bundle(
            train_X=objective_X,
            train_Y=objective_Y,
            config=self.model_config,
            model_registry=self.model_registry,
        )
        self.bundle = fit_model(self.bundle, self.fit_config)
        self.model = self.bundle.model
        self.mll = self.bundle.mll
        self.bundle.metadata["observation"] = observation_data.report()
        self.bundle.metadata["observed_target_mask"] = observation_data.observed_mask

        if llm_plan is not None:
            self.llm_plan = llm_plan
            self.bundle.metadata["llm_plan"] = llm_plan
            self.bundle.metadata["llm_selected_model_config"] = resolved_model_config

        self.failure_bundle = None
        self.failure_model = None
        if failure_config is not None and bool(observation_data.failed_mask.any()):
            from .factory import build_model

            success_X, success_Y = observation_data.success_training_data()
            failure_model_config = failure_config.model_config or _default_failure_model_config(
                self.model_config
            )
            failure_fit_config = failure_config.fit_config or self.fit_config
            self.failure_bundle = build_model(
                train_X=success_X,
                train_Y=success_Y,
                config=failure_model_config,
                model_registry=self.model_registry,
            )
            self.failure_bundle = fit_model(self.failure_bundle, failure_fit_config)
            self.failure_model = self.failure_bundle.model
            self.bundle.metadata["experiment_failure_model"] = {
                "enabled": True,
                "model_type": str(failure_model_config.model_type),
                "n_completed": int(success_X.shape[0]),
                "n_failed": int(observation_data.failed_mask.sum().item()),
            }
        else:
            self.bundle.metadata["experiment_failure_model"] = {
                "enabled": False,
                "reason": "not_configured"
                if failure_config is None
                else "no_failed_experiments",
            }
        return self

    def fit_observations(
        self,
        observations: ObservationData,
        *,
        failure_config: ExperimentFailureConfig | None = None,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> "BayesianOptimizer":
        """Explicit observation-state entry point."""

        return self.fit(
            observation_data=observations,
            failure_config=failure_config,
            model_config=model_config,
            fit_config=fit_config,
        )

    def refit(self, *, fit_config: FitConfig | None = None) -> "BayesianOptimizer":
        """Refit from the preserved canonical observation table."""

        if self.observations is None:
            return super().refit(fit_config=fit_config)
        return self.fit(
            observation_data=self.observations,
            failure_config=self.failure_config,
            model_config=self.model_config,
            fit_config=fit_config or self.fit_config,
        )

    def _resolve_data_context(self, data_context: DataContext | None = None) -> DataContext:
        context = super()._resolve_data_context(data_context)
        if (
            context.X_pending is None
            and self.observations is not None
            and bool(self.observations.pending_mask.any())
        ):
            context.X_pending = self.observations.pending_X
        return context

    def _prepare_observation_acquisition(
        self,
        acq_config: Any,
        data_context: DataContext | None,
    ) -> tuple[Any, DataContext, Any]:
        """Build the regular acquisition, then compose experiment success once."""

        self._check_fitted()
        base_context = self._resolve_data_context(data_context)
        context = replace(base_context, extra=dict(base_context.extra))
        resolved_config = self._resolve_acquisition_config(acq_config)
        resolved_config = _resolve_objective_config_n_w_from_input_transform(
            acq_config=resolved_config,
            bundle=self.bundle,
        )
        resolved_config, context = resolve_acquisition_defaults(
            self.bundle,
            resolved_config,
            context,
        )
        resolved_config = _filter_context_fields_for_acqf(resolved_config)

        from .factory import build_acquisition

        acqf = build_acquisition(
            bundle=self.bundle,
            config=resolved_config,
            data_context=context,
        )
        if self.failure_model is not None and self.failure_config is not None:
            from bochan.acquisition.feasible import ExperimentSuccessWeightedAcquisition

            acqf = ExperimentSuccessWeightedAcquisition(
                acqf=acqf,
                success_model=self.failure_model,
                min_success_probability=self.failure_config.min_success_probability,
                eta=self.failure_config.eta,
                reduce_q=self.failure_config.reduce_q,
            )
        return resolved_config, context, acqf

    def acquisition(
        self,
        acq_config: Any,
        *,
        data_context: DataContext | None = None,
    ) -> Any:
        """Build an observation-aware acquisition through normal composition."""

        _, _, acqf = self._prepare_observation_acquisition(acq_config, data_context)
        return acqf

    def candidate(
        self,
        acq_config: Any,
        opt_config: OptimizeConfig,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> CandidateResult | tuple[Any, Any]:
        """Generate candidates while accounting for failure and pending trials."""

        resolved_config, context, acqf = self._prepare_observation_acquisition(
            acq_config,
            data_context,
        )
        opt_bounds = bounds if bounds is not None else context.bounds
        if opt_bounds is None:
            opt_bounds = self.bounds
        if opt_bounds is None and self.observations is not None:
            opt_bounds = _infer_bounds_from_train_X(self.observations.X)
            self.bounds = opt_bounds
            context.bounds = opt_bounds

        opt_config = resolve_information_optimizer_defaults(resolved_config, opt_config)
        opt_config = self._merge_llm_settings_into_opt_config(opt_config)
        cat_dims = self.bundle.cat_dims if self.bundle is not None else []
        opt_config = _resolve_optimizer_from_cat_dims(
            opt_config=opt_config,
            cat_dims=cat_dims,
        )
        opt_config = _resolve_mixed_fixed_features_from_train_X(
            opt_config=opt_config,
            train_X=self.train_X,
            cat_dims=cat_dims,
        )
        opt_config = _resolve_mixed_optimizer_callable(opt_config)

        from .factory import optimize_candidates

        candidates, acq_value = optimize_candidates(
            acqf=acqf,
            bounds=opt_bounds,
            config=opt_config,
        )
        result = CandidateResult(
            candidates=candidates,
            acq_value=acq_value,
            acqf=acqf,
            acq_config=resolved_config,
            opt_config=opt_config,
            data_context=context,
        )
        self.history.append(result)
        if return_result:
            return result
        return candidates, acq_value

    def cross_validate(self, *args: Any, **kwargs: Any) -> Any:
        """Reject ambiguous fold semantics for partial/failure observations."""

        if self.observations is not None:
            has_partial = not bool(self.observations.observed_mask.all())
            has_failure_state = bool(
                self.observations.failed_mask.any() or self.observations.pending_mask.any()
            )
            if has_partial or has_failure_state:
                raise ValueError(
                    "Cross-validation for partial / failed / pending observation states "
                    "is not defined in this release. Evaluate each output and the success "
                    "classifier with an explicit observation-aware validation protocol."
                )
        return super().cross_validate(*args, **kwargs)


__all__ = ["BayesianOptimizer"]
