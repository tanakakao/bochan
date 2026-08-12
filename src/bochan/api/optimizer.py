"""Canonical public Bayesian optimization API."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .acquisition.defaults import (
    resolve_llm_selected_model_config,
    resolve_multi_output_model_config,
)
from .acquisition.service import (
    build_acquisition,
    is_nsgaii_strategy,
    resolve_acquisition,
    resolve_acquisition_class,
)
from .candidate.output import select_best_candidate_set
from .configs import (
    AcquisitionConfig,
    CandidateResult,
    DataContext,
    FitConfig,
    ModelConfig,
    OptimizeConfig,
)
from .engine import BayesianOptimizer as _CoreBayesianOptimizer
from .engine import (
    _infer_bounds_from_train_X,
    _resolve_mixed_fixed_features_from_train_X,
    _resolve_mixed_optimizer_callable,
)
from .experiment_failure import attach_observation_state
from .factory import fit_model
from .information_acquisition_defaults import resolve_information_optimizer_defaults
from .llm import LLMCandidateExplanationMixin, LLMSuggestionMixin
from .observation import ExperimentFailureConfig, ObservationData
from .observation.service import build_objective_bundle
from .optimizer_api import optimize_candidates, resolve_optimizer_from_cat_dims


class BayesianOptimizer(
    LLMCandidateExplanationMixin,
    LLMSuggestionMixin,
    _CoreBayesianOptimizer,
):
    """High-level optimizer shared by tensor, tabular, and serving APIs."""

    observations: ObservationData | None = None
    failure_config: ExperimentFailureConfig | None = None
    failure_bundle: Any | None = None
    failure_model: Any | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.acq_config: AcquisitionConfig | None = None
        self.opt_config: OptimizeConfig | None = None
        self.last_suggestion: Any | None = None
        self.last_acquisition_suggestion: Any | None = None
        self.last_candidate_explanation: Any | None = None
        self.last_candidate_explanation_prompt: str | None = None
        self._llm_refit_required = False

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
    ) -> BayesianOptimizer:
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
        elif any(
            value is not None
            for value in (observed_mask, failed_mask, pending_mask)
        ):
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
        self.model_config = self._merge_llm_settings_into_model_config(
            resolved_model_config
        )
        self.fit_config = base_fit_config
        self.train_X = objective_X
        self.train_Y = objective_Y

        if self.bounds is None:
            self.bounds = _infer_bounds_from_train_X(observation_data.X)
        if self.data_context is not None:
            self._resolve_data_context(self.data_context)

        self.bundle = build_objective_bundle(
            train_X=objective_X,
            train_Y=objective_Y,
            config=self.model_config,
            model_registry=self.model_registry,
        )
        self.bundle = fit_model(self.bundle, self.fit_config)
        self.model = self.bundle.model
        self.mll = self.bundle.mll

        attach_observation_state(
            self,
            observation_data,
            failure_config=failure_config,
        )

        if llm_plan is not None:
            self.llm_plan = llm_plan
            self.bundle.metadata["llm_plan"] = llm_plan
            self.bundle.metadata["llm_selected_model_config"] = resolved_model_config

        self._llm_refit_required = False
        return self

    def fit_observations(
        self,
        observations: ObservationData,
        *,
        failure_config: ExperimentFailureConfig | None = None,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> BayesianOptimizer:
        return self.fit(
            observation_data=observations,
            failure_config=failure_config,
            model_config=model_config,
            fit_config=fit_config,
        )

    def refit(self, *, fit_config: FitConfig | None = None) -> BayesianOptimizer:
        if self.observations is None:
            return super().refit(fit_config=fit_config)
        return self.fit(
            observation_data=self.observations,
            failure_config=self.failure_config,
            model_config=self.model_config,
            fit_config=fit_config or self.fit_config,
        )

    def tell_observations(
        self,
        observations: ObservationData,
        *,
        refit: bool = True,
        fit_config: FitConfig | None = None,
    ) -> BayesianOptimizer:
        if self.observations is None:
            raise RuntimeError("Call fit(...) before tell_observations(...).")
        self.observations = self.observations.append(observations)
        self.train_X, self.train_Y = self.observations.objective_training_data()
        if refit:
            self.refit(fit_config=fit_config)
        return self

    def tell(
        self,
        X_new: Any,
        Y_new: Any,
        *,
        status: Any = "success",
        observed_mask: Any | None = None,
        refit: bool = True,
        fit_config: FitConfig | None = None,
    ) -> BayesianOptimizer:
        import torch

        X_tensor = torch.as_tensor(X_new)
        n_rows = int(X_tensor.shape[0]) if X_tensor.ndim > 1 else 1
        statuses = [status] * n_rows if isinstance(status, str) else list(status)
        observations = ObservationData.from_status(
            X_new,
            Y_new,
            status=statuses,
            observed_mask=observed_mask,
        )
        return self.tell_observations(
            observations,
            refit=refit,
            fit_config=fit_config,
        )

    def update_data(
        self,
        X_new: Any,
        Y_new: Any,
        *,
        append: bool = True,
    ) -> BayesianOptimizer:
        if not append:
            self.fit(
                X_new,
                Y_new,
                model_config=self.model_config,
                fit_config=self.fit_config,
                failure_config=self.failure_config,
            )
            return self
        if self.observations is None:
            return super().update_data(X_new, Y_new)
        return self.tell(X_new, Y_new, status="success", refit=False)

    def _resolve_data_context(
        self,
        data_context: DataContext | None = None,
    ) -> DataContext:
        context = super()._resolve_data_context(data_context)
        if (
            context.X_pending is None
            and self.observations is not None
            and bool(self.observations.pending_mask.any())
        ):
            context.X_pending = self.observations.pending_X
        return context

    def _resolve_acquisition_config(
        self,
        acq_config: AcquisitionConfig,
    ) -> AcquisitionConfig:
        return resolve_acquisition_class(self, acq_config)

    def _configured_acquisition(
        self,
        config: AcquisitionConfig | None,
    ) -> AcquisitionConfig:
        resolved = config if config is not None else self.acq_config
        if resolved is None:
            raise ValueError(
                "acq_config is required. Pass it explicitly or configure a default first."
            )
        if self._llm_refit_required:
            raise RuntimeError(
                "Model or fit settings changed after fitting. Call fit() or refit() "
                "before building an acquisition."
            )
        return resolved

    def _configured_optimizer(
        self,
        config: OptimizeConfig | None,
    ) -> OptimizeConfig:
        resolved = config if config is not None else self.opt_config
        if resolved is None:
            raise ValueError(
                "opt_config is required. Pass it explicitly or configure a default first."
            )
        return resolved

    def _prepare_acquisition(
        self,
        acq_config: AcquisitionConfig | None,
        data_context: DataContext | None,
    ) -> tuple[AcquisitionConfig, DataContext, Any]:
        self._check_fitted()
        configured = self._configured_acquisition(acq_config)
        context = self._resolve_data_context(data_context)
        context = replace(context, extra=dict(context.extra))
        resolved, context = resolve_acquisition(self, configured, context)
        acqf = build_acquisition(
            bundle=self.bundle,
            config=resolved,
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
        return resolved, context, acqf

    def acquisition(
        self,
        acq_config: AcquisitionConfig | None = None,
        *,
        data_context: DataContext | None = None,
    ) -> Any:
        _, _, acqf = self._prepare_acquisition(acq_config, data_context)
        return acqf

    def candidate(
        self,
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> CandidateResult | tuple[Any, Any]:
        resolved_config, context, acqf = self._prepare_acquisition(
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
        if opt_bounds is None and self.train_X is not None:
            opt_bounds = _infer_bounds_from_train_X(self.train_X)
            self.bounds = opt_bounds
            context.bounds = opt_bounds

        resolved_opt_config = self._configured_optimizer(opt_config)
        resolved_opt_config = resolve_information_optimizer_defaults(
            resolved_config,
            resolved_opt_config,
        )
        if is_nsgaii_strategy(resolved_config):
            resolved_opt_config = replace(resolved_opt_config, optimizer="nsgaii")
        resolved_opt_config = self._merge_llm_settings_into_opt_config(
            resolved_opt_config
        )

        cat_dims = self.bundle.cat_dims if self.bundle is not None else []
        resolved_opt_config = resolve_optimizer_from_cat_dims(
            opt_config=resolved_opt_config,
            cat_dims=cat_dims,
        )
        resolved_opt_config = _resolve_mixed_fixed_features_from_train_X(
            opt_config=resolved_opt_config,
            train_X=self.train_X,
            cat_dims=cat_dims,
        )
        resolved_opt_config = _resolve_mixed_optimizer_callable(
            resolved_opt_config
        )

        candidates, acq_value = optimize_candidates(
            acqf=acqf,
            bounds=opt_bounds,
            config=resolved_opt_config,
        )
        candidates, acq_value = select_best_candidate_set(
            candidates,
            acq_value,
            q=resolved_opt_config.q,
            return_best_only=resolved_opt_config.return_best_only,
            acqf=acqf,
        )
        result = CandidateResult(
            candidates=candidates,
            acq_value=acq_value,
            acqf=acqf,
            acq_config=resolved_config,
            opt_config=resolved_opt_config,
            data_context=context,
        )
        self.history.append(result)
        if return_result:
            return result
        return candidates, acq_value

    def ask(
        self,
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> CandidateResult | tuple[Any, Any]:
        return self.candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            data_context=data_context,
            bounds=bounds,
            return_result=return_result,
        )

    def cross_validate(self, *args: Any, **kwargs: Any) -> Any:
        if self.observations is not None:
            has_partial = not bool(self.observations.observed_mask.all())
            has_failure_state = bool(
                self.observations.failed_mask.any()
                or self.observations.pending_mask.any()
            )
            if has_partial or has_failure_state:
                raise ValueError(
                    "Cross-validation for partial / failed / pending observation states "
                    "is not defined. Use an explicit observation-aware validation protocol."
                )
        return super().cross_validate(*args, **kwargs)


__all__ = ["BayesianOptimizer"]
