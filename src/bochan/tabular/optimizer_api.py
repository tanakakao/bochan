'''Public tabular optimizer API aligned with :mod:`bochan.api` config fields.'''

from __future__ import annotations

from typing import Any

from bochan.api import AcquisitionConfig, FitConfig, ModelConfig, OptimizeConfig

from .builders import UNSET, make_acquisition_config, make_fit_config, make_optimize_config
from .optimizer import TabularBayesianOptimizer as _BaseTabularBayesianOptimizer


class TabularBayesianOptimizer(_BaseTabularBayesianOptimizer):
    '''Pandas / numpy friendly optimizer with public API convenience fields.

    This subclass preserves the existing tabular implementation while exposing
    recently added high-level API fields through direct keyword arguments.
    Existing ``model_config``, ``fit_config``, ``acq_config``, and ``opt_config``
    objects remain fully supported.
    '''

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
        *,
        fit_beta: float | None | Any = UNSET,
        beta: float | None | Any = UNSET,
        **kwargs: Any,
    ) -> None:
        if fit_beta is not UNSET and beta is not UNSET:
            raise ValueError("Specify either fit_beta or beta, not both.")
        if beta is not UNSET:
            fit_beta = beta
        if fit_beta is not UNSET:
            fit_config = make_fit_config(fit_config, fit_beta=fit_beta)
        super().__init__(model_config=model_config, fit_config=fit_config, **kwargs)

    def fit(
        self,
        data: Any | None = None,
        y: Any | None = None,
        *,
        fit_config: FitConfig | None = None,
        fit_beta: float | None | Any = UNSET,
        beta: float | None | Any = UNSET,
        **kwargs: Any,
    ) -> "TabularBayesianOptimizer":
        if fit_beta is not UNSET and beta is not UNSET:
            raise ValueError("Specify either fit_beta or beta, not both.")
        if beta is not UNSET:
            fit_beta = beta
        if fit_beta is not UNSET:
            fit_config = make_fit_config(fit_config or self.fit_config, fit_beta=fit_beta)
        return super().fit(data=data, y=y, fit_config=fit_config, **kwargs)

    def candidate(
        self,
        acq_config: AcquisitionConfig | None = None,
        opt_config: OptimizeConfig | None = None,
        *,
        constraints: Any = UNSET,
        outcome_constraint_config: Any = UNSET,
        objective_eq_targets: Any = UNSET,
        objective_eq_target: Any = UNSET,
        objective_maximize: Any = UNSET,
        objective_aggregate_mean_when_no_risk: Any = UNSET,
        objective_allow_unexpanded: Any = UNSET,
        objective_ordinal_likelihood: Any = UNSET,
        evo_method: Any = UNSET,
        **kwargs: Any,
    ) -> Any:
        acq_values = {
            "constraints": constraints,
            "outcome_constraint_config": outcome_constraint_config,
            "objective_eq_targets": objective_eq_targets,
            "objective_eq_target": objective_eq_target,
            "objective_maximize": objective_maximize,
            "objective_aggregate_mean_when_no_risk": objective_aggregate_mean_when_no_risk,
            "objective_allow_unexpanded": objective_allow_unexpanded,
            "objective_ordinal_likelihood": objective_ordinal_likelihood,
        }
        if any(value is not UNSET for value in acq_values.values()):
            acq_config = make_acquisition_config(acq_config, **acq_values)
        if evo_method is not UNSET:
            opt_config = make_optimize_config(opt_config, evo_method=evo_method)
        return super().candidate(acq_config=acq_config, opt_config=opt_config, **kwargs)


__all__ = ["TabularBayesianOptimizer"]
