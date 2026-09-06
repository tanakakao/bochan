"""Cost-observation extension for the public Bayesian optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from .configs import AcquisitionConfig, FitConfig, ModelConfig
from .observation import ExperimentFailureConfig, ObservationData
from .optimizer import BayesianOptimizer as _BayesianOptimizer


def _row_count(value: Any) -> int:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        return 1
    return int(tensor.shape[0]) if tensor.ndim > 1 else int(tensor.numel())


def _normalize_cost_observations(
    value: Any,
    *,
    X: Any,
    name: str,
) -> torch.Tensor:
    """Normalize evaluation costs to a finite positive ``n x 1`` tensor."""

    if isinstance(X, torch.Tensor):
        cost = torch.as_tensor(value, dtype=X.dtype, device=X.device)
    else:
        cost = torch.as_tensor(value)
        if not cost.is_floating_point():
            cost = cost.to(dtype=torch.get_default_dtype())
    if cost.ndim == 0:
        cost = cost.reshape(1, 1)
    elif cost.ndim == 1:
        cost = cost.unsqueeze(-1)
    if cost.ndim != 2 or int(cost.shape[-1]) != 1:
        raise ValueError(f"{name} must have shape n or n x 1.")
    if int(cost.shape[0]) != _row_count(X):
        raise ValueError(f"{name} must contain one cost value per input row.")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError(f"{name} must contain only finite values.")
    if not bool((cost > 0).all()):
        raise ValueError(f"{name} must be strictly positive.")
    return cost


def _success_statuses(status: Any, *, n_rows: int) -> bool:
    values = [status] * n_rows if isinstance(status, str) else list(status)
    return len(values) == n_rows and all(str(value).lower() == "success" for value in values)


class BayesianOptimizer(_BayesianOptimizer):
    """Public optimizer with synchronized evaluation-cost observations."""

    train_cost: torch.Tensor | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.train_cost = None

    def fit(
        self,
        train_X: Any | None = None,
        train_Y: Any | None = None,
        train_Yvar: Any | None = None,
        *,
        train_cost: Any | None = None,
        observation_data: ObservationData | None = None,
        observed_mask: Any | None = None,
        failed_mask: Any | None = None,
        pending_mask: Any | None = None,
        failure_config: ExperimentFailureConfig | None = None,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> BayesianOptimizer:
        """Fit the objective model and optionally initialize cost observations."""

        previous_cost = self.train_cost
        internal_observation_refit = observation_data is not None and train_cost is None
        if train_cost is not None:
            if observation_data is not None:
                raise ValueError(
                    "train_cost cannot be combined with observation_data; pass direct "
                    "successful train_X/train_Y observations."
                )
            if any(
                value is not None
                for value in (observed_mask, failed_mask, pending_mask, failure_config)
            ):
                raise ValueError(
                    "train_cost currently supports direct successful observations only."
                )
            if train_X is None:
                raise ValueError("train_X is required when train_cost is supplied.")
            normalized_cost = _normalize_cost_observations(
                train_cost,
                X=train_X,
                name="train_cost",
            )
        else:
            normalized_cost = None

        result = super().fit(
            train_X,
            train_Y,
            train_Yvar,
            observation_data=observation_data,
            observed_mask=observed_mask,
            failed_mask=failed_mask,
            pending_mask=pending_mask,
            failure_config=failure_config,
            model_config=model_config,
            fit_config=fit_config,
        )
        if normalized_cost is not None:
            if _row_count(self.train_X) != int(normalized_cost.shape[0]):
                raise ValueError(
                    "train_cost must align with the successful objective training rows."
                )
            self.train_cost = normalized_cost
        elif internal_observation_refit:
            self.train_cost = previous_cost
        else:
            self.train_cost = None
        return result

    def refit(self, *, fit_config: FitConfig | None = None) -> BayesianOptimizer:
        """Refit objective models without discarding synchronized cost history."""

        cost = self.train_cost
        result = super().refit(fit_config=fit_config)
        self.train_cost = cost
        return result

    def tell(
        self,
        X_new: Any,
        Y_new: Any,
        new_Yvar: Any | None = None,
        *,
        new_cost: Any | None = None,
        status: Any = "success",
        observed_mask: Any | None = None,
        refit: bool = True,
        fit_config: FitConfig | None = None,
    ) -> BayesianOptimizer:
        """Append objective and optional cost observations atomically by row."""

        tracking_cost = self.train_cost is not None
        if tracking_cost and new_cost is None:
            raise ValueError(
                "new_cost is required because this optimizer tracks train_cost."
            )
        if not tracking_cost and new_cost is not None:
            raise ValueError(
                "Cannot start cost tracking after fit without historical train_cost. "
                "Call fit(..., train_cost=...) first."
            )

        normalized_cost = None
        if new_cost is not None:
            n_rows = _row_count(X_new)
            if observed_mask is not None or not _success_statuses(status, n_rows=n_rows):
                raise ValueError(
                    "new_cost currently supports successful observations without observed_mask only."
                )
            normalized_cost = _normalize_cost_observations(
                new_cost,
                X=X_new,
                name="new_cost",
            )

        result = super().tell(
            X_new,
            Y_new,
            new_Yvar,
            status=status,
            observed_mask=observed_mask,
            refit=refit,
            fit_config=fit_config,
        )
        if normalized_cost is not None:
            self.train_cost = torch.cat(
                [self.train_cost, normalized_cost.to(self.train_cost)],
                dim=0,
            )
        return result

    def update_data(
        self,
        X_new: Any,
        Y_new: Any,
        new_Yvar: Any | None = None,
        *,
        new_cost: Any | None = None,
        append: bool = True,
    ) -> BayesianOptimizer:
        """Update objective/cost state while preserving the row-alignment contract."""

        if not append:
            return self.fit(
                X_new,
                Y_new,
                new_Yvar,
                train_cost=new_cost,
                model_config=self.model_config,
                fit_config=self.fit_config,
                failure_config=self.failure_config,
            )
        if self.train_cost is not None or new_cost is not None:
            return self.tell(
                X_new,
                Y_new,
                new_Yvar,
                new_cost=new_cost,
                status="success",
                refit=False,
            )
        return super().update_data(
            X_new,
            Y_new,
            new_Yvar,
            append=append,
        )

    def _cost_state_acquisition_config(
        self,
        config: AcquisitionConfig | None,
    ) -> AcquisitionConfig | None:
        """Fill deferred learned-GP cost config from optimizer cost history."""

        if config is None:
            return None
        kwargs = dict(config.acqf_kwargs or {})
        raw = kwargs.get("cost_config")
        if raw is None:
            return config

        from bochan.models.multifidelity import FidelityCostConfig

        if isinstance(raw, Mapping):
            kind = str(raw.get("kind", "affine")).strip().lower()
            if kind != "learned_gp":
                return config
            if any(key in raw for key in ("train_X", "train_cost", "cost_model")):
                return config
            if self.train_cost is None:
                raise ValueError(
                    "cost_config kind='learned_gp' requires optimizer train_cost. "
                    "Call fit(..., train_cost=...) first or provide cost training data explicitly."
                )
            resolved = dict(raw)
            resolved["train_X"] = self.train_X
            resolved["train_cost"] = self.train_cost
            kwargs["cost_config"] = resolved
            return replace(config, acqf_kwargs=kwargs)

        if isinstance(raw, FidelityCostConfig) and raw.kind == "learned_gp":
            if raw.cost_model is not None or raw.train_X is not None:
                return config
            if self.train_cost is None:
                raise ValueError(
                    "Deferred learned_gp cost_config requires optimizer train_cost."
                )
            kwargs["cost_config"] = replace(
                raw,
                train_X=self.train_X,
                train_cost=self.train_cost,
            )
            return replace(config, acqf_kwargs=kwargs)
        return config

    def _prepare_acquisition(
        self,
        acq_config: AcquisitionConfig | None,
        data_context: Any | None,
    ) -> tuple[AcquisitionConfig, Any, Any]:
        configured = acq_config if acq_config is not None else self.acq_config
        configured = self._cost_state_acquisition_config(configured)
        return super()._prepare_acquisition(configured, data_context)


__all__ = ["BayesianOptimizer"]
