"""Support helpers for wide correlated ordinal acquisitions."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

_APPLIED = False


class _WideOrdinalTaskProxy:
    """Expose one task of a wide correlated ordinal model as a scalar model."""

    def __init__(self, parent: Any, task_index: int) -> None:
        self.parent = parent
        self.task_index = int(task_index)

    @property
    def likelihood(self):
        return self.parent.likelihood

    @property
    def ordinal_likelihood(self):
        return self.parent.likelihood

    @property
    def input_transform(self):
        return getattr(self.parent, "input_transform", None)

    def eval(self):
        self.parent.eval()
        likelihood = getattr(self.parent, "likelihood", None)
        if likelihood is not None and hasattr(likelihood, "eval"):
            likelihood.eval()
        return self

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ):
        """Evaluate the underlying long-format model at one fixed task id."""

        if output_indices not in (None, [0], (0,)):
            raise ValueError("A fixed ordinal task proxy exposes one output only.")
        X = torch.as_tensor(X)
        task = torch.full(
            (*X.shape[:-1], 1),
            float(self.task_index),
            device=X.device,
            dtype=X.dtype,
        )
        X_long = torch.cat([X, task], dim=-1)

        from bochan.models.ordinal.base.multitask import MultiTaskOrdinalGPModel

        return MultiTaskOrdinalGPModel.posterior(
            self.parent,
            X_long,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def __getattr__(self, name: str):
        return getattr(self.parent, name)


def _wide_task_proxies(model: Any) -> list[Any] | None:
    """Return fixed-task proxies when ``model`` is a wide correlated adapter."""

    if not callable(getattr(model, "_wrap_wide_posterior", None)):
        return None
    try:
        num_tasks = int(model.num_tasks)
    except (AttributeError, TypeError, ValueError):
        return None
    if num_tasks < 1:
        return None
    return [_WideOrdinalTaskProxy(model, index) for index in range(num_tasks)]


def apply_ordinal_multitask() -> None:
    """Install likelihood and task-proxy support.

    A long-format correlated ordinal model learns one shared likelihood and task
    covariance, while older multi-output acquisitions expect a ModelList-style
    ``model.models`` collection. Fixed-task proxies preserve the correlated
    parent posterior and expose one scalar task to those acquisitions without
    copying model or likelihood parameters.

    Level-set estimation now inherits the ordinal active-learning multi-output
    base directly, so it uses the active-learning submodel resolver and no longer
    needs a separate level-set runtime patch.
    """

    global _APPLIED
    if _APPLIED:
        return

    from bochan.acquisition.ordinal.active_learning import multi_output as active
    from bochan.acquisition.ordinal.bayesian_optimization import multi_output as bo

    original_extract = bo._extract_ordinal_likelihoods

    def supported_extract(
        model: Any,
        ordinal_likelihoods: Any = None,
    ) -> list[Any]:
        likelihoods = list(original_extract(model, ordinal_likelihoods))
        try:
            num_outputs = int(getattr(model, "num_outputs", 1))
        except (TypeError, ValueError):
            num_outputs = 1
        if len(likelihoods) == 1 and num_outputs > 1:
            return likelihoods * num_outputs
        return likelihoods

    supported_extract._bochan_wide_multitask_variantsible = True  # type: ignore[attr-defined]
    supported_extract._bochan_original = original_extract  # type: ignore[attr-defined]
    bo._extract_ordinal_likelihoods = supported_extract

    original_active_resolve = active._resolve_submodels

    def supported_active_resolve(model: Any) -> list[Any]:
        proxies = _wide_task_proxies(model)
        return proxies if proxies is not None else list(original_active_resolve(model))

    active._resolve_submodels = supported_active_resolve
    _APPLIED = True


__all__ = ["apply_ordinal_multitask"]