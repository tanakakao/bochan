"""InputPerturbation support for multiclass qNParEGO."""

from __future__ import annotations

from functools import wraps
from typing import Any

from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class _FixedXMultiOutputObjective(MCMultiOutputObjective):
    """Forward a fixed raw X to an objective that is called without X."""

    def __init__(self, objective: MCMultiOutputObjective, X: Tensor) -> None:
        super().__init__()
        self.objective = objective
        self.X = X
        self._verify_output_shape = False

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        return self.objective(samples, X=self.X)


def _aggregates_input_perturbation(objective: object | None) -> bool:
    return bool(getattr(objective, "_bochan_aggregates_input_perturbation", False))


def configure_multiclass_nparego_input_perturbation(module: Any) -> None:
    """Pass raw candidate and baseline inputs to qNParEGO objectives."""
    acquisition_cls = module.qMultiOutputMulticlassNParEGO
    if getattr(acquisition_cls, "_bochan_input_perturbation_patched", False):
        return

    original_init = acquisition_cls.__init__
    original_forward = acquisition_cls.forward

    @wraps(original_init)
    def supported_init(self, model, X_baseline: Tensor, ref_point, *args, **kwargs):
        objective = kwargs.get("objective")
        wrapped_objective = objective
        if _aggregates_input_perturbation(objective):
            Xb = module.ensure_q_batch(X_baseline)
            wrapped_objective = _FixedXMultiOutputObjective(objective, Xb)
            kwargs["objective"] = wrapped_objective

        original_init(self, model, X_baseline, ref_point, *args, **kwargs)

        if wrapped_objective is not objective:
            self.base_objective = objective
            self.objective = objective

    @wraps(original_forward)
    @t_batch_mode_transform()
    def supported_forward(self, X: Tensor) -> Tensor:
        Xq = module.ensure_q_batch(X)
        post = self.model.posterior(Xq)
        samples = self.get_posterior_samples(post)
        values = self.base_objective(samples, X=Xq)
        scalarized = self._scalarize(values)
        improvement = (scalarized - self.best_value.to(scalarized)).clamp_min(0.0)
        return module._reduce_sample_and_q_to_tbatch(improvement, Xq)

    acquisition_cls.__init__ = supported_init
    acquisition_cls.forward = supported_forward
    acquisition_cls._bochan_input_perturbation_patched = True
    acquisition_cls._bochan_original_init = original_init
    acquisition_cls._bochan_original_forward = original_forward


__all__ = ["configure_multiclass_nparego_input_perturbation"]
