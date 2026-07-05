"""InputPerturbation compatibility for multiclass qNEHVI baselines."""

from __future__ import annotations

from functools import wraps
from typing import Any

from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
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


def patch_multiclass_nehvi_baseline_input(module: Any) -> None:
    """Pass raw baseline inputs to multiclass qNEHVI objectives.

    The multiclass qNEHVI helper evaluates ``posterior.mean`` with ``X=None``.
    InputPerturbation-aware objectives need the raw q dimension from X to decide
    whether the posterior values are already reduced or still expanded by
    ``n_w``. Wrap the objective only for baseline partitioning so it receives
    the corresponding raw ``X_baseline``.
    """
    original = module._baseline_partitioning_from_model
    if getattr(original, "_bochan_nehvi_baseline_input_patched", False):
        return

    @wraps(original)
    def compatible_baseline_partitioning(
        *,
        model,
        X_baseline: Tensor,
        ref_point: Tensor,
        objective: MCMultiOutputObjective,
    ) -> Any:
        Xb = module.ensure_q_batch(X_baseline)
        fixed_x_objective = _FixedXMultiOutputObjective(objective, Xb)
        return original(
            model=model,
            X_baseline=X_baseline,
            ref_point=ref_point,
            objective=fixed_x_objective,
        )

    setattr(compatible_baseline_partitioning, "_bochan_nehvi_baseline_input_patched", True)
    setattr(compatible_baseline_partitioning, "_bochan_original", original)
    module._baseline_partitioning_from_model = compatible_baseline_partitioning


__all__ = ["patch_multiclass_nehvi_baseline_input"]
