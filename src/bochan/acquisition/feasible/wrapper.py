from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .constraints import (
    FeasibilityConstraintSpec,
    constraint_value_from_output,
    normalize_output_index,
    soft_feasibility_from_constraint_values,
)

QReduction = Literal["mean", "min", "prod", "max", "none"]
ConstraintReduction = Literal["prod", "min", "mean"]
PosteriorMode = Literal["objective", "mean", "probability", "expected_utility"]


class FeasibilityWeightedAcquisition(AcquisitionFunction):
    """既存 acquisition に soft feasibility を掛ける wrapper。

    既存の `base_acqf(X)` をそのまま使い、別途 model posterior から
    ``P(feasible | X)`` 風の soft weight を計算して掛ける。

    これは以下の用途を想定する。
      - BoTorch の `constraints=` 引数に乗せにくい自作 acquisition
      - UCB / active learning / level-set などの score-based acquisition
      - HybridMultiOutputModel の分類・順序回帰出力を feasibility として使う場合
    """

    def __init__(
        self,
        acqf: AcquisitionFunction,
        model,
        constraints: Sequence[FeasibilityConstraintSpec],
        *,
        eta: float = 1e-3,
        posterior_mode: PosteriorMode = "objective",
        reduce_constraints: ConstraintReduction = "prod",
        reduce_q: QReduction = "mean",
        min_feasibility: float = 0.0,
        detach_feasibility: bool = False,
    ) -> None:
        super().__init__(model=model)

        if len(constraints) == 0:
            raise ValueError("At least one feasibility constraint is required.")
        if float(eta) <= 0.0:
            raise ValueError("eta must be positive.")
        if reduce_constraints not in {"prod", "min", "mean"}:
            raise ValueError("reduce_constraints must be 'prod', 'min', or 'mean'.")
        if reduce_q not in {"mean", "min", "prod", "max", "none"}:
            raise ValueError("reduce_q must be 'mean', 'min', 'prod', 'max', or 'none'.")
        if posterior_mode not in {"objective", "mean", "probability", "expected_utility"}:
            raise ValueError(
                "posterior_mode must be 'objective', 'mean', 'probability', or 'expected_utility'."
            )

        self.acqf = acqf
        self.constraints = list(constraints)
        self.eta = float(eta)
        self.posterior_mode = posterior_mode
        self.reduce_constraints = reduce_constraints
        self.reduce_q = reduce_q
        self.min_feasibility = float(min_feasibility)
        self.detach_feasibility = bool(detach_feasibility)
        self.set_X_pending(getattr(acqf, "X_pending", None))

    def _posterior(self, X: Tensor):
        if self.posterior_mode == "objective" and callable(getattr(self.model, "objective_posterior", None)):
            return self.model.objective_posterior(X)
        if self.posterior_mode == "probability" and callable(getattr(self.model, "probability_posterior", None)):
            return self.model.probability_posterior(X)
        if self.posterior_mode == "expected_utility" and callable(
            getattr(self.model, "expected_utility_posterior", None)
        ):
            return self.model.expected_utility_posterior(X)
        if self.posterior_mode == "mean" and callable(getattr(self.model, "mean_posterior", None)):
            return self.model.mean_posterior(X)

        # HybridMultiOutputModel 以外でも使えるように fallback する。
        return self.model.posterior(X, output_mode=self.posterior_mode)

    def constraint_values(self, X: Tensor) -> Tensor:
        """posterior mean 上で制約値を評価する。

        Returns:
            Tensor:
                shape = ``X.shape[:-1] + (n_constraints,)``。
                各 constraint は ``<= 0`` が feasible。
        """

        posterior = self._posterior(X)
        mean = posterior.mean
        output_names = getattr(self.model, "output_names", None)

        values = []
        for spec in self.constraints:
            idx = normalize_output_index(spec.output, output_names=output_names)
            if idx >= mean.shape[-1]:
                raise IndexError(
                    f"Constraint output index {idx} is out of range for "
                    f"posterior.mean.shape={tuple(mean.shape)}."
                )
            y = mean[..., idx]
            values.append(constraint_value_from_output(y, spec).unsqueeze(-1))

        return torch.cat(values, dim=-1)

    def feasibility_per_point(self, X: Tensor) -> Tensor:
        """各 q 点ごとの soft feasibility を返す。shape = ``X.shape[:-1]``。"""

        values = self.constraint_values(X)
        pf = soft_feasibility_from_constraint_values(
            values,
            eta=self.eta,
            reduce_constraints=self.reduce_constraints,
        )
        if self.min_feasibility > 0.0:
            pf = pf.clamp_min(self.min_feasibility)
        if self.detach_feasibility:
            pf = pf.detach()
        return pf

    def feasibility(self, X: Tensor) -> Tensor:
        """q-batch を集約した soft feasibility を返す。"""

        pf = self.feasibility_per_point(X)

        if self.reduce_q == "none":
            return pf
        if self.reduce_q == "mean":
            return pf.mean(dim=-1)
        if self.reduce_q == "min":
            return pf.min(dim=-1).values
        if self.reduce_q == "prod":
            return pf.prod(dim=-1)
        if self.reduce_q == "max":
            return pf.max(dim=-1).values

        raise ValueError(f"Unknown reduce_q={self.reduce_q!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        base_value = self.acqf(X)
        pf = self.feasibility(X)

        if self.reduce_q == "none":
            # base_value が joint score (*batch) の場合は、q feasibility を平均して合わせる。
            if base_value.shape == pf.shape[:-1]:
                pf = pf.mean(dim=-1)

        try:
            return base_value * pf
        except RuntimeError as e:
            raise RuntimeError(
                "Could not multiply base acquisition value by feasibility. "
                f"base_value.shape={tuple(base_value.shape)}, feasibility.shape={tuple(pf.shape)}. "
                "Consider reduce_q='mean', 'min', or 'prod'."
            ) from e

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        if hasattr(self.acqf, "set_X_pending"):
            self.acqf.set_X_pending(X_pending)
        self.X_pending = X_pending


__all__ = [
    "ConstraintReduction",
    "FeasibilityWeightedAcquisition",
    "PosteriorMode",
    "QReduction",
]
