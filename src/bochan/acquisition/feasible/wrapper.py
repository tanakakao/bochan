from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .constraints import (
    FeasibilityConstraintSpec,
    OrdinalRankConstraintSpec,
    constraint_value_from_class_probs,
    constraint_value_from_ordinal_probs,
    constraint_value_from_output,
    normalize_output_index,
    soft_feasibility_from_constraint_values,
)

QReduction = Literal["mean", "min", "prod", "max", "none"]
ConstraintReduction = Literal["prod", "min", "mean"]
PosteriorMode = Literal["objective", "mean", "probability", "expected_utility"]
ConstraintSpec = FeasibilityConstraintSpec | OrdinalRankConstraintSpec


def combine_acquisition_with_feasibility(base_value: Tensor, feasibility: Tensor) -> Tensor:
    """Combine acquisition utility and feasibility without sign inversion.

    Positive acquisition values keep multiplicative weighting ``a * p_f``.
    Negative values use the bounded penalty ``a * (2 - p_f)`` so decreasing
    feasibility always makes a maximization acquisition worse without dividing
    by probabilities near zero.
    """

    if not torch.is_floating_point(base_value):
        base_value = base_value.to(dtype=torch.get_default_dtype())
    pf = feasibility.to(dtype=base_value.dtype, device=base_value.device)
    return torch.where(
        base_value >= 0,
        base_value * pf,
        base_value * (2.0 - pf),
    )


class FeasibilityWeightedAcquisition(AcquisitionFunction):
    """既存 acquisition に soft feasibility を適用する wrapper。

    `target_class` / `target_classes` 付きの `FeasibilityConstraintSpec` と
    `OrdinalRankConstraintSpec` は、model.class_probs_list() から確率を取得して
    評価する。そのため、モデル定義側の positive_class に依存しない。

    獲得値が負になり得る acquisition でも、feasibility が低下するほど
    maximization score が必ず悪化する sign-safe weighting を用いる。
    """

    def __init__(
        self,
        acqf: AcquisitionFunction,
        model,
        constraints: Sequence[ConstraintSpec],
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

        try:
            return self.model.posterior(X, output_mode=self.posterior_mode)
        except TypeError as exc:
            if "output_mode" not in str(exc):
                raise
            return self.model.posterior(X)

    def _output_index(self, output: Any) -> int:
        return normalize_output_index(
            output,
            output_names=getattr(self.model, "output_names", None),
        )

    def _ordinal_submodel(self, output: Any) -> Any | None:
        """Return the ordinal model that owns one constrained output, if known."""

        idx = self._output_index(output)
        specs = getattr(self.model, "specs", None)
        if specs is not None and idx < len(specs):
            spec = specs[idx]
            if str(getattr(spec, "task_type", "")) == "ordinal":
                return getattr(spec, "model", None)
            return None

        models = getattr(self.model, "models", None)
        if models is not None and idx < len(models):
            candidate = models[idx]
            if getattr(candidate, "ordinal_likelihood", None) is not None:
                return candidate
            likelihood = getattr(candidate, "likelihood", None)
            if likelihood is not None and "ordinal" in type(likelihood).__name__.lower():
                return candidate
            return None

        if idx == 0:
            if getattr(self.model, "ordinal_likelihood", None) is not None:
                return self.model
            likelihood = getattr(self.model, "likelihood", None)
            if likelihood is not None and "ordinal" in type(likelihood).__name__.lower():
                return self.model
        return None

    @staticmethod
    def _call_latent_posterior(model: Any, X: Tensor) -> Any | None:
        for name in ("latent_posterior", "posterior_latent", "posterior_f", "posterior"):
            fn = getattr(model, name, None)
            if not callable(fn):
                continue
            try:
                return fn(X=X)
            except TypeError:
                try:
                    return fn(X)
                except TypeError:
                    continue
        return None

    def _ordinal_gradient_proxy(
        self,
        X: Tensor,
        output: Any,
        exact_probs: Tensor,
    ) -> Tensor | None:
        """Build a stable ordered-logit gradient proxy from latent posterior mean.

        The forward value remains the full posterior-predictive class probability.
        Only candidate gradients use the plug-in probability at the latent mean,
        avoiding unstable variance/quadrature derivatives near degenerate points.
        """

        ordinal_model = self._ordinal_submodel(output)
        if ordinal_model is None:
            return None
        posterior = self._call_latent_posterior(ordinal_model, X)
        if posterior is None:
            return None
        latent_mean = getattr(posterior, "mean", None)
        if latent_mean is None:
            return None

        likelihood = getattr(ordinal_model, "ordinal_likelihood", None)
        if likelihood is None:
            likelihood = getattr(ordinal_model, "likelihood", None)
        if likelihood is None:
            return None

        probability_fn = getattr(likelihood, "probs_from_latent", None)
        if not callable(probability_fn):
            probability_fn = getattr(likelihood, "class_probs_from_f", None)
        if not callable(probability_fn):
            return None

        proxy = probability_fn(latent_mean)
        while proxy.ndim > exact_probs.ndim and proxy.shape[0] == 1:
            proxy = proxy.squeeze(0)
        if proxy.shape != exact_probs.shape:
            return None
        return proxy.to(dtype=exact_probs.dtype, device=exact_probs.device)

    def _class_probs_for_output(self, X: Tensor, output) -> Tensor:
        if not callable(getattr(self.model, "class_probs_list", None)):
            raise AttributeError(
                "Class-probability feasibility constraints require "
                "model.class_probs_list(X, output_indices=...)."
            )
        probs_list = self.model.class_probs_list(X, output_indices=[output])
        if len(probs_list) != 1:
            raise RuntimeError(
                "Expected exactly one class probability tensor for "
                f"output={output!r}, got {len(probs_list)}."
            )
        exact_probs = probs_list[0]
        proxy = self._ordinal_gradient_proxy(X, output, exact_probs)
        if proxy is None or not exact_probs.requires_grad:
            return exact_probs

        # Straight-through surrogate: preserve exact posterior-predictive values
        # while taking candidate gradients through the stable latent-mean proxy.
        return exact_probs.detach() + proxy - proxy.detach()

    def _ordinal_rank_constraint_value(
        self,
        X: Tensor,
        spec: OrdinalRankConstraintSpec,
    ) -> Tensor:
        """OrdinalRankConstraintSpec を class probability から評価する。"""

        probs = self._class_probs_for_output(X, spec.output)
        return constraint_value_from_ordinal_probs(probs, spec)

    def _class_probability_constraint_value(
        self,
        X: Tensor,
        spec: FeasibilityConstraintSpec,
    ) -> Tensor:
        """target_class / target_classes を class probability から評価する。"""

        probs = self._class_probs_for_output(X, spec.output)
        return constraint_value_from_class_probs(probs, spec)

    def constraint_values(self, X: Tensor) -> Tensor:
        """posterior mean / class probability 上で制約値を評価する。

        Returns:
            Tensor:
                shape = ``X.shape[:-1] + (n_constraints,)``。
                各 constraint は ``<= 0`` が feasible。
        """

        output_names = getattr(self.model, "output_names", None)
        values = []

        continuous_specs = [
            spec
            for spec in self.constraints
            if isinstance(spec, FeasibilityConstraintSpec) and not spec.has_target_classes
        ]
        posterior_mean = None
        if len(continuous_specs) > 0:
            posterior_mean = self._posterior(X).mean

        for spec in self.constraints:
            if isinstance(spec, OrdinalRankConstraintSpec):
                values.append(self._ordinal_rank_constraint_value(X, spec).unsqueeze(-1))
                continue
            if spec.has_target_classes:
                values.append(self._class_probability_constraint_value(X, spec).unsqueeze(-1))
                continue

            idx = normalize_output_index(spec.output, output_names=output_names)
            if posterior_mean is None:
                posterior_mean = self._posterior(X).mean
            if idx >= posterior_mean.shape[-1]:
                raise IndexError(
                    f"Constraint output index {idx} is out of range for "
                    f"posterior.mean.shape={tuple(posterior_mean.shape)}."
                )
            y = posterior_mean[..., idx]
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

        if self.reduce_q == "none" and base_value.shape == pf.shape[:-1]:
            pf = pf.mean(dim=-1)

        try:
            return combine_acquisition_with_feasibility(base_value, pf)
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not combine base acquisition value with feasibility. "
                f"base_value.shape={tuple(base_value.shape)}, feasibility.shape={tuple(pf.shape)}. "
                "Consider reduce_q='mean', 'min', or 'prod'."
            ) from exc

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if hasattr(self.acqf, "set_X_pending"):
            self.acqf.set_X_pending(X_pending)
        self.X_pending = X_pending


__all__ = [
    "ConstraintReduction",
    "ConstraintSpec",
    "FeasibilityWeightedAcquisition",
    "PosteriorMode",
    "QReduction",
    "combine_acquisition_with_feasibility",
]
