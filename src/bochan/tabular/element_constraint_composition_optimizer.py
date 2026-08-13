"""Linear element constraints for multi-site composition optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .composition_element_constraints import (
    CompositionElementConstraintProjector,
    CompositionElementConstraintResolver,
)
from .composition_total_constraints import CompositionTotalConstraintResolver
from .composition_variable_total_transform import CompositionVariableTotalTransform
from .converter import dataframe_to_tensors
from .element_column_composition_optimizer import (
    TabularBayesianOptimizer as _ElementColumnTabularBayesianOptimizer,
)

class TabularBayesianOptimizer(_ElementColumnTabularBayesianOptimizer):
    """Support linear equality and inequality constraints between elements.

    Constraints are expressed in atomic or weight amounts and may span multiple
    composition sites. Fixed-total Fraction coordinates are also forwarded to
    the existing named linear-constraint optimizer. For CLR, ALR, ILR, variable
    totals, sparsity, or stepped compositions, candidates are repaired after
    inverse transformation with a mixed-integer linear projection.
    """

    composition_variable_total_transform = CompositionVariableTotalTransform()
    composition_element_constraint_resolver = CompositionElementConstraintResolver()

    def __init__(
        self,
        model_config: Any | None = None,
        fit_config: Any | None = None,
        *,
        composition_element_constraints: Sequence[Any] | None = None,
        composition_constraint_rerank: bool = True,
        composition_constraint_rerank_factor: int = 4,
        composition_constraint_max_supports: int = 256,
        **kwargs: Any,
    ) -> None:
        self.composition_element_constraints = (
            self.composition_element_constraint_resolver.normalize(
                composition_element_constraints
            )
        )
        self.composition_constraint_rerank = bool(composition_constraint_rerank)
        self.composition_constraint_rerank_factor = int(
            composition_constraint_rerank_factor
        )
        self.composition_constraint_max_supports = int(
            composition_constraint_max_supports
        )
        if self.composition_constraint_rerank_factor < 1:
            raise ValueError("composition_constraint_rerank_factor must be >= 1.")
        if self.composition_constraint_max_supports < 1:
            raise ValueError("composition_constraint_max_supports must be >= 1.")
        super().__init__(
            model_config=model_config,
            fit_config=fit_config,
            **kwargs,
        )
        self._make_element_constraint_projector().validate()

    def _make_element_constraint_projector(
        self,
    ) -> CompositionElementConstraintProjector:
        """Build the projector from the current fitted composition context."""

        return CompositionElementConstraintProjector(
            composition_sites=self.composition_sites,
            composition_element_constraints=self.composition_element_constraints,
            composition_transformers=getattr(self, "composition_transformers_", {}),
            max_supports=self.composition_constraint_max_supports,
        )

    def _named_element_constraints(self) -> list[tuple[Any, ...]]:
        """Resolve optimizer-native constraints through the explicit resolver."""

        return self.composition_element_constraint_resolver.named_constraints(
            self.composition_element_constraints,
            self.composition_sites,
            self.composition_transformers_,
        )













    def inverse_compositions(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        restored = super().inverse_compositions(
            data,
            repair=repair,
            keep_coordinates=keep_coordinates,
        )
        restored = self.composition_variable_total_transform.inverse_compositions(
            data,
            restored,
            composition_sites=self.composition_sites,
            composition_transformers=self.composition_transformers_,
            multi_site_composition_enabled=self.multi_site_composition_enabled,
            repair=repair,
        )
        if (
            repair
            and self.multi_site_composition_enabled
            and self.composition_element_constraints
        ):
            restored = (
                self._make_element_constraint_projector().repair_frame(restored)
            )
        return restored


    @staticmethod
    def _requested_q(opt_config: Any, kwargs: Mapping[str, Any]) -> int:
        direct = kwargs.get("q")
        if isinstance(direct, int):
            return max(1, direct)
        if isinstance(opt_config, Mapping) and isinstance(opt_config.get("q"), int):
            return max(1, int(opt_config["q"]))
        configured = getattr(opt_config, "q", None)
        return max(1, int(configured)) if isinstance(configured, int) else 1

    def _rerank_candidates(
        self,
        candidates: Any,
        acqf: Any,
        requested_q: int,
    ) -> tuple[Any, Any]:
        import torch

        unique = candidates.drop_duplicates().reset_index(drop=True)
        transformed = self.transform_compositions(unique)
        data_config = replace(
            self.data_config,
            input_cols=self.dataset.feature_names,
            target_cols=None,
        )
        X = dataframe_to_tensors(transformed, data_config).X
        with torch.no_grad():
            try:
                scores = acqf(X.unsqueeze(-2))
            except (RuntimeError, ValueError, TypeError):
                scores = acqf(X)
        scores = scores.detach().reshape(-1)
        if scores.numel() != len(unique):
            raise ValueError(
                "The acquisition function did not return one score per repaired "
                "candidate."
            )
        order = torch.argsort(scores, descending=True)[:requested_q]
        indices = order.detach().cpu().numpy().tolist()
        return unique.iloc[indices].reset_index(drop=True), scores[order]

    def candidate(
        self,
        acq_config: Any | None = None,
        opt_config: Any | None = None,
        *,
        return_dataframe: bool = True,
        return_result: bool = False,
        return_composition: bool = True,
        keep_composition_coordinates: bool = False,
        composition_constraint_rerank: bool | None = None,
        composition_constraint_rerank_factor: int | None = None,
        **kwargs: Any,
    ) -> Any:
        named_constraints = self._named_element_constraints()
        opt_config = CompositionTotalConstraintResolver.merge_optimize_config(
            opt_config,
            named_constraints,
        )
        rerank = (
            self.composition_constraint_rerank
            if composition_constraint_rerank is None
            else bool(composition_constraint_rerank)
        )
        if (
            not self.composition_element_constraints
            or not rerank
            or return_result
            or not return_dataframe
            or not return_composition
        ):
            return super().candidate(
                acq_config=acq_config,
                opt_config=opt_config,
                return_dataframe=return_dataframe,
                return_result=return_result,
                return_composition=return_composition,
                keep_composition_coordinates=keep_composition_coordinates,
                **kwargs,
            )

        requested_q = self._requested_q(opt_config, kwargs)
        factor = (
            self.composition_constraint_rerank_factor
            if composition_constraint_rerank_factor is None
            else int(composition_constraint_rerank_factor)
        )
        if factor < 1:
            raise ValueError("composition_constraint_rerank_factor must be >= 1.")
        call_kwargs = dict(kwargs)
        call_kwargs["q"] = requested_q * factor
        result = super().candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            return_dataframe=True,
            return_result=True,
            return_composition=False,
            **call_kwargs,
        )
        raw_candidates = self.candidates_to_dataframe(result.candidates)
        repaired = self.inverse_compositions(
            raw_candidates,
            repair=True,
            keep_coordinates=keep_composition_coordinates,
        )
        try:
            return self._rerank_candidates(repaired, result.acqf, requested_q)
        except (RuntimeError, ValueError, TypeError, KeyError):
            selected = repaired.drop_duplicates().head(requested_q).reset_index(drop=True)
            return selected, result.acq_value


__all__ = ["TabularBayesianOptimizer"]
