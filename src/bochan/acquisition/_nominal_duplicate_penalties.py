"""Nominal-candidate duplicate semantics for one-to-many input transforms.

InputPerturbation expands each optimization candidate into multiple evaluation
rows. Those rows must remain available to posterior / uncertainty scoring, but
they are not independent candidates for hard duplicate exclusion.

The mixin in this module keeps existing soft distance penalties in each
acquisition family's transformed feature space and evaluates hard duplicate
identity on the raw nominal q-batch and raw pending / observed references.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor

from ._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)


def _ensure_q_batch(X: Tensor) -> Tensor:
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _coerce_reference(X_ref: Any, *, ref: Tensor) -> Tensor | None:
    if X_ref is None:
        return None
    if torch.is_tensor(X_ref):
        out = X_ref
    elif isinstance(X_ref, (list, tuple)):
        pieces = [
            piece
            for item in X_ref
            if item is not None
            and (piece := _coerce_reference(item, ref=ref)) is not None
            and piece.numel() > 0
        ]
        if not pieces:
            return None
        out = torch.cat([piece.reshape(-1, piece.shape[-1]) for piece in pieces], dim=-2)
    else:
        out = torch.as_tensor(X_ref)
    return out.to(device=ref.device, dtype=ref.dtype).detach()


def _invalid_batch_from_penalty(penalty: Tensor) -> Tensor:
    if penalty.ndim == 0:
        return torch.isinf(penalty)
    return torch.isinf(penalty).any(dim=-1)


def _expand_invalid_like(invalid: Tensor, value: Tensor) -> Tensor:
    mask = invalid
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(value)


class NominalDuplicatePenaltyMixin:
    """Use raw candidates for hard duplicate identity checks.

    Posterior evaluation, score shape, risk aggregation, and soft transformed-
    space repulsion stay unchanged. Only the hard duplicate term is suppressed
    while the inherited soft term is calculated and then reapplied using raw
    optimization candidates.
    """

    _bochan_raw_X_for_duplicate_penalty: Tensor | None = None

    @contextmanager
    def _without_hard_duplicate_flags(self, *names: str) -> Iterator[None]:
        previous: dict[str, Any] = {}
        for name in names:
            if hasattr(self, name):
                previous[name] = getattr(self, name)
                setattr(self, name, False)
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(self, name, value)

    @contextmanager
    def _without_numeric_hard_duplicate_penalty(self) -> Iterator[None]:
        if not hasattr(self, "hard_duplicate_penalty"):
            yield
            return
        previous = getattr(self, "hard_duplicate_penalty")
        setattr(self, "hard_duplicate_penalty", 0.0)
        try:
            yield
        finally:
            setattr(self, "hard_duplicate_penalty", previous)

    def _raw_X(self, ref: Tensor) -> Tensor | None:
        raw_X = self._bochan_raw_X_for_duplicate_penalty
        if raw_X is None:
            return None
        return _ensure_q_batch(raw_X).to(device=ref.device, dtype=ref.dtype)

    def _nominal_same_batch_invalid(self, ref: Tensor) -> Tensor | None:
        raw_X = self._raw_X(ref)
        if raw_X is None:
            return None
        penalty = hard_same_batch_duplicate_penalty_per_point(
            raw_X,
            enabled=bool(getattr(self, "exclude_same_batch_duplicates", True)),
            tolerance=float(getattr(self, "hard_duplicate_tol", 1e-8)),
        )
        return _invalid_batch_from_penalty(penalty)

    def _nominal_reference_invalid(
        self,
        ref: Tensor,
        attr: str,
        enabled_attr: str,
    ) -> Tensor | None:
        raw_X = self._raw_X(ref)
        if raw_X is None:
            return None
        X_ref = _coerce_reference(getattr(self, attr, None), ref=raw_X)
        penalty = hard_reference_duplicate_penalty_per_point(
            raw_X,
            X_ref,
            enabled=bool(getattr(self, enabled_attr, True)),
            tolerance=float(getattr(self, "hard_duplicate_tol", 1e-8)),
        )
        return _invalid_batch_from_penalty(penalty)

    def _add_invalid_to_value(self, value: Tensor, invalid: Tensor | None) -> Tensor:
        if invalid is None:
            return value
        return torch.where(
            _expand_invalid_like(invalid, value),
            torch.full_like(value, torch.inf),
            value,
        )

    def _pending_penalty_per_point(self, X: Tensor) -> Tensor:
        with self._without_hard_duplicate_flags("exclude_pending_duplicates"):
            soft = super()._pending_penalty_per_point(X)
        invalid = self._nominal_reference_invalid(
            soft,
            "X_pending",
            "exclude_pending_duplicates",
        )
        return self._add_invalid_to_value(soft, invalid)

    def _observed_penalty_per_point(self, X: Tensor) -> Tensor:
        with self._without_hard_duplicate_flags("exclude_observed_duplicates"):
            soft = super()._observed_penalty_per_point(X)
        invalid = self._nominal_reference_invalid(
            soft,
            "X_observed",
            "exclude_observed_duplicates",
        )
        return self._add_invalid_to_value(soft, invalid)

    def _same_batch_duplicate_penalty_per_point(self, X: Tensor) -> Tensor:
        raw_X = self._raw_X(X)
        if raw_X is None:
            return super()._same_batch_duplicate_penalty_per_point(X)
        invalid = self._nominal_same_batch_invalid(X)
        zeros = X.new_zeros(X.shape[:-1])
        return self._add_invalid_to_value(zeros, invalid)

    def _same_batch_penalty_per_point(self, X: Tensor) -> Tensor:
        """Ordinal multi-output soft+hard same-batch penalty hook."""
        with self._without_hard_duplicate_flags("exclude_same_batch_duplicates"):
            soft = super()._same_batch_penalty_per_point(X)
        return self._add_invalid_to_value(
            soft,
            self._nominal_same_batch_invalid(soft),
        )

    def _same_batch_penalty(self, X: Tensor) -> Tensor:
        """Multiclass batch-level soft+hard same-batch penalty hook."""
        with self._without_hard_duplicate_flags("exclude_same_batch_duplicates"):
            soft = super()._same_batch_penalty(X)
        return self._add_invalid_to_value(
            soft,
            self._nominal_same_batch_invalid(soft),
        )

    def _same_batch_repulsion(self, X: Tensor) -> Tensor:
        """Keep custom soft LSE repulsion but remove expanded-row hard hits."""
        with self._without_numeric_hard_duplicate_penalty():
            return super()._same_batch_repulsion(X)

    def _reference_repulsion(self, X: Tensor, X_ref: Any, weight: float) -> Tensor:
        """Keep custom soft LSE reference repulsion; common hard checks follow."""
        with self._without_numeric_hard_duplicate_penalty():
            return super()._reference_repulsion(X, X_ref, weight)

    def _pointwise_reference_penalty(self, X: Tensor) -> Tensor:
        """Ordinal single-output combined pending/observed/same-batch hook."""
        with self._without_hard_duplicate_flags(
            "exclude_same_batch_duplicates",
            "exclude_pending_duplicates",
            "exclude_observed_duplicates",
        ):
            soft = super()._pointwise_reference_penalty(X)
        invalid = self._nominal_same_batch_invalid(soft)
        pending = self._nominal_reference_invalid(
            soft,
            "X_pending",
            "exclude_pending_duplicates",
        )
        observed = self._nominal_reference_invalid(
            soft,
            "X_observed",
            "exclude_observed_duplicates",
        )
        for extra in (pending, observed):
            if extra is not None:
                invalid = extra if invalid is None else (invalid | extra)
        return self._add_invalid_to_value(soft, invalid)

    def _pointwise_repulsion_penalty(self, X: Tensor) -> Tensor:
        """Ordinal multi-output combined penalty hook."""
        return (
            self._pending_penalty_per_point(X)
            + self._observed_penalty_per_point(X)
            + self._same_batch_penalty_per_point(X)
        )

    def _aggregated_reference_penalty(self, X: Tensor) -> Tensor:
        """Fantasy acquisitions that expose a single aggregated penalty."""
        with self._without_hard_duplicate_flags(
            "exclude_same_batch_duplicates",
            "exclude_pending_duplicates",
            "exclude_observed_duplicates",
        ):
            soft = super()._aggregated_reference_penalty(X)
        invalid = self._nominal_same_batch_invalid(soft)
        pending = self._nominal_reference_invalid(
            soft,
            "X_pending",
            "exclude_pending_duplicates",
        )
        observed = self._nominal_reference_invalid(
            soft,
            "X_observed",
            "exclude_observed_duplicates",
        )
        for extra in (pending, observed):
            if extra is not None:
                invalid = extra if invalid is None else (invalid | extra)
        return self._add_invalid_to_value(soft, invalid)

    def forward(self, X: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        previous = self._bochan_raw_X_for_duplicate_penalty
        self._bochan_raw_X_for_duplicate_penalty = _ensure_q_batch(X)
        try:
            return super().forward(X, *args, **kwargs)
        finally:
            self._bochan_raw_X_for_duplicate_penalty = previous


__all__ = ["NominalDuplicatePenaltyMixin"]
