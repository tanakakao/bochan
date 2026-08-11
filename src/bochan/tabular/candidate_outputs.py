"""Candidate-output helpers retained at the tabular boundary.

Candidate-set normalization is a high-level API contract and now lives in
:mod:`bochan.api.candidate_output`; tabular code only reuses that pure helper.
"""

from bochan.api.candidate_output import (
    select_best_candidate_set as _select_best_candidate_set,
)

__all__ = ["_select_best_candidate_set"]
