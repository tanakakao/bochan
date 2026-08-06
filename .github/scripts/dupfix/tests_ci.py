from __future__ import annotations

from .common import read, replace_once, write

TEST_FILE = '''from __future__ import annotations

import torch

from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)
from bochan.acquisition.binary.base import _BinaryClassificationAcqBase
from bochan.acquisition.multiclass.bayesian_optimization.single_output import (
    _MulticlassProbabilityBOBase,
)


class _ConcreteBinary(_BinaryClassificationAcqBase):
    def forward(self, X):
        return X.sum(dim=(-1, -2))


class _ConcreteMulticlass(_MulticlassProbabilityBOBase):
    def forward(self, X):
        return X.sum(dim=(-1, -2))


def test_shared_hard_duplicate_helpers_only_reject_duplicates() -> None:
    duplicate = torch.tensor([[[0.2], [0.2], [0.4]]], dtype=torch.double)
    distinct = torch.tensor([[[0.2], [0.2002], [0.4]]], dtype=torch.double)
    pending = torch.tensor([[0.2]], dtype=torch.double)

    assert torch.isinf(hard_same_batch_duplicate_penalty_per_point(duplicate)).all()
    assert torch.equal(
        hard_same_batch_duplicate_penalty_per_point(distinct),
        torch.zeros(1, 3, dtype=torch.double),
    )
    assert torch.isinf(
        hard_reference_duplicate_penalty_per_point(duplicate[..., :1, :], pending)
    ).all()
    assert torch.equal(
        hard_reference_duplicate_penalty_per_point(distinct[..., 1:2, :], pending),
        torch.zeros(1, 1, dtype=torch.double),
    )


def test_binary_defaults_hard_exclude_same_batch_and_pending() -> None:
    acquisition = _ConcreteBinary(model=torch.nn.Identity())
    duplicate = torch.tensor([[[0.2], [0.2]]], dtype=torch.double)
    distinct = torch.tensor([[[0.2], [0.3]]], dtype=torch.double)

    assert acquisition.pending_penalty_weight == 0.0
    assert torch.isinf(acquisition._candidate_penalty_per_point(duplicate)).all()
    assert torch.equal(
        acquisition._candidate_penalty_per_point(distinct),
        torch.zeros_like(distinct[..., 0]),
    )

    acquisition.set_X_pending(torch.tensor([[0.2]], dtype=torch.double))
    assert torch.isinf(
        acquisition._candidate_penalty_per_point(
            torch.tensor([[[0.2]]], dtype=torch.double)
        )
    ).all()


def test_multiclass_defaults_hard_exclude_same_batch_and_pending() -> None:
    acquisition = _ConcreteMulticlass(model=torch.nn.Identity())
    duplicate = torch.tensor([[[0.2], [0.2]]], dtype=torch.double)
    distinct = torch.tensor([[[0.2], [0.3]]], dtype=torch.double)

    assert acquisition.same_batch_penalty_weight == 0.0
    assert torch.isinf(acquisition._same_batch_penalty(duplicate)).all()
    assert torch.equal(
        acquisition._same_batch_penalty(distinct),
        torch.zeros(1, dtype=torch.double),
    )

    acquisition.set_X_pending(torch.tensor([[0.2]], dtype=torch.double))
    assert torch.isinf(
        acquisition._pending_penalty_per_point(
            torch.tensor([[[0.2]]], dtype=torch.double)
        )
    ).all()


def test_hard_duplicate_exclusion_can_be_disabled() -> None:
    acquisition = _ConcreteBinary(
        model=torch.nn.Identity(),
        exclude_same_batch_duplicates=False,
        exclude_pending_duplicates=False,
    )
    acquisition.set_X_pending(torch.tensor([[0.2]], dtype=torch.double))
    duplicate = torch.tensor([[[0.2], [0.2]]], dtype=torch.double)

    assert torch.equal(
        acquisition._candidate_penalty_per_point(duplicate),
        torch.zeros(1, 2, dtype=torch.double),
    )
'''


def add_tests_and_ci() -> None:
    write("tests/test_classification_duplicate_exclusion.py", TEST_FILE)
    workflow = ".github/workflows/wide-multitask-smoke.yml"
    text = read(workflow)
    text = replace_once(
        text,
        "            tests/test_wide_multitask_acquisition_consistency.py \\\n"
        "            tests/test_wide_multitask_classification_acquisitions.py \\\n",
        "            tests/test_wide_multitask_acquisition_consistency.py \\\n"
        "            tests/test_wide_multitask_classification_acquisitions.py \\\n"
        "            tests/test_classification_duplicate_exclusion.py \\\n",
        label="wide pytest list",
    )
    text = replace_once(
        text,
        "            src/bochan/acquisition/classification_constraints.py \\\n",
        "            src/bochan/acquisition/classification_constraints.py \\\n"
        "            src/bochan/acquisition/_duplicate_exclusion.py \\\n"
        "            src/bochan/acquisition/binary/base.py \\\n"
        "            src/bochan/acquisition/ordinal/active_learning/single_output.py \\\n"
        "            src/bochan/acquisition/ordinal/active_learning/multi_output.py \\\n"
        "            src/bochan/acquisition/ordinal/active_learning/hetero_single_output.py \\\n"
        "            src/bochan/acquisition/ordinal/active_learning/hetero_multi_output.py \\\n"
        "            src/bochan/acquisition/multiclass/bayesian_optimization/single_output.py \\\n"
        "            src/bochan/acquisition/multiclass/active_learning/multi_output.py \\\n",
        label="wide ruff source list",
    )
    text = replace_once(
        text,
        "            tests/test_wide_multitask_public_transform.py \\\n"
        "            tests/test_wide_multitask_acquisition_consistency.py \\\n"
        "            tests/test_wide_multitask_classification_acquisitions.py \\\n",
        "            tests/test_wide_multitask_public_transform.py \\\n"
        "            tests/test_wide_multitask_acquisition_consistency.py \\\n"
        "            tests/test_wide_multitask_classification_acquisitions.py \\\n"
        "            tests/test_classification_duplicate_exclusion.py \\\n",
        label="wide ruff test list",
    )
    write(workflow, text)
