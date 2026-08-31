from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.models import SingleTaskGP
from torch import Tensor, nn

from bochan.api import CandidateRepairConfig, OptimizeConfig
from bochan.api.optimizer.service import optimize_candidates
from bochan.api.support.best_subset import optimize_best_subset_candidates
from bochan.api.support.multi_group_best_subset import (
    BEST_SUBSET_GROUPS_KWARG,
    optimize_grouped_best_subset_candidates,
)
from bochan.api.support.one_shot import resolve_one_shot_ic_generator
from bochan.tabular.composition.logratio_support import (
    RawDecisionOneShotAcquisition,
    wrap_raw_decision_acquisition,
)


class _FakeOneShot(OneShotAcquisitionFunction):
    def __init__(self, num_auxiliary: int = 2) -> None:
        super().__init__(model=nn.Identity())
        self.num_auxiliary = int(num_auxiliary)
        self.last_x: Tensor | None = None

    def get_augmented_q_batch_size(self, q: int) -> int:
        return int(q) + self.num_auxiliary

    def extract_candidates(self, X_full: Tensor) -> Tensor:
        return X_full[..., : -self.num_auxiliary, :]

    def forward(self, X: Tensor) -> Tensor:
        self.last_x = X
        return -X.square().sum(dim=(-1, -2))


class _ScaleBridge:
    model_dim = 3
    decision_dim = 3

    @staticmethod
    def decision_to_model(values: Tensor) -> Tensor:
        return values * 2.0

    @staticmethod
    def model_to_decision(values: Tensor) -> Tensor:
        return values / 2.0


def _sparse_config(*, q: int = 2, final_candidate_postprocess=None) -> OptimizeConfig:
    floors = [
        ([0], [1.0], 0.05),
        ([1], [1.0], 0.05),
        ([2], [1.0], 0.05),
    ]
    return OptimizeConfig(
        q=q,
        num_restarts=2,
        raw_samples=8,
        sequential=False,
        repair_config=CandidateRepairConfig(
            comp_idx=[0, 1, 2],
            k=1,
            support_selection="best_subset",
            inequality_constraints=floors,
            inequality_sense="ge",
        ),
        optimizer_kwargs={"best_subset_strategy": "exact"},
        final_candidate_postprocess=final_candidate_postprocess,
    )


def _selected_from_constraints(config: OptimizeConfig, *, q: int) -> int:
    inactive: set[int] = set()
    for indices, _coefficients, rhs in config.equality_constraints or ():
        tensor = torch.as_tensor(indices)
        if tensor.ndim == 2 and tensor.shape == (1, 2) and float(rhs) == 0.0:
            point, feature = map(int, tensor[0].tolist())
            if point < q:
                inactive.add(feature)
    selected = sorted({0, 1, 2} - inactive)
    assert len(selected) == 1
    return selected[0]


def test_one_shot_support_constraints_only_target_actual_q_points() -> None:
    acqf = _FakeOneShot()
    config = _sparse_config(q=2)
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)
    seen: list[OptimizeConfig] = []

    def optimize_one(*, acqf: Any, bounds: Tensor, config: OptimizeConfig):
        seen.append(config)
        selected = _selected_from_constraints(config, q=2)
        candidate = torch.zeros(2, 3, dtype=bounds.dtype)
        candidate[:, selected] = 0.5
        return candidate, torch.tensor(float(selected + 1), dtype=bounds.dtype)

    candidates, value = optimize_best_subset_candidates(
        acqf=acqf,
        bounds=bounds,
        config=config,
        optimize_one=optimize_one,
    )

    assert float(value) == pytest.approx(3.0)
    assert candidates[:, 2].tolist() == pytest.approx([0.5, 0.5])
    assert len(seen) == 3
    for inner in seen:
        assert inner.repair_config is None
        assert inner.ensure_unique_candidates is False
        inter_equalities = [
            item
            for item in inner.equality_constraints or ()
            if torch.as_tensor(item[0]).ndim == 2
        ]
        assert len(inter_equalities) == 4  # q=2 times two inactive dimensions.
        assert {
            int(torch.as_tensor(item[0])[0, 0]) for item in inter_equalities
        } == {0, 1}
        inter_inequalities = [
            item
            for item in inner.inequality_constraints or ()
            if torch.as_tensor(item[0]).ndim == 2
        ]
        assert len(inter_inequalities) == 2
        assert all(float(item[2]) == pytest.approx(0.05) for item in inter_inequalities)

    # The fake forward would score the q-only tensor differently. Best Subset must
    # use the optimizer's full-tree value instead of re-evaluating q-only candidates.
    assert acqf.last_x is None


def test_grouped_one_shot_uses_one_support_per_group_and_optimizer_value() -> None:
    acqf = _FakeOneShot()
    bounds = torch.tensor([[0.0] * 4, [1.0] * 4], dtype=torch.double)
    config = OptimizeConfig(
        q=2,
        num_restarts=2,
        raw_samples=8,
        repair_config=CandidateRepairConfig(
            comp_idx=[0, 1, 2, 3],
            k=2,
            support_selection="best_subset",
        ),
        optimizer_kwargs={
            BEST_SUBSET_GROUPS_KWARG: (
                {"name": "a", "comp_idx": [0, 1], "k": 1},
                {"name": "b", "comp_idx": [2, 3], "k": 1},
            ),
            "best_subset_strategy": "exact",
        },
    )

    def optimize_one(*, acqf: Any, bounds: Tensor, config: OptimizeConfig):
        inactive = {
            int(torch.as_tensor(item[0])[0, 1])
            for item in config.equality_constraints or ()
            if torch.as_tensor(item[0]).ndim == 2
        }
        support = sorted({0, 1, 2, 3} - inactive)
        assert len(support) == 2
        candidate = torch.zeros(2, 4, dtype=bounds.dtype)
        candidate[:, support] = 0.5
        score = float(sum(support))
        return candidate, torch.tensor(score, dtype=bounds.dtype)

    candidates, value = optimize_grouped_best_subset_candidates(
        acqf=acqf,
        bounds=bounds,
        config=config,
        optimize_one=optimize_one,
    )

    assert float(value) == pytest.approx(4.0)  # support [1, 3]
    assert candidates[0].tolist() == pytest.approx([0.0, 0.5, 0.0, 0.5])
    assert acqf.last_x is None


def test_one_shot_rejects_final_projection_and_mixed_enumeration() -> None:
    acqf = _FakeOneShot()
    bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)

    with pytest.raises(ValueError, match="conditional re-optimization"):
        optimize_best_subset_candidates(
            acqf=acqf,
            bounds=bounds,
            config=_sparse_config(final_candidate_postprocess=lambda x: x),
            optimize_one=lambda **kwargs: (torch.zeros(2, 3), torch.tensor(0.0)),
        )

    mixed = _sparse_config()
    mixed.fixed_features_list = [{2: 0.0}, {2: 1.0}]
    with pytest.raises(ValueError, match="fixed_features_list"):
        optimize_best_subset_candidates(
            acqf=acqf,
            bounds=bounds,
            config=mixed,
            optimize_one=lambda **kwargs: (torch.zeros(2, 3), torch.tensor(0.0)),
        )


def test_raw_one_shot_wrapper_preserves_augmented_contract() -> None:
    base = _FakeOneShot(num_auxiliary=2)
    bridge = _ScaleBridge()
    wrapped = wrap_raw_decision_acquisition(base, bridge)

    assert isinstance(wrapped, RawDecisionOneShotAcquisition)
    assert isinstance(wrapped, OneShotAcquisitionFunction)
    assert wrapped.get_augmented_q_batch_size(2) == 4

    raw = torch.arange(12, dtype=torch.double).reshape(1, 4, 3)
    value = wrapped(raw)
    assert torch.isfinite(value).all()
    assert base.last_x is not None
    torch.testing.assert_close(base.last_x, raw * 2.0)
    extracted = wrapped.extract_candidates(raw)
    torch.testing.assert_close(extracted, raw[..., :2, :])


def test_one_shot_initializer_expands_to_full_tree_with_interpoint_constraints() -> None:
    acqf = _FakeOneShot(num_auxiliary=2)
    generator = resolve_one_shot_ic_generator(acqf)
    assert generator is not None
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    equality = [
        (
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([1.0], dtype=torch.double),
            0.0,
        )
    ]
    initial = generator(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
        equality_constraints=equality,
    )

    assert initial is not None
    assert initial.shape == (2, 3, 2)  # actual q=1 + two auxiliary points.
    assert initial[:, 0, 1].abs().max().item() <= 1e-8


def test_real_qkg_optimizes_best_subset_with_actual_q_support_constraints() -> None:
    train_x = torch.tensor(
        [[0.10, 0.90], [0.35, 0.65], [0.65, 0.35], [0.90, 0.10]],
        dtype=torch.double,
    )
    train_y = torch.tensor([[0.2], [0.8], [1.0], [0.4]], dtype=torch.double)
    model = SingleTaskGP(train_x, train_y)
    acqf = qKnowledgeGradient(
        model=model,
        num_fantasies=2,
        current_value=torch.tensor(1.0, dtype=torch.double),
    )
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    config = OptimizeConfig(
        q=1,
        num_restarts=2,
        raw_samples=16,
        sequential=False,
        repair_config=CandidateRepairConfig(
            comp_idx=[0, 1],
            k=1,
            support_selection="best_subset",
            inequality_constraints=[
                ([0], [1.0], 0.05),
                ([1], [1.0], 0.05),
            ],
            inequality_sense="ge",
        ),
        optimizer_kwargs={
            "best_subset_strategy": "exact",
            "options": {"maxiter": 20, "batch_limit": 2},
        },
    )

    candidates, value = optimize_candidates(acqf, bounds, config)

    assert candidates.shape == (1, 2)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(torch.as_tensor(value)).all()
    active = candidates[0].abs() > 1e-6
    assert int(active.sum().item()) == 1
    assert float(candidates[0, active][0]) >= 0.05 - 1e-5
