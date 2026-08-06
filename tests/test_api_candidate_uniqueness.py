from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from bochan.api import OptimizeConfig
from bochan.api.candidate_uniqueness import ensure_unique_candidates


class _PendingAwareAcquisition:
    def __init__(self) -> None:
        self.X_pending = None
        self.pending_calls: list[Any] = []

    def set_X_pending(self, X_pending=None) -> None:
        self.X_pending = X_pending
        self.pending_calls.append(X_pending)

    def __call__(self, X):
        return -((X - 0.25) ** 2).sum()


class _PendingUnsupportedAcquisition:
    def __call__(self, X):
        return -((X - 0.25) ** 2).sum()


def _config(**kwargs: Any) -> OptimizeConfig:
    values = {
        "q": 3,
        "num_restarts": 2,
        "raw_samples": 8,
        "sequential": False,
    }
    values.update(kwargs)
    return OptimizeConfig(**values)


def test_unique_batch_is_returned_without_refill() -> None:
    candidates = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double)
    acq_value = torch.tensor(1.0, dtype=torch.double)

    def unexpected_refill(**kwargs):
        raise AssertionError(f"refill must not run: {kwargs}")

    result = ensure_unique_candidates(
        acqf=_PendingAwareAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        config=_config(),
        candidates=candidates,
        acq_value=acq_value,
        optimize_once=unexpected_refill,
    )

    assert result[0] is candidates
    assert result[1] is acq_value


def test_duplicate_slots_use_same_acquisition_and_restart_pool() -> None:
    acquisition = _PendingAwareAcquisition()
    candidates = torch.tensor([[0.1], [0.1], [0.2]], dtype=torch.double)
    calls: list[OptimizeConfig] = []

    def optimize_once(*, acqf, bounds, config):
        assert acqf is acquisition
        assert torch.equal(
            bounds,
            torch.tensor([[0.0], [1.0]], dtype=torch.double),
        )
        calls.append(config)
        return (
            torch.tensor([[[0.1]], [[0.3]], [[0.4]]], dtype=torch.double),
            torch.tensor([3.0, 2.0, 1.0], dtype=torch.double),
        )

    result, _ = ensure_unique_candidates(
        acqf=acquisition,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        config=_config(),
        candidates=candidates,
        acq_value=torch.tensor(0.0, dtype=torch.double),
        optimize_once=optimize_once,
    )

    assert torch.equal(
        result,
        torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double),
    )
    assert len(calls) == 1
    assert calls[0].q == 1
    assert calls[0].sequential is False
    assert calls[0].return_best_only is False
    assert calls[0].num_restarts == 16
    assert calls[0].raw_samples == 256
    assert torch.equal(
        acquisition.pending_calls[0],
        torch.tensor([[0.1], [0.2]], dtype=torch.double),
    )
    assert acquisition.X_pending is None



def test_refill_removes_q_specific_optimizer_kwargs() -> None:
    candidates = torch.tensor([[0.1], [0.1], [0.2]], dtype=torch.double)
    captured: list[OptimizeConfig] = []

    def optimize_once(*, acqf, bounds, config):
        del acqf, bounds
        captured.append(config)
        return (
            torch.tensor([[[0.3]]], dtype=torch.double),
            torch.tensor([1.0], dtype=torch.double),
        )

    result, _ = ensure_unique_candidates(
        acqf=_PendingUnsupportedAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        config=_config(
            optimizer_kwargs={
                "batch_initial_conditions": torch.zeros(2, 3, 1),
                "acq_function_sequence": [object()],
                "return_full_tree": True,
                "options": {"maxiter": 20},
            }
        ),
        candidates=candidates,
        acq_value=torch.tensor(0.0, dtype=torch.double),
        optimize_once=optimize_once,
    )

    assert torch.equal(
        result,
        torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double),
    )
    assert captured[0].optimizer_kwargs == {"options": {"maxiter": 20}}

def test_restart_pool_works_without_native_pending_support() -> None:
    candidates = torch.tensor([[0.1], [0.1], [0.2]], dtype=torch.double)

    def optimize_once(**kwargs):
        del kwargs
        return (
            torch.tensor([[[0.1]], [[0.35]]], dtype=torch.double),
            torch.tensor([10.0, 1.0], dtype=torch.double),
        )

    result, _ = ensure_unique_candidates(
        acqf=_PendingUnsupportedAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        config=_config(),
        candidates=candidates,
        acq_value=torch.tensor(0.0, dtype=torch.double),
        optimize_once=optimize_once,
    )

    assert torch.equal(
        result,
        torch.tensor([[0.1], [0.2], [0.35]], dtype=torch.double),
    )


def test_restart_pool_prefers_highest_scoring_distinct_candidate() -> None:
    candidates = torch.tensor([[0.1], [0.1], [0.2]], dtype=torch.double)

    def optimize_once(**kwargs):
        del kwargs
        return (
            torch.tensor([[[0.4]], [[0.3]]], dtype=torch.double),
            torch.tensor([0.1, 0.9], dtype=torch.double),
        )

    result, _ = ensure_unique_candidates(
        acqf=_PendingUnsupportedAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        config=_config(),
        candidates=candidates,
        acq_value=torch.tensor(0.0, dtype=torch.double),
        optimize_once=optimize_once,
    )

    assert result[-1].item() == pytest.approx(0.3)


@pytest.mark.parametrize(
    "optimizer",
    ["nsgaii", "thompson_sampling", "llm_candidate_set"],
)
def test_native_batch_optimizers_keep_their_results(optimizer: str) -> None:
    candidates = torch.tensor([[0.1], [0.1], [0.2]], dtype=torch.double)

    def unexpected_refill(**kwargs):
        raise AssertionError(f"native batch refill must not run: {kwargs}")

    result, _ = ensure_unique_candidates(
        acqf=_PendingAwareAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        config=_config(optimizer=optimizer),
        candidates=candidates,
        acq_value=torch.tensor(0.0, dtype=torch.double),
        optimize_once=unexpected_refill,
    )

    assert result is candidates


def test_non_best_only_restart_output_is_not_reinterpreted_as_q_batch() -> None:
    candidates = torch.tensor(
        [[[0.1], [0.1], [0.2]], [[0.2], [0.3], [0.4]]],
        dtype=torch.double,
    )

    result, _ = ensure_unique_candidates(
        acqf=_PendingAwareAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        config=_config(return_best_only=False),
        candidates=candidates,
        acq_value=torch.tensor([1.0, 0.5], dtype=torch.double),
        optimize_once=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"refill must not run: {kwargs}")
        ),
    )

    assert result is candidates


def test_unresolved_duplicates_preserve_requested_candidate_count() -> None:
    candidates = torch.tensor([[0.1], [0.1], [0.2]], dtype=torch.double)

    def optimize_once(**kwargs):
        del kwargs
        return (
            torch.tensor([[[0.1]], [[0.2]]], dtype=torch.double),
            torch.tensor([2.0, 1.0], dtype=torch.double),
        )

    with pytest.warns(RuntimeWarning, match="duplicate slot"):
        result, _ = ensure_unique_candidates(
            acqf=_PendingUnsupportedAcquisition(),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            config=_config(duplicate_refill_attempts=2),
            candidates=candidates,
            acq_value=torch.tensor(0.0, dtype=torch.double),
            optimize_once=optimize_once,
        )

    assert result.shape == (3, 1)
    assert torch.unique(result, dim=0).shape[0] == 2


def test_candidate_uniqueness_defaults_are_validated() -> None:
    config = _config()

    assert config.ensure_unique_candidates is True
    assert config.duplicate_tolerance == pytest.approx(1e-10)
    assert config.duplicate_refill_attempts == 4
    assert config.duplicate_pool_restarts == 16

    with pytest.raises(ValueError, match="duplicate_tolerance"):
        _config(duplicate_tolerance=-1.0)
    with pytest.raises(ValueError, match="duplicate_refill_attempts"):
        _config(duplicate_refill_attempts=0)
    with pytest.raises(ValueError, match="duplicate_pool_restarts"):
        _config(duplicate_pool_restarts=0)


def test_candidate_fix_contains_no_web_monkey_patch_or_acquisition_wrapper() -> None:
    root = Path(__file__).parents[1]
    web_init = (root / "src" / "bochan" / "serving" / "webapp" / "__init__.py").read_text()
    optimizer_source = (root / "src" / "bochan" / "api" / "optimizer_api.py").read_text()
    uniqueness_source = (
        root / "src" / "bochan" / "api" / "candidate_uniqueness.py"
    ).read_text()

    assert "candidate_batch_diversity" not in web_init
    assert "TabularBayesianOptimizer.candidate =" not in uniqueness_source
    assert "_ExcludedCandidateAcquisition" not in uniqueness_source
    assert "AcquisitionFunction" not in uniqueness_source
    assert "ensure_unique_candidates" in optimizer_source
