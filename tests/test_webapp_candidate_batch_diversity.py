from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

_MODULE = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "src"
        / "bochan"
        / "serving"
        / "webapp"
        / "candidate_batch_diversity.py"
    )
)
ExcludedCandidateAcquisition = _MODULE["_ExcludedCandidateAcquisition"]
candidate_uniqueness_metadata = _MODULE["candidate_uniqueness_metadata"]
install_web_candidate_batch_diversity = _MODULE[
    "install_web_candidate_batch_diversity"
]
prepare_web_candidate_request = _MODULE["prepare_web_candidate_request"]
refill_duplicate_candidate_result = _MODULE[
    "_refill_duplicate_candidate_result"
]
resolve_web_candidate_sequential = _MODULE["resolve_web_candidate_sequential"]
web_candidate_context = _MODULE["_WEB_CANDIDATE_CONTEXT"]


@dataclass
class _OptConfig:
    q: int
    sequential: bool


@dataclass
class _DataContext:
    bounds: Any
    X_pending: Any = None


@dataclass
class _CandidateResult:
    candidates: Any
    acqf: Any
    acq_config: Any
    opt_config: _OptConfig
    data_context: _DataContext
    acq_value: Any = None


class _ConstantAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=torch.nn.Identity())
        self.pending_calls: list[Any] = []
        self.X_pending = None

    def set_X_pending(self, X_pending=None) -> None:
        self.X_pending = X_pending
        self.pending_calls.append(X_pending)

    def forward(self, X):
        return torch.ones(X.shape[:-2], dtype=X.dtype, device=X.device)


def test_web_q3_normal_search_preserves_joint_batch_selection() -> None:
    effective, strategy = resolve_web_candidate_sequential(
        q=3,
        search_method="normal",
        requested=False,
    )

    assert effective is False
    assert strategy == "joint_batch"


def test_web_q3_mixed_search_preserves_requested_sequential_selection() -> None:
    effective, strategy = resolve_web_candidate_sequential(
        q=3,
        search_method="normal",
        requested=True,
    )

    assert effective is True
    assert strategy == "sequential_pending"


@pytest.mark.parametrize(
    "method",
    ["nsgaii", "nsga2", "thompson_sampling", "optimize_thompson_sampling"],
)
def test_native_batch_search_methods_keep_their_own_batch_selection(
    method: str,
) -> None:
    effective, strategy = resolve_web_candidate_sequential(
        q=3,
        search_method=method,
        requested=True,
    )

    assert effective is False
    assert strategy == "native_batch"


def test_prepare_web_candidate_request_keeps_original_false_setting() -> None:
    request = SimpleNamespace(
        optimizer=SimpleNamespace(
            name="normal",
            q=3,
            num_restarts=4,
            raw_samples=16,
            sequential=False,
        )
    )

    prepared, metadata = prepare_web_candidate_request(request)

    assert prepared is request
    assert prepared.optimizer.sequential is False
    assert metadata == {
        "candidate_batch_strategy": "joint_batch",
        "candidate_sequential_requested": False,
        "candidate_sequential_effective": False,
    }


def test_generic_exclusion_penalty_does_not_require_native_pending_support() -> None:
    base = _ConstantAcquisition()
    wrapped = ExcludedCandidateAcquisition(
        base,
        excluded=torch.tensor([[0.5]], dtype=torch.double),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        radius=0.05,
        penalty_weight=1_000.0,
    )

    duplicate_score = wrapped(torch.tensor([[[0.5]]], dtype=torch.double))
    distant_score = wrapped(torch.tensor([[[0.9]]], dtype=torch.double))

    assert duplicate_score.item() < -900.0
    assert distant_score.item() == pytest.approx(1.0, abs=1e-8)


def test_duplicate_slots_are_refilled_after_joint_q_optimization() -> None:
    base_acqf = _ConstantAcquisition()
    initial = _CandidateResult(
        candidates=torch.tensor([[0.1], [0.1], [0.2]], dtype=torch.double),
        acqf=base_acqf,
        acq_config=SimpleNamespace(name="EI"),
        opt_config=_OptConfig(q=3, sequential=False),
        data_context=_DataContext(
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double)
        ),
    )
    calls: list[tuple[int, bool, Any]] = []

    def fake_optimize_refill_candidate(*, acqf, bounds, opt_config):
        calls.append((opt_config.q, opt_config.sequential, bounds))
        duplicate_score = acqf(torch.tensor([[[0.1]]], dtype=torch.double))
        alternative_score = acqf(torch.tensor([[[0.3]]], dtype=torch.double))
        assert alternative_score.item() > duplicate_score.item()
        return torch.tensor([[0.3]], dtype=torch.double), torch.tensor(1.0)

    function_globals = refill_duplicate_candidate_result.__globals__
    original_optimizer = function_globals["_optimize_refill_candidate"]
    function_globals["_optimize_refill_candidate"] = fake_optimize_refill_candidate
    state = {"search_method": "normal"}
    token = web_candidate_context.set(state)
    try:
        result = refill_duplicate_candidate_result(initial)
    finally:
        web_candidate_context.reset(token)
        function_globals["_optimize_refill_candidate"] = original_optimizer

    assert initial.opt_config.sequential is False
    assert calls[0][0:2] == (1, False)
    assert torch.equal(
        result.candidates,
        torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double),
    )
    assert state["candidate_initial_duplicate_count"] == 1
    assert state["candidate_refill_count"] == 1
    assert state["candidate_native_pending_used"] is True
    assert state["candidate_exclusion_penalty_used"] is True
    assert torch.equal(
        base_acqf.pending_calls[0],
        torch.tensor([[0.1], [0.2]], dtype=torch.double),
    )
    assert base_acqf.X_pending is None


def test_candidate_uniqueness_metadata_uses_final_encoded_values() -> None:
    metadata = candidate_uniqueness_metadata(
        {
            "candidates": [
                {"encoded_values": {"x": 0.1, "category": 1.0}},
                {"encoded_values": {"x": 0.1, "category": 1.0}},
                {"encoded_values": {"x": 0.2, "category": 1.0}},
            ]
        }
    )

    assert metadata == {
        "candidate_count": 3,
        "candidate_unique_count": 2,
        "candidate_duplicate_count": 1,
    }


def test_installed_adapter_preserves_false_and_reports_diagnostics() -> None:
    def original(request, store):
        del store
        return {
            "candidates": [
                {"encoded_values": {"x": 0.1}},
                {"encoded_values": {"x": 0.2}},
                {"encoded_values": {"x": 0.3}},
            ],
            "metadata": {
                "workflow_sequential": request.optimizer.sequential,
            },
        }

    workflows = SimpleNamespace(_run_regression_web_workflow=original)
    workflows_tabular = SimpleNamespace(run_regression_web_workflow=original)
    function_globals = install_web_candidate_batch_diversity.__globals__
    original_installer = function_globals["_install_tabular_candidate_refill"]
    function_globals["_install_tabular_candidate_refill"] = lambda: None
    try:
        install_web_candidate_batch_diversity(workflows, workflows_tabular)
    finally:
        function_globals["_install_tabular_candidate_refill"] = original_installer

    request = SimpleNamespace(
        optimizer=SimpleNamespace(name="normal", q=3, sequential=False)
    )
    result = workflows._run_regression_web_workflow(request, None)

    assert result["metadata"]["workflow_sequential"] is False
    assert result["metadata"]["candidate_batch_strategy"] == "joint_batch"
    assert result["metadata"]["candidate_duplicate_count"] == 0
    assert result["metadata"]["candidate_refill_count"] == 0
    assert result["metadata"]["candidate_native_pending_used"] is False
    assert result["metadata"]["candidate_exclusion_penalty_used"] is False
