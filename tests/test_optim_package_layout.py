from __future__ import annotations

from pathlib import Path

import bochan.optim as optim
from bochan.optim import evolutionary, gradient, llm, nsgaii, thompson


def test_optim_flat_modules_are_removed() -> None:
    root = Path(optim.__file__).resolve().parent
    removed = [
        "evo.py",
        "standard.py",
        "torch_opt.py",
        "torch_multitask.py",
        "nsgaii.py",
        "nsgaii_adapter.py",
        "nsgaii_constraints.py",
        "nsgaii_diversity.py",
        "nsgaii_outputs.py",
        "nsgaii_strategy.py",
        "thompson_sampling.py",
        "thompson_sampling_adapter.py",
        "llm.py",
    ]
    assert all(not (root / name).exists() for name in removed)


def test_optim_public_exports_use_algorithm_packages() -> None:
    assert optim.optimize_acqf_k_sparse is gradient.optimize_acqf_k_sparse
    assert optim.optimize_acqf_torch is gradient.optimize_acqf_torch
    assert optim.optimize_acqf_evo is evolutionary.optimize_acqf_evo
    assert optim.optimize_acqf_nsgaii is nsgaii.optimize_acqf_nsgaii
    assert optim.optimize_thompson_sampling is thompson.optimize_thompson_sampling
    assert optim.optimize_acqf_llm_candidate_set is llm.optimize_acqf_llm_candidate_set
    assert nsgaii.build_nsgaii_strategy is not None
