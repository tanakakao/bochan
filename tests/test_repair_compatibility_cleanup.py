from __future__ import annotations

import ast
from pathlib import Path

import torch

from bochan.api.configs.base import CandidateRepairConfig, OptimizeConfig
from bochan.api.factory import _build_post_processing_func
from bochan.constraints.postprocess import make_grid_k_sparse_post_processing_func


def _constraint_import_levels(path: str) -> list[int]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return [
        node.level
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "constraints.k_sparse"
    ]


def test_candidate_repair_does_not_inspect_caller_frames() -> None:
    source = Path("src/bochan/constraints/postprocess.py").read_text(encoding="utf-8")

    assert "currentframe" not in source
    assert "import inspect" not in source
    assert "grid_base: Tensor | None = None" in source


def test_optimizer_constraint_imports_use_package_relative_path_only() -> None:
    paths = [
        "src/bochan/optim/gradient/botorch.py",
        "src/bochan/optim/gradient/torch.py",
        "src/bochan/optim/evolutionary/core.py",
    ]

    for path in paths:
        source = Path(path).read_text(encoding="utf-8")
        assert _constraint_import_levels(path) == [3]
        assert "from constraints.k_sparse" not in source


def test_direct_grid_repair_defaults_to_lower_bound_origin() -> None:
    bounds = torch.tensor([[0.25], [1.0]], dtype=torch.double)
    post_process = make_grid_k_sparse_post_processing_func(
        bounds=bounds,
        steps=torch.tensor([0.2], dtype=torch.double),
        numeric_indices=[0],
    )

    result = post_process(torch.tensor([[0.31]], dtype=torch.double))

    assert torch.allclose(result, torch.tensor([[0.25]], dtype=torch.double))


def test_high_level_repair_without_private_bounds_uses_zero_origin() -> None:
    bounds = torch.tensor([[0.25], [1.0]], dtype=torch.double)
    config = OptimizeConfig(
        repair_config=CandidateRepairConfig(
            steps=torch.tensor([0.2], dtype=torch.double),
            numeric_indices=[0],
        )
    )

    post_process = _build_post_processing_func(config, bounds)
    assert post_process is not None
    result = post_process(torch.tensor([[0.31]], dtype=torch.double))

    assert torch.allclose(result, torch.tensor([[0.4]], dtype=torch.double))


def test_high_level_repair_with_explicit_bounds_uses_lower_bound_origin() -> None:
    bounds = torch.tensor([[0.25], [1.0]], dtype=torch.double)
    config = OptimizeConfig(
        repair_config=CandidateRepairConfig(
            bounds=bounds,
            steps=torch.tensor([0.2], dtype=torch.double),
            numeric_indices=[0],
        )
    )

    post_process = _build_post_processing_func(config, bounds)
    assert post_process is not None
    result = post_process(torch.tensor([[0.31]], dtype=torch.double))

    assert torch.allclose(result, torch.tensor([[0.25]], dtype=torch.double))
