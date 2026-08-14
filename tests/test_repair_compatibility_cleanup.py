from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import torch

import bochan.api.factory as api_factory
from bochan.api.configs.base import CandidateRepairConfig, OptimizeConfig
from bochan.constraints.postprocess import make_grid_k_sparse_post_processing_func


def _source_tree(path: str) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _constraint_import_levels(path: str) -> list[int]:
    return [
        node.level
        for node in ast.walk(_source_tree(path))
        if isinstance(node, ast.ImportFrom)
        and node.module == "constraints.k_sparse"
    ]


def _build_post_processing(config: OptimizeConfig, bounds: torch.Tensor) -> Any:
    builder = vars(api_factory)["_build_post_processing_func"]
    return builder(config, bounds)


def test_candidate_repair_does_not_inspect_caller_frames() -> None:
    tree = _source_tree("src/bochan/constraints/postprocess.py")
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    repair_factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "make_grid_k_sparse_post_processing_func"
    )
    keyword_args = repair_factory.args.kwonlyargs
    keyword_defaults = repair_factory.args.kw_defaults
    grid_base_index = next(
        index for index, arg in enumerate(keyword_args) if arg.arg == "grid_base"
    )

    assert "inspect" not in imported_modules
    assert "inspect" not in imported_names
    assert "currentframe" not in names
    assert "currentframe" not in attributes
    assert isinstance(keyword_defaults[grid_base_index], ast.Constant)
    assert keyword_defaults[grid_base_index].value is None


def test_optimizer_constraint_imports_use_package_relative_path_only() -> None:
    paths = [
        "src/bochan/optim/gradient/botorch.py",
        "src/bochan/optim/gradient/torch.py",
        "src/bochan/optim/evolutionary/core.py",
    ]

    for path in paths:
        assert _constraint_import_levels(path) == [3]


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

    post_process = _build_post_processing(config, bounds)
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

    post_process = _build_post_processing(config, bounds)
    assert post_process is not None
    result = post_process(torch.tensor([[0.31]], dtype=torch.double))

    assert torch.allclose(result, torch.tensor([[0.25]], dtype=torch.double))
