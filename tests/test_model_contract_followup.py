from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "src" / "bochan" / "models"

PROJECTED_DISTRIBUTIONAL_PATHS = (
    MODELS_ROOT
    / "regression"
    / "beta"
    / "high_dim"
    / "decomposition.py",
    MODELS_ROOT
    / "regression"
    / "gamma"
    / "high_dim"
    / "decomposition.py",
    MODELS_ROOT
    / "regression"
    / "count"
    / "negative_binomial"
    / "high_dim"
    / "decomposition.py",
)
POISSON_PATH = (
    MODELS_ROOT
    / "regression"
    / "count"
    / "poisson"
    / "high_dim"
    / "decomposition.py"
)
MULTICLASS_PATH = (
    MODELS_ROOT / "classification" / "multiclass" / "high_dim" / "decomposition.py"
)
REMBO_PATHS = (*PROJECTED_DISTRIBUTIONAL_PATHS, POISSON_PATH, MULTICLASS_PATH)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _train_x() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.1, 0.2],
            [0.2, 0.3, 0.4],
            [0.4, 0.5, 0.6],
            [0.6, 0.7, 0.8],
            [0.8, 0.9, 1.0],
            [1.0, 0.8, 0.6],
        ],
        dtype=torch.double,
    )


def test_projected_conditioning_uses_canonical_num_inducing() -> None:
    offenders: list[str] = []
    for path in PROJECTED_DISTRIBUTIONAL_PATHS:
        source = path.read_text(encoding="utf-8")
        if "self.num_inducing_points" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders


def test_rembo_config_uses_internal_n_components_keyword() -> None:
    offenders: list[str] = []
    for path in REMBO_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "REMBOConfig":
                continue
            if any(keyword.arg == "latent_dim" for keyword in node.keywords):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders


def test_poisson_projected_base_drops_n_components_compatibility() -> None:
    source = POISSON_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(POISSON_PATH))
    checked = 0
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        if class_node.name not in {
            "_ContinuousProjectedPoissonModel",
            "_MixedProjectedPoissonModel",
        }:
            continue
        init = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        names = {
            arg.arg
            for arg in [*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs]
        }
        checked += 1
        assert "latent_dim" in names
        assert "n_components" not in names
    assert checked == 2
    assert "except TypeError" not in source


def test_projected_latent_dim_assignments_have_no_redundant_fallback() -> None:
    offenders: list[str] = []
    marker = "latent_dim if latent_dim is not None else latent_dim"
    for path in PROJECTED_DISTRIBUTIONAL_PATHS:
        if marker in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders


def test_negative_binomial_rembo_mixed_has_single_definition() -> None:
    path = PROJECTED_DISTRIBUTIONAL_PATHS[-1]
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "REMBONegativeBinomialMixedGPModel"
    ]
    assert len(definitions) == 1
    assert 'kwargs.get("n_components")' not in source


@pytest.mark.parametrize(
    ("model_path", "class_name", "train_y"),
    [
        (
            "bochan.models.regression.beta.high_dim",
            "REMBOBetaGPModel",
            torch.tensor(
                [0.15, 0.25, 0.35, 0.55, 0.75, 0.85], dtype=torch.double
            ),
        ),
        (
            "bochan.models.regression.gamma.high_dim",
            "REMBOGammaGPModel",
            torch.tensor([0.5, 0.8, 1.1, 1.5, 2.0, 2.5], dtype=torch.double),
        ),
        (
            "bochan.models.regression.count.negative_binomial.high_dim",
            "REMBONegativeBinomialGPModel",
            torch.tensor([0.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.double),
        ),
        (
            "bochan.models.regression.count.poisson.high_dim",
            "REMBOPoissonGPModel",
            torch.tensor([0.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.double),
        ),
        (
            "bochan.models.classification.multiclass.high_dim",
            "REMBOMulticlassClassificationGPModel",
            torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.double),
        ),
    ],
)
def test_rembo_models_build_with_canonical_public_latent_dim(
    model_path: str,
    class_name: str,
    train_y: torch.Tensor,
) -> None:
    cls = getattr(importlib.import_module(model_path), class_name)
    model = cls(
        train_X=_train_x(),
        train_Y=train_y,
        latent_dim=2,
        num_inducing=3,
    )
    assert model.latent_dim == 2


@pytest.mark.parametrize(
    ("model_path", "class_name", "train_y", "new_y"),
    [
        (
            "bochan.models.regression.beta.high_dim",
            "PCABetaGPModel",
            torch.tensor(
                [0.15, 0.25, 0.35, 0.55, 0.75, 0.85], dtype=torch.double
            ),
            torch.tensor([0.45], dtype=torch.double),
        ),
        (
            "bochan.models.regression.gamma.high_dim",
            "PCAGammaGPModel",
            torch.tensor([0.5, 0.8, 1.1, 1.5, 2.0, 2.5], dtype=torch.double),
            torch.tensor([1.3], dtype=torch.double),
        ),
        (
            "bochan.models.regression.count.negative_binomial.high_dim",
            "PCANegativeBinomialGPModel",
            torch.tensor([0.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.double),
            torch.tensor([1.0], dtype=torch.double),
        ),
    ],
)
def test_projected_conditioning_preserves_num_inducing(
    model_path: str,
    class_name: str,
    train_y: torch.Tensor,
    new_y: torch.Tensor,
) -> None:
    cls = getattr(importlib.import_module(model_path), class_name)
    model = cls(
        train_X=_train_x(),
        train_Y=train_y,
        latent_dim=2,
        num_inducing=3,
    )
    updated = model.condition_on_observations(
        torch.tensor([[0.35, 0.45, 0.55]], dtype=torch.double),
        new_y,
    )
    assert updated.num_inducing == model.num_inducing == 3


def test_poisson_rejects_removed_n_components_alias() -> None:
    from bochan.models.regression.count.poisson.high_dim import (
        PCAPoissonGPModel,
    )

    with pytest.raises(TypeError):
        PCAPoissonGPModel(
            train_X=_train_x(),
            train_Y=torch.tensor(
                [0.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.double
            ),
            n_components=2,
            num_inducing=3,
        )
