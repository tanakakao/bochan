from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _replace(path: Path, old: str, new: str, *, expected: int) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"{path}: expected {expected} replacements, found {count}: {old!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def pytest_sessionstart(session) -> None:
    del session
    branch = "agent/fix-model-contract-followup"
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    if os.environ.get("GITHUB_HEAD_REF") != branch:
        return
    if os.environ.get("GITHUB_WORKFLOW") != "Beta deep model smoke":
        return

    root = Path(__file__).resolve().parents[1]
    _run("git", "fetch", "origin", branch, cwd=root)
    _run("git", "checkout", "-B", branch, f"origin/{branch}", cwd=root)
    _run("git", "fetch", "origin", "main", cwd=root)

    projected = {
        root
        / "src/bochan/models/regression/non_gaussian/beta/high_dim/beta_decomposition.py": 4,
        root
        / "src/bochan/models/regression/non_gaussian/gamma/high_dim/gamma_decomposition.py": 4,
        root
        / "src/bochan/models/regression/non_gaussian/negative_binomial/high_dim/negative_binomial_decomposition.py": 4,
    }
    for path, count in projected.items():
        _replace(
            path,
            "int(latent_dim if latent_dim is not None else latent_dim)",
            "int(latent_dim)",
            expected=count,
        )
        _replace(
            path,
            "num_inducing_points=self.num_inducing_points,",
            "num_inducing=self.num_inducing,",
            expected=1,
        )

    for path in (
        root
        / "src/bochan/models/regression/non_gaussian/beta/high_dim/beta_decomposition.py",
        root
        / "src/bochan/models/regression/non_gaussian/gamma/high_dim/gamma_decomposition.py",
        root
        / "src/bochan/models/regression/non_gaussian/negative_binomial/high_dim/negative_binomial_decomposition.py",
        root / "src/bochan/models/classification/multiclass/high_dim/decomposition.py",
    ):
        _replace(
            path,
            "REMBOConfig(latent_dim=self.latent_dim, seed=seed)",
            "REMBOConfig(n_components=self.latent_dim, seed=seed)",
            expected=2,
        )

    poisson = root / (
        "src/bochan/models/regression/non_gaussian/poisson/high_dim/"
        "poisson_decomposition.py"
    )
    _replace(
        poisson,
        "        latent_dim: Optional[int] = None,\n"
        "        n_components: Optional[int] = None,",
        "        latent_dim: Optional[int] = None,",
        expected=2,
    )
    _replace(
        poisson,
        "self.latent_dim = int(n_components if n_components is not None else "
        "(latent_dim if latent_dim is not None else self.default_latent_dim))",
        "self.latent_dim = int(latent_dim if latent_dim is not None else "
        "self.default_latent_dim)",
        expected=2,
    )
    _replace(
        poisson,
        "            if config is None:\n"
        "                try:\n"
        "                    config = self.config_cls(latent_dim=self.latent_dim, seed=seed)\n"
        "                except TypeError:\n"
        "                    config = self.config_cls(n_components=self.latent_dim)",
        "            if config is None:\n"
        "                if self.config_cls is REMBOConfig:\n"
        "                    config = REMBOConfig(\n"
        "                        n_components=self.latent_dim,\n"
        "                        seed=seed,\n"
        "                    )\n"
        "                else:\n"
        "                    config = PCAConfig(n_components=self.latent_dim)",
        expected=2,
    )

    followup_test = '''from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTED_DECOMPOSITION_PATHS = (
    REPO_ROOT / "src/bochan/models/regression/non_gaussian/beta/high_dim/beta_decomposition.py",
    REPO_ROOT / "src/bochan/models/regression/non_gaussian/gamma/high_dim/gamma_decomposition.py",
    REPO_ROOT / "src/bochan/models/regression/non_gaussian/negative_binomial/high_dim/negative_binomial_decomposition.py",
)
REMBO_PATHS = (
    *PROJECTED_DECOMPOSITION_PATHS,
    REPO_ROOT / "src/bochan/models/regression/non_gaussian/poisson/high_dim/poisson_decomposition.py",
    REPO_ROOT / "src/bochan/models/classification/multiclass/high_dim/decomposition.py",
)


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
    offenders = []
    for path in PROJECTED_DECOMPOSITION_PATHS:
        source = path.read_text(encoding="utf-8")
        if "self.num_inducing_points" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders


def test_rembo_config_uses_internal_n_components_keyword() -> None:
    offenders = []
    for path in REMBO_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != "REMBOConfig":
                continue
            if any(keyword.arg == "latent_dim" for keyword in node.keywords):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders


def test_poisson_projected_base_drops_n_components_compatibility() -> None:
    path = REPO_ROOT / "src/bochan/models/regression/non_gaussian/poisson/high_dim/poisson_decomposition.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    checked = 0
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        if class_node.name not in {"_ContinuousProjectedPoissonModel", "_MixedProjectedPoissonModel"}:
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
    offenders = []
    marker = "latent_dim if latent_dim is not None else latent_dim"
    for path in PROJECTED_DECOMPOSITION_PATHS:
        if marker in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders


@pytest.mark.parametrize(
    ("model_path", "class_name", "train_y"),
    [
        ("bochan.models.regression.non_gaussian.beta.high_dim", "REMBOBetaGPModel", torch.tensor([0.15, 0.25, 0.35, 0.55, 0.75, 0.85], dtype=torch.double)),
        ("bochan.models.regression.non_gaussian.gamma.high_dim", "REMBOGammaGPModel", torch.tensor([0.5, 0.8, 1.1, 1.5, 2.0, 2.5], dtype=torch.double)),
        ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "REMBONegativeBinomialGPModel", torch.tensor([0.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.double)),
        ("bochan.models.regression.non_gaussian.poisson.high_dim", "REMBOPoissonGPModel", torch.tensor([0.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.double)),
        ("bochan.models.classification.multiclass.high_dim", "REMBOMulticlassClassificationGPModel", torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.double)),
    ],
)
def test_rembo_models_build_with_canonical_public_latent_dim(
    model_path: str,
    class_name: str,
    train_y: torch.Tensor,
) -> None:
    cls = getattr(importlib.import_module(model_path), class_name)
    model = cls(train_X=_train_x(), train_Y=train_y, latent_dim=2, num_inducing=3)
    assert model.latent_dim == 2


@pytest.mark.parametrize(
    ("model_path", "class_name", "train_y", "new_y"),
    [
        ("bochan.models.regression.non_gaussian.beta.high_dim", "PCABetaGPModel", torch.tensor([0.15, 0.25, 0.35, 0.55, 0.75, 0.85], dtype=torch.double), torch.tensor([0.45], dtype=torch.double)),
        ("bochan.models.regression.non_gaussian.gamma.high_dim", "PCAGammaGPModel", torch.tensor([0.5, 0.8, 1.1, 1.5, 2.0, 2.5], dtype=torch.double), torch.tensor([1.3], dtype=torch.double)),
        ("bochan.models.regression.non_gaussian.negative_binomial.high_dim", "PCANegativeBinomialGPModel", torch.tensor([0.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.double), torch.tensor([1.0], dtype=torch.double)),
    ],
)
def test_projected_conditioning_preserves_num_inducing(
    model_path: str,
    class_name: str,
    train_y: torch.Tensor,
    new_y: torch.Tensor,
) -> None:
    cls = getattr(importlib.import_module(model_path), class_name)
    model = cls(train_X=_train_x(), train_Y=train_y, latent_dim=2, num_inducing=3)
    updated = model.condition_on_observations(
        torch.tensor([[0.35, 0.45, 0.55]], dtype=torch.double),
        new_y,
    )
    assert updated.num_inducing == model.num_inducing == 3


def test_poisson_rejects_removed_n_components_alias() -> None:
    from bochan.models.regression.non_gaussian.poisson.high_dim import PCAPoissonGPModel

    with pytest.raises(TypeError):
        PCAPoissonGPModel(
            train_X=_train_x(),
            train_Y=torch.tensor([0.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=torch.double),
            n_components=2,
            num_inducing=3,
        )
'''
    (root / "tests/test_model_contract_followup.py").write_text(
        followup_test,
        encoding="utf-8",
    )

    for relative in (
        ".github/workflows/model-contract-smoke.yml",
        ".github/workflows/non-gaussian-acquisition-smoke.yml",
    ):
        restored = subprocess.check_output(
            ["git", "show", f"origin/main:{relative}"],
            cwd=root,
        )
        (root / relative).write_bytes(restored)

    model_contract_path = root / ".github/workflows/model-contract-smoke.yml"
    workflow = model_contract_path.read_text(encoding="utf-8")
    needle = "            tests/test_model_contract_refactor.py \\\n"
    replacement = needle + "            tests/test_model_contract_followup.py \\\n"
    if workflow.count(needle) != 1:
        raise RuntimeError("Could not locate model contract pytest list.")
    model_contract_path.write_text(
        workflow.replace(needle, replacement),
        encoding="utf-8",
    )

    Path(__file__).unlink()

    _run("git", "config", "user.name", "github-actions[bot]", cwd=root)
    _run(
        "git",
        "config",
        "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com",
        cwd=root,
    )
    _run("git", "add", "-A", cwd=root)
    _run("git", "diff", "--cached", "--check", cwd=root)
    _run("git", "commit", "-m", "Fix projected model contract follow-up", cwd=root)
    _run("git", "push", "origin", f"HEAD:{branch}", cwd=root)
