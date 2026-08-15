"""Architecture guards for pre-release model API cleanup."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_input_transform_config_has_no_hidden_metadata_tunnel() -> None:
    """Input-transform configuration is forwarded explicitly, not through bounds metadata."""
    config_source = _read("src/bochan/api/configs/base.py")
    transform_source = _read("src/bochan/models/transforms/input.py")
    build_source = _read("src/bochan/api/modeling/build.py")

    assert "__bochan_input_transform_config__" not in config_source
    assert "__bochan_input_transform_config__" not in transform_source
    assert "normalize=tf_config.normalize" in build_source


def test_beta_models_do_not_expose_init_concentration() -> None:
    """The Beta family has one public precision spelling: concentration."""
    beta_root = REPO_ROOT / "src" / "bochan" / "models" / "regression" / "beta"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in beta_root.rglob("*.py")
        if "init_concentration" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_non_gaussian_conditioning_uses_canonical_num_inducing() -> None:
    """Rebuilt non-Gaussian models must use their canonical num_inducing argument."""
    paths = [
        "src/bochan/models/regression/beta/base/aligned.py",
        "src/bochan/models/regression/beta/base/multitask.py",
        "src/bochan/models/regression/gamma/base/aligned.py",
        "src/bochan/models/regression/gamma/base/multitask.py",
        "src/bochan/models/regression/count/poisson/base/aligned.py",
        "src/bochan/models/regression/count/poisson/base/multitask.py",
        "src/bochan/models/regression/count/negative_binomial/base/aligned.py",
        "src/bochan/models/regression/count/negative_binomial/base/multitask.py",
    ]
    offenders = [
        path
        for path in paths
        if "num_inducing_points=self.num_inducing_points" in _read(path)
    ]
    assert offenders == []


def test_pfns4bo_patch_is_scoped_to_checkpoint_loading() -> None:
    """PFNs4BO checkpoint compatibility must not mutate torch at package import time."""
    package_source = _read("src/bochan/models/regression/foundation/__init__.py")
    loader_source = _read("src/bochan/models/regression/foundation/pfn.py")

    assert "apply_pfns4bo_torch_compat()" not in package_source
    assert "apply_pfns4bo_torch_compat()" in loader_source
    assert "weights_only=False" in loader_source
    assert "except TypeError" not in loader_source
