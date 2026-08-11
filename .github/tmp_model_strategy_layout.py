from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=ROOT, check=True)


def move(source: str, destination: str) -> None:
    src = ROOT / source
    dst = ROOT / destination
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        git("mv", source, destination)
        return
    if not dst.exists():
        raise RuntimeError(f"Neither source nor destination exists: {source} -> {destination}")


moves = [
    ("src/bochan/models/wide_multitask.py", "src/bochan/models/multitask/wide.py"),
    ("src/bochan/models/wide_multitask_variants.py", "src/bochan/models/multitask/task_feature.py"),
    ("src/bochan/models/wide_mixed_multitask.py", "src/bochan/models/multitask/mixed.py"),
    ("src/bochan/models/regression/_multitask.py", "src/bochan/models/multitask/validation.py"),
    ("src/bochan/models/components/kronecker_multitask.py", "src/bochan/models/multitask/kronecker.py"),
    (
        "src/bochan/models/classification/binary/base/multioutput.py",
        "src/bochan/models/multioutput/binary.py",
    ),
    (
        "src/bochan/models/classification/multiclass/base/multioutput.py",
        "src/bochan/models/multioutput/multiclass.py",
    ),
    ("src/bochan/models/ordinal/base/multioutput.py", "src/bochan/models/multioutput/ordinal.py"),
]
for source, destination in moves:
    move(source, destination)

replacements = [
    ("bochan.models.wide_multitask_variants", "bochan.models.multitask.task_feature"),
    ("bochan.models.wide_mixed_multitask", "bochan.models.multitask.mixed"),
    ("bochan.models.wide_multitask", "bochan.models.multitask.wide"),
    ("bochan.models.regression._multitask", "bochan.models.multitask.validation"),
    ("bochan.models.components.kronecker_multitask", "bochan.models.multitask.kronecker"),
    ("bochan.models.classification.binary.base.multioutput", "bochan.models.multioutput.binary"),
    ("bochan.models.classification.multiclass.base.multioutput", "bochan.models.multioutput.multiclass"),
    ("bochan.models.ordinal.base.multioutput", "bochan.models.multioutput.ordinal"),
    ("src/bochan/models/wide_multitask_variants.py", "src/bochan/models/multitask/task_feature.py"),
    ("src/bochan/models/wide_mixed_multitask.py", "src/bochan/models/multitask/mixed.py"),
    ("src/bochan/models/wide_multitask.py", "src/bochan/models/multitask/wide.py"),
    ("src/bochan/models/regression/_multitask.py", "src/bochan/models/multitask/validation.py"),
    ("src/bochan/models/components/kronecker_multitask.py", "src/bochan/models/multitask/kronecker.py"),
    (
        "src/bochan/models/classification/binary/base/multioutput.py",
        "src/bochan/models/multioutput/binary.py",
    ),
    (
        "src/bochan/models/classification/multiclass/base/multioutput.py",
        "src/bochan/models/multioutput/multiclass.py",
    ),
    ("src/bochan/models/ordinal/base/multioutput.py", "src/bochan/models/multioutput/ordinal.py"),
]

text_suffixes = {".py", ".md", ".yml", ".yaml", ".toml", ".txt"}
skip_paths = {
    ROOT / ".github" / "tmp_model_strategy_layout.py",
    ROOT / ".github" / "workflows" / "tmp_model_strategy_layout.yml",
}
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts or path in skip_paths:
        continue
    if path.suffix not in text_suffixes and path.name != "AGENTS.md":
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")

# Imports inside files whose relative package depth changed.
task_feature = ROOT / "src/bochan/models/multitask/task_feature.py"
text = task_feature.read_text(encoding="utf-8")
text = text.replace("from .wide_multitask import", "from .wide import")
task_feature.write_text(text, encoding="utf-8")

mixed = ROOT / "src/bochan/models/multitask/mixed.py"
text = mixed.read_text(encoding="utf-8")
text = text.replace(
    "from .wide_multitask_variants import WideMultiTaskGP",
    "from .task_feature import WideMultiTaskGP",
)
mixed.write_text(text, encoding="utf-8")

# Keep shallow family exports, but make the canonical implementation ownership
# bochan.models.multioutput.* rather than family-local files.
family_dirs = {
    ROOT / "src/bochan/models/classification/binary/base": "bochan.models.multioutput.binary",
    ROOT / "src/bochan/models/classification/multiclass/base": "bochan.models.multioutput.multiclass",
    ROOT / "src/bochan/models/ordinal/base": "bochan.models.multioutput.ordinal",
}
for directory, module in family_dirs.items():
    for path in directory.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        updated = text.replace("from .multioutput import", f"from {module} import")
        if updated != text:
            path.write_text(updated, encoding="utf-8")

# Wide posterior event shape belongs to the posterior itself, not a runtime patch.
wide = ROOT / "src/bochan/models/multitask/wide.py"
text = wide.read_text(encoding="utf-8")
needle = (
    "    @property\n"
    "    def dtype(self) -> torch.dtype:\n"
    "        return self.posterior.dtype\n\n"
    "    @property\n"
    "    def base_sample_shape(self) -> torch.Size:\n"
)
replacement = (
    "    @property\n"
    "    def dtype(self) -> torch.dtype:\n"
    "        return self.posterior.dtype\n\n"
    "    @property\n"
    "    def event_shape(self) -> torch.Size:\n"
    "        mean = self.mean\n"
    "        trailing = 2 if self.scalar_task_values else 3\n"
    "        return torch.Size(mean.shape[-trailing:])\n\n"
    "    @property\n"
    "    def base_sample_shape(self) -> torch.Size:\n"
)
if "def event_shape(self) -> torch.Size:" not in text:
    if needle not in text:
        raise RuntimeError("Could not locate _WidePosterior dtype/base_sample_shape block.")
    text = text.replace(needle, replacement, 1)
wide.write_text(text, encoding="utf-8")

ordinal_acq = ROOT / "src/bochan/acquisition/ordinal_multitask.py"
text = ordinal_acq.read_text(encoding="utf-8")
patch_block = (
    "    from bochan.acquisition.wide_posterior_events import (\n"
    "        apply_wide_posterior_events,\n"
    "    )\n\n"
    "    apply_wide_posterior_events()\n\n"
)
if patch_block in text:
    text = text.replace(patch_block, "", 1)
text = text.replace(
    '"""Install wide posterior, likelihood, and task-proxy support.',
    '"""Install likelihood and task-proxy support.',
    1,
)
ordinal_acq.write_text(text, encoding="utf-8")

wide_patch = ROOT / "src/bochan/acquisition/wide_posterior_events.py"
if wide_patch.exists():
    git("rm", str(wide_patch.relative_to(ROOT)))

# Fix a pre-existing duplicate decorator while relocating the module.
ordinal_multioutput = ROOT / "src/bochan/models/multioutput/ordinal.py"
text = ordinal_multioutput.read_text(encoding="utf-8")
text = text.replace(
    "    @staticmethod\n    @staticmethod\n    def _get_submodel_train_input_raw",
    "    @staticmethod\n    def _get_submodel_train_input_raw",
    1,
)
ordinal_multioutput.write_text(text, encoding="utf-8")

multitask_init = ROOT / "src/bochan/models/multitask/__init__.py"
multitask_init.write_text(
    '''"""Shared infrastructure for correlated multi-task models."""

from .kronecker import (
    BlockDesignVariationalELBO,
    LatentKroneckerMultiTaskGP,
    canonicalize_block_design_targets,
    canonicalize_shared_inducing_points,
)
from .mixed import WideMixedMultiTaskGP
from .task_feature import (
    PerturbationAwareStratifiedStandardize,
    PerturbationAwareWidePosterior,
    TaskFeatureInputTransform,
    WideMultiTaskBinaryClassificationGPModel,
    WideMultiTaskGP,
    WideMultiTaskMulticlassClassificationGPModel,
    WideMultiTaskOrdinalGPModel,
    wide_to_long,
)
from .validation import (
    long_to_sparse_wide,
    validate_complete_block,
    validate_long_multitask_data,
)

__all__ = [
    "BlockDesignVariationalELBO",
    "LatentKroneckerMultiTaskGP",
    "PerturbationAwareStratifiedStandardize",
    "PerturbationAwareWidePosterior",
    "TaskFeatureInputTransform",
    "WideMixedMultiTaskGP",
    "WideMultiTaskBinaryClassificationGPModel",
    "WideMultiTaskGP",
    "WideMultiTaskMulticlassClassificationGPModel",
    "WideMultiTaskOrdinalGPModel",
    "canonicalize_block_design_targets",
    "canonicalize_shared_inducing_points",
    "long_to_sparse_wide",
    "validate_complete_block",
    "validate_long_multitask_data",
    "wide_to_long",
]
''',
    encoding="utf-8",
)

multioutput_init = ROOT / "src/bochan/models/multioutput/__init__.py"
multioutput_init.write_text(
    '''"""Independent multi-output model aggregators.

Canonical implementations live in ``binary``, ``multiclass``, and ``ordinal``.
Unlike ``bochan.models.multitask``, these wrappers combine independently fitted
submodels and do not introduce learned task covariance.
"""
''',
    encoding="utf-8",
)

multifidelity_init = ROOT / "src/bochan/models/multifidelity/__init__.py"
multifidelity_init.write_text(
    '''"""Cross-family extension point for shared multi-fidelity infrastructure.

Likelihood-specific multi-fidelity models remain owned by their regression or
classification family. Shared fidelity-axis transforms, adapters, and validation
should be placed here as they are introduced.
"""
''',
    encoding="utf-8",
)

architecture = ROOT / "src/bochan/models/ARCHITECTURE.md"
architecture.write_text(
    '''# Model package ownership

Model code is organized along two axes: **model family** and **cross-cutting strategy**.

## Family-owned concrete models

Likelihood- or task-specific implementations stay under their owning family, for example:

- `regression/gaussian`, `regression/beta`, `regression/gamma`, `regression/count`
- `classification/binary`, `classification/multiclass`
- `ordinal`

Mixed variants that are specific to one concrete model remain with that family.

## Cross-cutting strategy packages

- `multitask/`: correlated outputs/tasks, task-feature adapters, shared ICM/Kronecker infrastructure, and validation.
- `multioutput/`: wrappers that aggregate independently fitted single-output models.
- `multifidelity/`: shared fidelity-axis abstractions and adapters. Concrete Gaussian/Bernoulli/etc. multi-fidelity models remain family-owned unless their implementation is genuinely cross-family.

This distinction is intentional: **multi-output does not imply multi-task correlation**.

## Adding a new strategy

1. Put family-independent mechanics in a dedicated top-level strategy package.
2. Keep likelihood-specific concrete models under the family tree.
3. Depend from family code toward shared strategy code; adapters may explicitly wrap family models when that is their role.
4. Do not add forwarding modules at removed paths. Update imports and the model registry directly.
5. Keep public data-shape contracts documented at the strategy boundary.

This layout lets future multi-fidelity, transfer-learning, and related strategies grow without adding more one-off modules directly under `models/`.
''',
    encoding="utf-8",
)

layout_test = ROOT / "tests/test_model_strategy_package_layout.py"
layout_test.write_text(
    '''from pathlib import Path


def test_cross_cutting_model_strategy_layout() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "bochan" / "models"

    expected = [
        root / "multitask" / "wide.py",
        root / "multitask" / "task_feature.py",
        root / "multitask" / "mixed.py",
        root / "multitask" / "validation.py",
        root / "multitask" / "kronecker.py",
        root / "multioutput" / "binary.py",
        root / "multioutput" / "multiclass.py",
        root / "multioutput" / "ordinal.py",
        root / "multifidelity" / "__init__.py",
    ]
    removed = [
        root / "wide_multitask.py",
        root / "wide_multitask_variants.py",
        root / "wide_mixed_multitask.py",
        root / "regression" / "_multitask.py",
        root / "components" / "kronecker_multitask.py",
        root / "classification" / "binary" / "base" / "multioutput.py",
        root / "classification" / "multiclass" / "base" / "multioutput.py",
        root / "ordinal" / "base" / "multioutput.py",
    ]

    assert all(path.is_file() for path in expected)
    assert not any(path.exists() for path in removed)
''',
    encoding="utf-8",
)

forbidden = [old for old, _ in replacements]
forbidden.extend(
    [
        "from .wide_multitask import",
        "from .wide_multitask_variants import",
        "bochan.acquisition.wide_posterior_events",
    ]
)
failures: list[str] = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts or path in skip_paths:
        continue
    if path.suffix not in text_suffixes and path.name != "AGENTS.md":
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    for token in forbidden:
        if token in text:
            failures.append(f"{path.relative_to(ROOT)}: {token}")
if failures:
    raise RuntimeError("Stale model-layout references remain:\n" + "\n".join(failures))

# The temporary automation must not appear in the final PR diff.
for temporary in skip_paths:
    if temporary.exists():
        temporary.unlink()
