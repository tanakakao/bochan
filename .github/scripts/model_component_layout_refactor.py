from __future__ import annotations

import re
import shutil
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "bochan"
MODELS_ROOT = SRC_ROOT / "models"


def _move(old: str, new: str) -> None:
    old_path = REPO_ROOT / old
    new_path = REPO_ROOT / new
    if not old_path.exists():
        raise FileNotFoundError(old_path)
    if new_path.exists():
        raise FileExistsError(new_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))


def _replace_text(path: Path, replacements: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def _write_ordinal_kernel() -> None:
    path = MODELS_ROOT / "ordinal" / "base" / "kernel.py"
    path.write_text(
        textwrap.dedent(
            '''\
            from __future__ import annotations

            from typing import Sequence

            from botorch.models.kernels.categorical import CategoricalKernel
            from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, RBFKernel, ScaleKernel


            def _normalize_dims(cat_dims: Sequence[int], d: int) -> list[int]:
                """Normalize possibly-negative categorical feature indices."""
                dims: list[int] = []
                for idx in cat_dims:
                    j = idx if idx >= 0 else d + idx
                    if j < 0 or j >= d:
                        raise ValueError(f"Invalid categorical dim {idx} for input dim {d}.")
                    dims.append(int(j))
                return sorted(set(dims))


            def _get_cont_dims(d: int, cat_dims: Sequence[int]) -> list[int]:
                """Return continuous feature indices complementary to ``cat_dims``."""
                cat_set = set(_normalize_dims(cat_dims, d))
                return [i for i in range(d) if i not in cat_set]


            def _make_cont_kernel(
                cont_dims: Sequence[int],
                kernel_name: str = "matern52",
            ) -> Kernel | None:
                """Build the continuous part of the ordinal mixed-input kernel."""
                cont_dims = list(cont_dims)
                if not cont_dims:
                    return None

                if kernel_name.lower() == "rbf":
                    return ScaleKernel(
                        RBFKernel(
                            ard_num_dims=len(cont_dims),
                            active_dims=tuple(cont_dims),
                        )
                    )
                if kernel_name.lower() == "matern52":
                    return ScaleKernel(
                        MaternKernel(
                            nu=2.5,
                            ard_num_dims=len(cont_dims),
                            active_dims=tuple(cont_dims),
                        )
                    )
                raise ValueError(f"Unknown continuous kernel: {kernel_name}")


            def _make_cat_kernel(cat_dims: Sequence[int]) -> Kernel | None:
                """Build the categorical part of the ordinal mixed-input kernel."""
                cat_dims = list(cat_dims)
                if not cat_dims:
                    return None
                return ScaleKernel(CategoricalKernel(active_dims=tuple(cat_dims)))


            def build_mixed_ordinal_kernel(
                d: int,
                cat_dims: Sequence[int],
                cont_kernel_name: str = "matern52",
            ) -> Kernel:
                """Build the canonical mixed kernel for ordinal GP models."""
                cat_dims = _normalize_dims(cat_dims, d)
                cont_dims = _get_cont_dims(d, cat_dims)

                if not cat_dims:
                    kernel = _make_cont_kernel(cont_dims, cont_kernel_name)
                    if kernel is None:
                        raise ValueError("Failed to build continuous kernel.")
                    return kernel
                if not cont_dims:
                    kernel = _make_cat_kernel(cat_dims)
                    if kernel is None:
                        raise ValueError("Failed to build categorical kernel.")
                    return kernel

                cont_kernel_1 = _make_cont_kernel(cont_dims, cont_kernel_name)
                cont_kernel_2 = _make_cont_kernel(cont_dims, cont_kernel_name)
                cat_kernel_1 = _make_cat_kernel(cat_dims)
                cat_kernel_2 = _make_cat_kernel(cat_dims)
                if any(
                    kernel is None
                    for kernel in (
                        cont_kernel_1,
                        cont_kernel_2,
                        cat_kernel_1,
                        cat_kernel_2,
                    )
                ):
                    raise RuntimeError("Failed to build mixed ordinal kernel.")

                return cont_kernel_1 + cat_kernel_1 + ProductKernel(
                    cont_kernel_2,
                    cat_kernel_2,
                )


            __all__ = ["build_mixed_ordinal_kernel"]
            '''
        ),
        encoding="utf-8",
    )


def _rewrite_binary_kernel() -> None:
    path = MODELS_ROOT / "classification" / "binary" / "base" / "kernel.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "def categorical_kernel(cat_dims, ord_dims, batch_shape):",
        "def build_binary_mixed_kernel(cat_dims, ord_dims, batch_shape):",
    )
    if "def build_binary_mixed_kernel" not in text:
        raise RuntimeError("Binary kernel function rename did not apply.")
    if "__all__" not in text:
        text = text.rstrip() + '\n\n__all__ = ["build_binary_mixed_kernel"]\n'
    path.write_text(text, encoding="utf-8")


def _collapse_ordinal_models() -> None:
    core_path = MODELS_ROOT / "ordinal" / "base" / "models_core.py"
    models_path = MODELS_ROOT / "ordinal" / "base" / "models.py"
    core = core_path.read_text(encoding="utf-8")

    core = core.replace(
        "from botorch.models.kernels.categorical import CategoricalKernel\n",
        "",
    )
    core = core.replace(
        "from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, RBFKernel, ScaleKernel",
        "from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel",
    )

    kernel_block = re.compile(
        r"\ndef _normalize_dims\(.*?\n\ndef _prepare_input_transform\(",
        flags=re.S,
    )
    replacement = textwrap.dedent(
        '''

        from bochan.models.ordinal.base.kernel import (
            _get_cont_dims,
            _make_cat_kernel,
            _make_cont_kernel,
            _normalize_dims,
            build_mixed_ordinal_kernel,
        )


        def _prepare_input_transform(
        '''
    ).rstrip("\n")
    core, count = kernel_block.subn(replacement, core, count=1)
    if count != 1:
        raise RuntimeError(f"Ordinal kernel block replacement count={count}.")

    latent_marker = "        latent_model = _OrdinalLatentGP(\n"
    if latent_marker not in core:
        raise RuntimeError("OrdinalGPModel latent model marker not found.")
    matern_default = textwrap.dedent(
        '''
                if covar_module is None:
                    covar_module = ScaleKernel(
                        MaternKernel(
                            nu=2.5,
                            ard_num_dims=raw_train_X.shape[-1],
                        )
                    ).to(device=raw_train_X.device, dtype=raw_train_X.dtype)

        '''
    )
    core = core.replace(latent_marker, matern_default + latent_marker, 1)

    core = core.rstrip() + textwrap.dedent(
        '''


        __all__ = [
            "OrdinalGPModel",
            "OrdinalMixedGPModel",
            "_BaseOrdinalGPModel",
            "_MixedOrdinalLatentGP",
            "_OrdinalLatentGP",
            "_canonicalize_inducing_points",
            "_check_categorical_columns_unchanged",
            "_expand_raw_X_to_match_transformed_q",
            "_get_cont_dims",
            "_infer_num_classes_from_train_Y",
            "_make_cat_kernel",
            "_make_cont_kernel",
            "_normalize_dims",
            "_prepare_input_transform",
            "_transform_tensor",
            "_transform_tensor_for_training",
            "build_mixed_ordinal_kernel",
        ]
        '''
    )
    models_path.write_text(core, encoding="utf-8")
    core_path.unlink()


def _update_references() -> None:
    replacements = {
        "bochan.posteriors.bernoulli": "bochan.models.classification.binary.base.posterior",
        "bochan.posteriors.classification_ensemble": "bochan.models.classification.common.posterior",
        "bochan.posteriors.ordinal_ensemble": "bochan.models.ordinal.posterior",
        "bochan.kernels.categorical_kernel": "bochan.models.classification.binary.base.kernel",
        "bochan.kernels.ordinal_kernel": "bochan.models.ordinal.base.kernel",
        "bochan.models.classification.multiclass.base.posteriors": "bochan.models.classification.multiclass.base.posterior",
        "src/bochan/posteriors/bernoulli.py": "src/bochan/models/classification/binary/base/posterior.py",
        "src/bochan/posteriors/classification_ensemble.py": "src/bochan/models/classification/common/posterior.py",
        "src/bochan/posteriors/ordinal_ensemble.py": "src/bochan/models/ordinal/posterior.py",
        "src/bochan/kernels/categorical_kernel.py": "src/bochan/models/classification/binary/base/kernel.py",
        "src/bochan/kernels/ordinal_kernel.py": "src/bochan/models/ordinal/base/kernel.py",
        "src/bochan/models/classification/multiclass/base/posteriors.py": "src/bochan/models/classification/multiclass/base/posterior.py",
        "models/ordinal/base/models_core.py": "models/ordinal/base/models.py",
        "from .posteriors import": "from .posterior import",
    }
    editable_suffixes = {".py", ".md", ".yml", ".yaml", ".toml", ".txt"}
    helper_names = {
        "model-component-layout-refactor.yml",
        "model-component-layout-refactor-pr.yml",
        "model_component_layout_refactor.py",
    }
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in editable_suffixes:
            continue
        if ".git" in path.parts or path.name in helper_names:
            continue
        _replace_text(path, replacements)

    binary_root = MODELS_ROOT / "classification" / "binary"
    for path in binary_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        updated = text.replace("categorical_kernel", "build_binary_mixed_kernel")
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _remove_empty_top_level_dirs() -> None:
    for stale_dir in (SRC_ROOT / "posteriors", SRC_ROOT / "kernels"):
        if not stale_dir.exists():
            continue
        leftovers = list(stale_dir.iterdir())
        if leftovers:
            raise RuntimeError(f"Unexpected files remain in {stale_dir}: {leftovers}")
        stale_dir.rmdir()


def _extend_layout_tests() -> None:
    path = REPO_ROOT / "tests" / "test_model_package_layout.py"
    source = path.read_text(encoding="utf-8")
    if "test_model_kernels_and_posteriors_are_family_owned" in source:
        return
    addition = textwrap.dedent(
        '''


        def test_model_kernels_and_posteriors_are_family_owned() -> None:
            assert not (SRC_ROOT / "kernels").exists()
            assert not (SRC_ROOT / "posteriors").exists()

            assert (
                MODELS_ROOT / "classification" / "binary" / "base" / "kernel.py"
            ).is_file()
            assert (
                MODELS_ROOT / "classification" / "binary" / "base" / "posterior.py"
            ).is_file()
            assert (
                MODELS_ROOT / "classification" / "common" / "posterior.py"
            ).is_file()
            assert (MODELS_ROOT / "ordinal" / "base" / "kernel.py").is_file()
            assert (MODELS_ROOT / "ordinal" / "posterior.py").is_file()
            assert (
                MODELS_ROOT / "classification" / "multiclass" / "base" / "posterior.py"
            ).is_file()


        def test_ordinal_base_has_one_canonical_model_and_kernel_implementation() -> None:
            ordinal_base = MODELS_ROOT / "ordinal" / "base"
            assert not (ordinal_base / "models_core.py").exists()

            models_source = (ordinal_base / "models.py").read_text(encoding="utf-8")
            kernel_source = (ordinal_base / "kernel.py").read_text(encoding="utf-8")

            assert "_OldOrdinalGPModel" not in models_source
            assert "def build_mixed_ordinal_kernel(" not in models_source
            assert kernel_source.count("def build_mixed_ordinal_kernel(") == 1
            assert models_source.count("class OrdinalGPModel(") == 1


        def test_removed_kernel_and_posterior_paths_are_not_referenced() -> None:
            forbidden = (
                "bochan.kernels",
                "bochan.posteriors",
                "src/bochan/kernels/",
                "src/bochan/posteriors/",
                "classification.multiclass.base.posteriors",
                "classification/multiclass/base/posteriors.py",
                "models.ordinal.base.models_core",
                "models/ordinal/base/models_core.py",
            )
            offenders: list[str] = []
            roots = (
                SRC_ROOT,
                REPO_ROOT / "tests",
                REPO_ROOT / "docs",
                REPO_ROOT / ".github",
            )
            helper_names = {
                "model-component-layout-refactor.yml",
                "model-component-layout-refactor-pr.yml",
                "model_component_layout_refactor.py",
            }
            for root in roots:
                if not root.exists():
                    continue
                for candidate in root.rglob("*"):
                    if candidate.name in helper_names:
                        continue
                    if (
                        not candidate.is_file()
                        or candidate.suffix not in {".py", ".md", ".yml", ".yaml"}
                    ):
                        continue
                    if candidate == Path(__file__):
                        continue
                    candidate_source = candidate.read_text(encoding="utf-8")
                    if any(token in candidate_source for token in forbidden):
                        offenders.append(str(candidate.relative_to(REPO_ROOT)))
            assert not offenders


        def test_binary_kernel_has_task_specific_builder_name() -> None:
            kernel_path = (
                MODELS_ROOT / "classification" / "binary" / "base" / "kernel.py"
            )
            source = kernel_path.read_text(encoding="utf-8")
            assert "def build_binary_mixed_kernel(" in source
            assert "def categorical_kernel(" not in source
        '''
    )
    path.write_text(source.rstrip() + addition + "\n", encoding="utf-8")


def _extend_agents() -> None:
    path = REPO_ROOT / "AGENTS.md"
    if not path.exists():
        return
    source = path.read_text(encoding="utf-8")
    if "### Model-owned kernels and posteriors" in source:
        return
    section = textwrap.dedent(
        '''


        ### Model-owned kernels and posteriors

        Kernel / posterior modules follow the same ownership boundary as the model family.
        Family-specific implementations must not be placed in top-level `bochan/kernels` or
        `bochan/posteriors` packages.

        Canonical locations include:

        ```text
        models/classification/binary/base/kernel.py
        models/classification/binary/base/posterior.py
        models/classification/common/posterior.py
        models/classification/multiclass/base/posterior.py
        models/ordinal/base/kernel.py
        models/ordinal/posterior.py
        models/hybrid/posterior.py
        models/components/deepgp_posterior.py
        ```

        A kernel or posterior belongs in `models/components/` only when it is genuinely
        shared by multiple model families. Do not retain forwarding modules at removed
        paths. Ordinal base models use a single canonical `base/models.py`; migration
        layers such as `models_core.py` are not allowed.
        '''
    )
    path.write_text(source.rstrip() + section + "\n", encoding="utf-8")


def main() -> None:
    legacy = SRC_ROOT / "posteriors" / "bernoulli.py"
    canonical = MODELS_ROOT / "classification" / "binary" / "base" / "posterior.py"
    if not legacy.exists():
        if canonical.exists():
            print("Canonical model component layout already applied.")
            return
        raise RuntimeError("Unexpected model component layout state.")

    _move(
        "src/bochan/posteriors/bernoulli.py",
        "src/bochan/models/classification/binary/base/posterior.py",
    )
    _move(
        "src/bochan/posteriors/classification_ensemble.py",
        "src/bochan/models/classification/common/posterior.py",
    )
    _move(
        "src/bochan/posteriors/ordinal_ensemble.py",
        "src/bochan/models/ordinal/posterior.py",
    )
    _move(
        "src/bochan/kernels/categorical_kernel.py",
        "src/bochan/models/classification/binary/base/kernel.py",
    )
    _move(
        "src/bochan/kernels/ordinal_kernel.py",
        "src/bochan/models/ordinal/base/kernel.py",
    )
    _move(
        "src/bochan/models/classification/multiclass/base/posteriors.py",
        "src/bochan/models/classification/multiclass/base/posterior.py",
    )

    _write_ordinal_kernel()
    _rewrite_binary_kernel()
    _collapse_ordinal_models()
    _update_references()
    _remove_empty_top_level_dirs()
    _extend_layout_tests()
    _extend_agents()

    if (SRC_ROOT / "posteriors").exists() or (SRC_ROOT / "kernels").exists():
        raise RuntimeError("Top-level kernel/posterior directories still exist.")
    if (MODELS_ROOT / "ordinal" / "base" / "models_core.py").exists():
        raise RuntimeError("Ordinal models_core.py still exists.")


if __name__ == "__main__":
    main()
