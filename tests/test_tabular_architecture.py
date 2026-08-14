from __future__ import annotations

import ast
from pathlib import Path

import bochan.api as api
import bochan.composition as composition
import bochan.tabular as tabular
from bochan.composition import CompositionTransformer
from bochan.tabular.composition import CompositionAdapter
from bochan.tabular.observation import ObservationAdapter
from bochan.tabular.optimizer.candidates import CandidateService
from bochan.tabular.optimizer.core import TabularBayesianOptimizer
from bochan.tabular.optimizer.diagnostics import DiagnosticsService


def test_tabular_public_optimizer_has_one_canonical_entry_point() -> None:
    assert tabular.TabularBayesianOptimizer is TabularBayesianOptimizer
    assert TabularBayesianOptimizer.__module__ == "bochan.tabular.optimizer.core"


def test_tabular_root_contains_only_public_python_entry_point() -> None:
    root = Path(tabular.__file__).resolve().parent
    assert sorted(path.name for path in root.glob("*.py")) == ["__init__.py"]


def test_canonical_optimizer_has_no_functional_mixin_inheritance() -> None:
    assert TabularBayesianOptimizer.__mro__ == (TabularBayesianOptimizer, object)


def test_optimizer_uses_explicit_domain_components() -> None:
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x"],
        target_cols=["y"],
    )
    assert isinstance(optimizer.composition, CompositionAdapter)
    assert isinstance(optimizer.observation, ObservationAdapter)
    assert isinstance(optimizer.candidates, CandidateService)
    assert isinstance(optimizer.diagnostics, DiagnosticsService)


def test_lower_level_components_do_not_define_optimizer_facades() -> None:
    import bochan.tabular.composition.adapter as composition_adapter
    import bochan.tabular.observation.adapter as observation_adapter
    import bochan.tabular.optimizer.candidates as candidates
    import bochan.tabular.optimizer.diagnostics as diagnostics

    for module in (
        composition_adapter,
        observation_adapter,
        candidates,
        diagnostics,
    ):
        assert "TabularBayesianOptimizer" not in vars(module)


def test_core_composition_has_no_tabular_or_pandas_imports() -> None:
    root = Path(composition.__file__).resolve().parent
    forbidden = ("bochan.tabular", "bochan.serving", "pandas")
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        for module in imported:
            assert not module.startswith(forbidden), (
                f"{path} must not depend on {module}."
            )


def test_composition_transformer_is_core_domain_type() -> None:
    assert CompositionTransformer.__module__ == "bochan.composition.transformer"


def test_tabular_composition_has_no_legacy_domain_modules() -> None:
    root = Path(tabular.__file__).resolve().parent / "composition"
    for name in ("formula.py", "simplex.py", "descriptors.py", "search_space.py"):
        assert not (root / name).exists()


def test_tabular_import_keeps_core_candidate_identity() -> None:
    sentinel_name = "_bochan_" + "candidate_before_tabular_outputs"
    assert not hasattr(api.BayesianOptimizer, sentinel_name)
    assert api.BayesianOptimizer.candidate.__module__ == "bochan.api.optimizer"


def test_tabular_import_does_not_patch_core_prediction() -> None:
    sentinel_name = "_bochan_" + "predict_before_tabular_labels"
    assert not hasattr(api.BayesianOptimizer, sentinel_name)
