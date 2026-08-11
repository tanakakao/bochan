from pathlib import Path


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


def test_canonical_strategy_modules_import_directly() -> None:
    from bochan.models.multioutput.binary import MultiOutputBinaryClassificationModel
    from bochan.models.multioutput.multiclass import (
        MultiOutputMulticlassClassificationModel,
    )
    from bochan.models.multioutput.ordinal import MultiOutputOrdinalModel
    from bochan.models.multitask.task_feature import WideMultiTaskGP

    assert MultiOutputBinaryClassificationModel.__name__
    assert MultiOutputMulticlassClassificationModel.__name__
    assert MultiOutputOrdinalModel.__name__
    assert WideMultiTaskGP.__name__
