from pathlib import Path

from bochan.acquisition.multiclass.active_learning import _multi_output_core as core
from bochan.acquisition.multiclass.active_learning.multi_output import (
    _DirectMultiOutputMulticlassAcqBase,
    qMultiOutputMulticlassBALD,
    qMultiOutputMulticlassGreedyJointBALD,
    qMultiOutputMulticlassJointBALD,
)


def test_active_learning_alignment_is_native() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    patch_module = (
        repo_root
        / "src"
        / "bochan"
        / "acquisition"
        / "multiclass"
        / "active_learning"
        / "alignment.py"
    )

    assert not patch_module.exists()
    assert not hasattr(
        core._DirectMultiOutputMulticlassAcqBase,
        "_bochan_original_coerce_explicit_multi_output_probs",
    )
    assert (
        _DirectMultiOutputMulticlassAcqBase._coerce_explicit_multi_output_probs
        is not core._DirectMultiOutputMulticlassAcqBase._coerce_explicit_multi_output_probs
    )


def test_native_alignment_preserves_public_bald_hierarchy() -> None:
    assert issubclass(qMultiOutputMulticlassBALD, _DirectMultiOutputMulticlassAcqBase)
    assert issubclass(qMultiOutputMulticlassJointBALD, qMultiOutputMulticlassBALD)
    assert issubclass(
        qMultiOutputMulticlassGreedyJointBALD,
        qMultiOutputMulticlassJointBALD,
    )
