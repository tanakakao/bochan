from pathlib import Path

from bochan.acquisition.multiclass.bayesian_optimization import multi_output


def test_multiclass_nehvi_baseline_patch_module_is_removed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    patch_module = (
        repo_root
        / "src"
        / "bochan"
        / "acquisition"
        / "multiclass"
        / "bayesian_optimization"
        / "nehvi_baseline.py"
    )

    assert not patch_module.exists()
    assert callable(multi_output._baseline_partitioning_from_model)
