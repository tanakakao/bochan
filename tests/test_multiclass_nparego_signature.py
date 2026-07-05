import inspect

from bochan.acquisition.multiclass.bayesian_optimization import (
    qMultiOutputMulticlassNParEGO,
)


def test_multiclass_nparego_keeps_baseline_context() -> None:
    parameters = inspect.signature(qMultiOutputMulticlassNParEGO).parameters
    for name in ("model", "X_baseline", "ref_point", "constraints", "eta", "fat"):
        assert name in parameters
