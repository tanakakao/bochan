from __future__ import annotations

import inspect

from bochan.acquisition.binary.bayesian_optimization import qMultiOutputBinaryNParEGO
from bochan.acquisition.multiclass.bayesian_optimization import qMultiOutputMulticlassNParEGO
from bochan.acquisition.ordinal.bayesian_optimization import qMultiOutputOrdinalNParEGO


def test_classification_nparego_signatures_preserve_context_and_constraints() -> None:
    for acquisition_cls in (
        qMultiOutputBinaryNParEGO,
        qMultiOutputOrdinalNParEGO,
        qMultiOutputMulticlassNParEGO,
    ):
        parameters = inspect.signature(acquisition_cls).parameters
        assert "model" in parameters
        assert "X_baseline" in parameters
        assert "ref_point" in parameters
        assert "constraints" in parameters
        assert "eta" in parameters
        assert "fat" in parameters
