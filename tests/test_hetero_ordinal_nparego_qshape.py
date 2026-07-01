import importlib
from types import SimpleNamespace


bo_module = importlib.import_module(
    "bochan.acquisition.ordinal.bayesian_optimization"
)


def test_hetero_ordinal_nparego_disables_early_objective_shape_check(monkeypatch):
    objective = SimpleNamespace(_verify_output_shape=True)
    acquisition = SimpleNamespace(objective=objective)

    monkeypatch.setattr(
        bo_module,
        "_qHeteroMultiOutputOrdinalNParEGO",
        lambda *args, **kwargs: acquisition,
    )

    result = bo_module.qHeteroMultiOutputOrdinalNParEGO(model=object())

    assert result is acquisition
    assert objective._verify_output_shape is False
