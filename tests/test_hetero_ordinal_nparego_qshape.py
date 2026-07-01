import importlib
from types import SimpleNamespace


bo_module = importlib.import_module(
    "bochan.acquisition.ordinal.bayesian_optimization"
)


def test_hetero_ordinal_nparego_ignores_external_scalar_objective(monkeypatch):
    captured = {}
    internal_objective = SimpleNamespace(_verify_output_shape=True)
    acquisition = SimpleNamespace(objective=internal_objective)

    def fake_constructor(*args, **kwargs):
        captured.update(kwargs)
        return acquisition

    monkeypatch.setattr(
        bo_module,
        "_qHeteroMultiOutputOrdinalNParEGO",
        fake_constructor,
    )

    external_objective = object()
    result = bo_module.qHeteroMultiOutputOrdinalNParEGO(
        model=object(),
        objective=external_objective,
    )

    assert result is acquisition
    assert captured["objective"] is None
    assert internal_objective._verify_output_shape is True
