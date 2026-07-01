import importlib

import torch


bo_module = importlib.import_module(
    "bochan.acquisition.ordinal.bayesian_optimization"
)


def test_integer_y_baseline_is_recomputed(monkeypatch):
    captured = {}

    def fake_constructor(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        bo_module,
        "_qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
        fake_constructor,
    )

    bo_module.qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
        model=object(),
        ref_point=torch.tensor([0.0, 0.0]),
        X_baseline=torch.zeros(3, 2),
        Y_baseline=torch.tensor([[0, 1], [1, 2], [2, 0]]),
    )

    assert captured["Y_baseline"] is None


def test_float_y_baseline_is_preserved(monkeypatch):
    captured = {}
    utility_baseline = torch.tensor(
        [[0.1, 0.3], [0.4, 0.8]],
        dtype=torch.double,
    )

    def fake_constructor(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        bo_module,
        "_qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
        fake_constructor,
    )

    bo_module.qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
        model=object(),
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
        X_baseline=torch.zeros(2, 2, dtype=torch.double),
        Y_baseline=utility_baseline,
    )

    assert captured["Y_baseline"] is utility_baseline
