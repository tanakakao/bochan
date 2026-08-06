from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bochan.visualization import show_1dplot_from_optimizer
from tests.test_binary_epistemic_uncertainty import _EpistemicBinaryModel


class _BinaryOptimizer:
    def __init__(self) -> None:
        self.model = _EpistemicBinaryModel(default_std=0.30)
        self.train_X = torch.tensor(
            [[0.02, 0.30], [0.98, 0.30]],
            dtype=torch.double,
        )
        self.train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        self.bounds = torch.tensor(
            [[0.02, 0.30], [0.98, 0.30]],
            dtype=torch.double,
        )
        self.model_config = SimpleNamespace(task_type="binary")
        self.bundle = SimpleNamespace(
            model=self.model,
            task_type="binary",
            metadata={
                "feature_cols": ["probability", "spread"],
                "target_cols": ["class"],
            },
            cat_dims=[],
        )


def test_binary_1d_uncertainty_is_bounded_to_probability_domain() -> None:
    optimizer = _BinaryOptimizer()
    candidate_result = SimpleNamespace(
        candidates=torch.tensor(
            [[0.02, 0.30], [0.98, 0.30]],
            dtype=torch.double,
        ),
        acq_value=None,
    )

    figure = show_1dplot_from_optimizer(
        optimizer,
        "probability",
        "class",
        feature_cols=["probability", "spread"],
        target_cols=["class"],
        value_dict={"spread": 0.30},
        candidate_result=candidate_result,
        n=21,
    )

    assert list(figure.layout.yaxis.range) == [0.0, 1.0]
    assert any(
        trace.name == "モデル不確実性 ±1σ（確率範囲内）"
        for trace in figure.data
    )

    for trace in figure.data:
        try:
            values = np.asarray(trace.y, dtype=float)
        except (TypeError, ValueError):
            continue
        finite = values[np.isfinite(values)]
        if finite.size:
            assert finite.min() >= 0.0
            assert finite.max() <= 1.0

    candidate_trace = next(trace for trace in figure.data if trace.name == "候補点")
    candidate_mean = np.asarray(candidate_trace.y, dtype=float)
    plus = np.asarray(candidate_trace.error_y.array, dtype=float)
    minus = np.asarray(candidate_trace.error_y.arrayminus, dtype=float)
    assert candidate_trace.error_y.symmetric is False
    assert np.all(candidate_mean + plus <= 1.0 + 1e-12)
    assert np.all(candidate_mean - minus >= -1e-12)
