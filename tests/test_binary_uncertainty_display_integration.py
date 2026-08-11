from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bochan.models.classification.binary.base.posterior import SimpleBernoulliPosterior
from bochan.serving.webapp.target_results import _display_predictions
from bochan.visualization.input_perturbation import prediction_mean_std
from tests.test_binary_epistemic_uncertainty import _EpistemicBinaryModel
from tests.test_hybrid_task_aware_posterior import _binary_hybrid


def test_simple_bernoulli_proxy_samples_stay_in_probability_domain() -> None:
    posterior = SimpleBernoulliPosterior(
        mean=torch.tensor([[0.5]], dtype=torch.double),
        variance=torch.tensor([[0.25]], dtype=torch.double),
    )
    sample_shape = torch.Size([2])
    base_samples = torch.tensor(
        [[[10.0]], [[-10.0]]],
        dtype=torch.double,
    )

    samples = posterior.rsample_from_base_samples(
        sample_shape=sample_shape,
        base_samples=base_samples,
    )

    assert torch.all(samples > 0.0)
    assert torch.all(samples < 1.0)


def test_web_binary_display_uses_epistemic_probability_std() -> None:
    model = _EpistemicBinaryModel()
    optimizer = SimpleNamespace(model=model)
    X = torch.tensor(
        [
            [0.50, 0.01],
            [0.50, 0.15],
        ],
        dtype=torch.double,
    )

    display, _ = _display_predictions(
        optimizer,
        X,
        target_columns=["feasible"],
        target_metadata={"feasible": {"internal_task": "binary"}},
        hybrid_model=False,
    )

    mean = display["feasible"]["mean"]
    std = display["feasible"]["std"]
    torch.testing.assert_close(mean, torch.tensor([0.5, 0.5], dtype=torch.double))
    assert std[1] > 10.0 * std[0]
    assert torch.all(std < 0.2)
    assert display["feasible"]["prediction_space"] == "probability"


def test_hybrid_web_binary_display_uses_probability_epistemic_variance() -> None:
    model = _binary_hybrid()
    optimizer = SimpleNamespace(model=model)
    X = torch.zeros(3, 2, dtype=torch.double)

    display, _ = _display_predictions(
        optimizer,
        X,
        target_columns=["property", "feasible"],
        target_metadata={
            "property": {"internal_task": "regression"},
            "feasible": {"internal_task": "binary"},
        },
        hybrid_model=True,
    )

    binary_mean = display["feasible"]["mean"]
    binary_std = display["feasible"]["std"]
    torch.testing.assert_close(
        binary_mean,
        torch.full((3,), 0.5, dtype=torch.double),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all(binary_std < 0.1)
    assert torch.all(binary_std < 0.5)
    assert display["feasible"]["prediction_space"] == "probability"


def test_hybrid_visualization_keeps_binary_epistemic_std() -> None:
    model = _binary_hybrid()
    X = torch.zeros(3, 2, dtype=torch.double)
    optimizer = SimpleNamespace(
        train_X=X,
        bundle=SimpleNamespace(
            model=model,
            model_config=None,
            metadata={},
        ),
        model_config=None,
    )

    mean, std = prediction_mean_std(optimizer, X)

    np.testing.assert_allclose(mean[:, 1], 0.5, atol=1e-6)
    assert np.all(std[:, 1] < 0.1)
    assert np.all(std[:, 1] < 0.5)
