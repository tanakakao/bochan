from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from bochan.models.classification.multiclass.deep.deepgp import (
    MulticlassDeepGPModel,
)
from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec
from bochan.models.regression.gaussian.deep.deepgp import DeepGPModel
from bochan.tabular import TabularBayesianOptimizer


class _RegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_inputs = (torch.zeros(6, 2, dtype=torch.double),)
        self.train_targets = torch.zeros(6, 1, dtype=torch.double)

    def posterior(self, X: torch.Tensor, **kwargs):
        del kwargs
        mean = X[..., :1]
        return SimpleNamespace(mean=mean, variance=torch.ones_like(mean))


class _SampledClassModel(nn.Module):
    def __init__(self, probabilities: torch.Tensor) -> None:
        super().__init__()
        self.probabilities = probabilities
        self.train_inputs = (torch.zeros(6, 2, dtype=torch.double),)
        self.train_targets = torch.zeros(6, dtype=torch.long)

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        batch_q = X.shape[:-1]
        probs = self.probabilities.to(device=X.device, dtype=X.dtype)
        return probs.reshape(
            probs.shape[0],
            *((1,) * len(batch_q)),
            *probs.shape[1:],
        ).expand(probs.shape[0], *batch_q, *probs.shape[1:])


@pytest.mark.parametrize("task_type", ["ordinal", "multiclass"])
def test_hybrid_class_probabilities_keep_q_with_leading_sample_axis(
    task_type: str,
) -> None:
    sampled_probs = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ],
        dtype=torch.double,
    )
    class_model = _SampledClassModel(sampled_probs)
    hybrid = HybridMultiOutputModel(
        [
            OutputSpec(
                name="property",
                task_type="regression",
                model=_RegressionModel(),
            ),
            OutputSpec(
                name="quality",
                task_type=task_type,
                model=class_model,
                utility_values=[0.0, 1.0, 2.0],
            ),
        ]
    )
    X = torch.rand(4, 2, 2, dtype=torch.double)

    posterior = hybrid.posterior(X)

    expected_utility = (
        sampled_probs * torch.arange(3, dtype=torch.double)
    ).sum(dim=-1).mean()
    assert posterior.mean.shape == torch.Size([4, 2, 2])
    assert torch.allclose(
        posterior.mean[..., 1],
        torch.full((4, 2), expected_utility, dtype=torch.double),
    )


def test_hybrid_class_probabilities_select_internal_output_axis() -> None:
    # sample x outputs x classes; output_index=1 must be selected only after the
    # public batch/q suffix has been identified.
    sampled_probs = torch.tensor(
        [
            [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]],
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
        ],
        dtype=torch.double,
    )
    class_model = _SampledClassModel(sampled_probs)
    hybrid = HybridMultiOutputModel(
        [
            OutputSpec(
                name="property",
                task_type="regression",
                model=_RegressionModel(),
            ),
            OutputSpec(
                name="quality",
                task_type="multiclass",
                model=class_model,
                output_index=1,
                utility_values=[0.0, 1.0, 2.0],
            ),
        ]
    )
    X = torch.rand(3, 2, 2, dtype=torch.double)

    posterior = hybrid.posterior(X)

    selected = sampled_probs[:, 1, :]
    expected_utility = (
        selected * torch.arange(3, dtype=torch.double)
    ).sum(dim=-1).mean()
    assert posterior.mean.shape == torch.Size([3, 2, 2])
    assert torch.allclose(
        posterior.mean[..., 1],
        torch.full((3, 2), expected_utility, dtype=torch.double),
    )


def test_hybrid_regression_multiclass_deepgp_posterior_preserves_batch_and_q() -> None:
    torch.manual_seed(0)
    train_X = torch.rand(12, 2, dtype=torch.double)
    regression = DeepGPModel(
        train_X=train_X,
        train_Y=(train_X[:, :1] + train_X[:, 1:2]),
        input_transform=None,
        outcome_transform=None,
        list_hidden_dims=[2],
        num_inducing=4,
    )
    multiclass = MulticlassDeepGPModel(
        train_X=train_X,
        train_Y=torch.arange(12) % 3,
        hidden_dim=2,
        list_hidden_dims=[2],
        num_inducing=4,
    )
    hybrid = HybridMultiOutputModel(
        [
            OutputSpec(
                name="property",
                task_type="regression",
                model=regression,
            ),
            OutputSpec(
                name="quality",
                task_type="multiclass",
                model=multiclass,
                utility_values=[0.0, 1.0, 2.0],
            ),
        ]
    ).eval()
    X = torch.rand(4, 2, 2, dtype=torch.double)

    posterior = hybrid.posterior(X)

    assert posterior.mean.shape == torch.Size([4, 2, 2])
    assert posterior.variance.shape == torch.Size([4, 2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_tabular_hybrid_deepgp_ehvi_candidate_generation() -> None:
    torch.manual_seed(0)
    frame = pd.DataFrame(
        {
            "x1": [float(index) / 12.0 for index in range(12)],
            "x2": [float(index % 4) / 4.0 for index in range(12)],
            "property": [0.1 + 0.05 * index for index in range(12)],
            "quality": ["a", "b", "c"] * 4,
        }
    )
    optimizer = TabularBayesianOptimizer(
        model_config={
            "task_type": "hybrid",
            "model_type": "deepgp",
            "input_transform_config": {
                "perturbation": False,
                "n_w": 4,
                "std": 0.1,
            },
        },
        fit_config={"skip_fit": True},
        multi_output_config={
            "output_configs": [
                {
                    "task_type": "regression",
                    "model_type": "deepgp",
                    "name": "property",
                    "model_kwargs": {
                        "list_hidden_dims": [2],
                        "num_inducing": 4,
                    },
                },
                {
                    "task_type": "multiclass",
                    "model_type": "deepgp",
                    "name": "quality",
                    "model_kwargs": {
                        "hidden_dim": 2,
                        "list_hidden_dims": [2],
                        "num_inducing": 4,
                    },
                },
            ],
            "use_hybrid": True,
        },
        input_cols=["x1", "x2"],
        target_cols=["property", "quality"],
    )
    optimizer.fit(frame)

    candidates, acq_value = optimizer.candidate(
        acq_config={"name": "ehvi"},
        opt_config={
            "q": 2,
            "num_restarts": 2,
            "raw_samples": 4,
        },
    )

    assert len(candidates) == 2
    assert torch.isfinite(torch.as_tensor(acq_value)).all()
