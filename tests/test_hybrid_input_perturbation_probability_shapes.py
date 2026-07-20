from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec
from bochan.tabular import TabularBayesianOptimizer


class _ExpandedRegressionModel(nn.Module):
    def __init__(self, n_w: int) -> None:
        super().__init__()
        self.n_w = int(n_w)
        self.train_inputs = (torch.zeros(6, 5, dtype=torch.double),)
        self.train_targets = torch.zeros(6, 1, dtype=torch.double)

    def posterior(self, X: torch.Tensor, **kwargs):
        del kwargs
        mean = X[..., :1].repeat_interleave(self.n_w, dim=-2)
        return SimpleNamespace(mean=mean, variance=torch.ones_like(mean))


class _ExpandedOrdinalModel(nn.Module):
    def __init__(self, n_w: int, sample_count: int = 3) -> None:
        super().__init__()
        self.n_w = int(n_w)
        self.sample_count = int(sample_count)
        self.train_inputs = (torch.zeros(6, 5, dtype=torch.double),)
        self.train_targets = torch.arange(6) % 3

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        base = torch.tensor([0.6, 0.3, 0.1], device=X.device, dtype=X.dtype)
        probs = base.expand(*X.shape[:-1], 3)
        probs = probs.repeat_interleave(self.n_w, dim=-2)
        return probs.unsqueeze(0).expand(self.sample_count, *probs.shape)


def test_hybrid_probability_shapes_preserve_perturbation_expanded_q() -> None:
    n_w = 4
    hybrid = HybridMultiOutputModel(
        [
            OutputSpec(
                name="property",
                task_type="regression",
                model=_ExpandedRegressionModel(n_w=n_w),
            ),
            OutputSpec(
                name="quality",
                task_type="ordinal",
                model=_ExpandedOrdinalModel(n_w=n_w),
                utility_values=[0.0, 1.0, 2.0],
            ),
        ]
    )
    X = torch.rand(256, 2, 5, dtype=torch.double)

    posterior = hybrid.posterior(X)
    probs = hybrid.class_probs_list(X, output_indices=["quality"])[0]

    assert posterior.mean.shape == torch.Size([256, 8, 2])
    assert posterior.variance.shape == torch.Size([256, 8, 2])
    assert probs.shape == torch.Size([256, 8, 3])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_tabular_hybrid_ordinal_ei_sequential_q10_with_input_perturbation() -> None:
    torch.manual_seed(0)
    records = []
    labels = ["a", "b", "c"]
    for index in range(18):
        records.append(
            {
                "x1": 0.05 + 0.04 * index,
                "x2": float(index % 5) / 4.0,
                "property": 0.1 + 0.04 * index,
                "y_ord_str": labels[index % 3],
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["y_ord_str"] = frame["y_ord_str"].astype(object)

    optimizer = TabularBayesianOptimizer(
        model_config={
            "task_type": "hybrid",
            "model_type": "base",
            "input_transform_config": {
                "perturbation": True,
                "n_w": 4,
                "std": 0.1,
            },
        },
        fit_config={"skip_fit": True},
        input_cols=["x1", "x2"],
        target_cols=["property", "y_ord_str"],
        multi_output_config={
            "output_configs": [
                {
                    "task_type": "regression",
                    "model_type": "base",
                    "name": "property",
                },
                {
                    "task_type": "ordinal",
                    "model_type": "base",
                    "name": "y_ord_str",
                    "ordered_categories": ["a", "b", "c"],
                },
            ],
            "use_hybrid": True,
        },
    )
    optimizer.fit(frame)

    candidates, acq_value = optimizer.candidate(
        objective_mode="scalar",
        objective_output="property",
        objective_direction="maximize",
        acq_config={"name": "ei"},
        outcome_constraint_config={
            "constraints": [
                {
                    "kind": "ordinal_rank",
                    "output": "y_ord_str",
                    "rank": "b",
                    "sense": "le",
                    "probability_threshold": 0.8,
                }
            ],
            "reduce_constraints": "prod",
            "reduce_q": "min",
            "eta": 0.02,
        },
        opt_config={
            "q": 10,
            "sequential": True,
            "num_restarts": 2,
            "raw_samples": 4,
            "optimizer_kwargs": {
                "options": {
                    "maxiter": 2,
                    "batch_limit": 2,
                }
            },
        },
    )

    assert len(candidates) == 10
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_tabular_hybrid_pca_multiclass_ehvi_sequential_q10_with_perturbation() -> None:
    torch.manual_seed(0)
    records = []
    labels = ["a", "b", "c"]
    for index in range(18):
        records.append(
            {
                "x1": 0.05 + 0.04 * index,
                "x2": float(index % 5) / 4.0,
                "x3": float(index % 3) / 2.0,
                "x4": float(index % 7) / 6.0,
                "x5": float(index % 4) / 3.0,
                "property": 0.1 + 0.04 * index,
                "y_ord_str": labels[index % 3],
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["y_ord_str"] = frame["y_ord_str"].astype(object)

    optimizer = TabularBayesianOptimizer(
        model_config={
            "task_type": "hybrid",
            "model_type": "pca",
            "input_transform_config": {
                "perturbation": True,
                "n_w": 4,
                "std": 0.1,
            },
        },
        fit_config={"skip_fit": True},
        input_cols=["x1", "x2", "x3", "x4", "x5"],
        target_cols=["property", "y_ord_str"],
        multi_output_config={
            "output_configs": [
                {
                    "task_type": "regression",
                    "model_type": "pca",
                    "name": "property",
                },
                {
                    "task_type": "multiclass",
                    "model_type": "pca",
                    "name": "y_ord_str",
                },
            ],
            "use_hybrid": True,
        },
    )
    optimizer.fit(frame)

    X = torch.rand(256, 2, 5, dtype=torch.double)
    posterior = optimizer.bo.model.posterior(X)
    probs = optimizer.bo.model.class_probs_list(
        X,
        output_indices=["y_ord_str"],
    )[0]

    assert posterior.mean.shape == torch.Size([256, 8, 2])
    assert posterior.variance.shape == torch.Size([256, 8, 2])
    assert probs.shape == torch.Size([256, 8, 3])

    candidates, acq_value = optimizer.candidate(
        acq_config={"name": "ehvi"},
        opt_config={
            "q": 10,
            "sequential": True,
            "num_restarts": 2,
            "raw_samples": 4,
            "optimizer_kwargs": {
                "options": {
                    "maxiter": 2,
                    "batch_limit": 2,
                }
            },
        },
    )

    assert len(candidates) == 10
    assert torch.isfinite(torch.as_tensor(acq_value)).all()
