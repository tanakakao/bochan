from __future__ import annotations

import pytest
import torch

from bochan.api import BayesianOptimizer, FitConfig, ModelConfig


@pytest.mark.parametrize("model_type", ["pca", "rembo"])
def test_api_ordinal_projected_mixed_fit_accepts_beta(model_type: str) -> None:
    train_X = torch.rand(18, 5, dtype=torch.double)
    train_X[:, 4] = torch.arange(18, dtype=torch.double) % 2
    train_Y = torch.arange(18) % 3

    bo = BayesianOptimizer(
        model_config=ModelConfig(
            task_type="ordinal",
            model_type=model_type,
            cat_dims=[4],
            outcome_transform=False,
            model_kwargs={
                "n_components": 2,
                "category_counts": {4: 2},
            },
        ),
        fit_config=FitConfig(
            num_epochs=0,
            lr=0.01,
            beta=0.01,
        ),
    )

    bo.fit(train_X, train_Y)

    assert bo.model is not None
    assert bo.mll is not None
