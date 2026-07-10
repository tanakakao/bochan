from __future__ import annotations

import torch
from botorch.models.model import Model

from bochan.acquisition.multiclass.bayesian_optimization import (
    qMultiOutputMulticlassExpectedImprovement,
)


class _DummyModel(Model):
    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        raise AssertionError("The shape regression test overrides probability sampling.")


class _DeepGPStyleExpectedImprovement(
    qMultiOutputMulticlassExpectedImprovement
):
    def _target_prob_samples_per_output(
        self,
        X: torch.Tensor,
        *,
        num_samples: int,
    ) -> torch.Tensor:
        # DeepGP can retain a latent sample axis while omitting the singleton
        # t-batch axis. After EI reduces q, samples and outputs, this becomes
        # value.shape == (10,) although X has t-batch shape (1,).
        latent_values = torch.linspace(
            0.1,
            0.9,
            10,
            device=X.device,
            dtype=X.dtype,
        )
        return latent_values.view(1, 10, 1, 1).expand(
            num_samples,
            10,
            X.shape[-2],
            1,
        )


def test_expected_improvement_averages_deepgp_latent_axis_for_single_tbatch():
    acqf = _DeepGPStyleExpectedImprovement(
        model=_DummyModel(),
        target_class=1,
        best_f=0.0,
        num_samples=8,
    )
    X = torch.zeros(1, 1, 2)

    value = acqf(X)

    assert value.shape == torch.Size([1])
    torch.testing.assert_close(value, torch.tensor([0.5]))
