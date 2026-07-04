from __future__ import annotations

import torch

from bochan.optim.nsgaii_adapter import _make_version_compatible_optimizer


class _TwoOutputAcquisition:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat([X[..., :1], 1.0 - X[..., :1]], dim=-1)


class _ParameterizedObjective(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.tensor([1.0, 2.0], dtype=torch.double)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.weights


def test_optimizer_disables_grad_for_botorch_objective_numpy_conversion() -> None:
    def botorch_like_optimize_with_nsgaii(
        acq_function,
        bounds,
        num_objectives,
        q=None,
        ref_point=None,
        objective=None,
        constraints=None,
        population_size=250,
        max_gen=None,
        seed=None,
        fixed_features=None,
    ):
        del num_objectives, q, ref_point, constraints
        del population_size, max_gen, seed, fixed_features

        X = torch.tensor(
            [[0.25], [0.75]],
            dtype=bounds.dtype,
            device=bounds.device,
        )
        # BoTorch evaluates the acquisition under no_grad, but some versions
        # apply the objective transform after leaving that context.
        with torch.no_grad():
            values = acq_function(X=X.unsqueeze(-2))
        values = objective(values)

        # This mirrors the failing conversion in BotorchPymooProblem._evaluate.
        values.cpu().numpy()
        return X, values.squeeze(-2)

    compatible = _make_version_compatible_optimizer(
        botorch_like_optimize_with_nsgaii
    )
    X, Y = compatible(
        acq_function=_TwoOutputAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_objectives=2,
        q=2,
        objective=_ParameterizedObjective(),
    )

    expected_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    expected_Y = torch.tensor([[0.25, 1.5], [0.75, 0.5]], dtype=torch.double)
    torch.testing.assert_close(X, expected_X)
    torch.testing.assert_close(Y, expected_Y)
    assert not Y.requires_grad
