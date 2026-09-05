import torch

from bochan.api.configs import OptimizeConfig
from bochan.api.optimizer.dispatch import optimize_candidates


class _Model:
    fidelity_features = (1, 2)
    target_fidelities = {1: 1.0, 2: 1.0}


class _Acq:
    model = _Model()


def test_multidimensional_discrete_dispatch_uses_mixed_backend():
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.2, 0.25, 0.5]], dtype=torch.double), torch.tensor(0.0)

    optimize_candidates(
        _Acq(),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
        OptimizeConfig(
            fidelity_values={1: [0.25, 1.0], 2: [0.5, 1.0]},
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    resolved = captured["config"]
    assert len(resolved.fixed_features_list) == 4
    assert "mixed" in str(resolved.optimizer)
