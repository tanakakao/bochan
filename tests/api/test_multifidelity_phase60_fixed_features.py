import torch

from bochan.api.configs import OptimizeConfig
from bochan.models.multifidelity.optimization import enumerate_discrete_fidelities_into_opt_config


class _Model:
    fidelity_features = (1, 2)
    target_fidelities = {1: 1.0, 2: 1.0}


def test_global_fixed_fidelity_filters_allowed_assignments():
    resolved = enumerate_discrete_fidelities_into_opt_config(
        OptimizeConfig(
            fidelity_values={1: [0.25, 1.0], 2: [0.5, 1.0]},
            fixed_features={1: 1.0},
        ),
        model=_Model(),
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
    )
    assert resolved.fixed_features_list == [
        {1: 1.0, 2: 0.5},
        {1: 1.0, 2: 1.0},
    ]
