import pytest

from bochan.api.configs import OptimizeConfig


def test_fidelity_query_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        OptimizeConfig(
            fidelity_values={-2: [0.25, 1.0], -1: [0.5, 1.0]},
            fidelity_assignments=[{-2: 1.0, -1: 1.0}],
        )
