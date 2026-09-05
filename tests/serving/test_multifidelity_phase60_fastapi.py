from bochan.serving.fastapi.schemas.configs import OptimizeConfigSchema
from bochan.serving.fastapi.schemas.requests import CandidateRequest


def test_optimize_schema_accepts_multidimensional_fidelity_values():
    schema = OptimizeConfigSchema(
        fidelity_values={-2: [0.25, 1.0], -1: [0.5, 1.0]},
    )
    assert schema.fidelity_values == {-2: [0.25, 1.0], -1: [0.5, 1.0]}


def test_candidate_request_accepts_explicit_fidelity_assignments():
    request = CandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "qei"},
            "fidelity_assignments": [
                {"-2": 0.25, "-1": 0.5},
                {"-2": 1.0, "-1": 1.0},
            ],
        }
    )
    assert request.fidelity_assignments == [
        {-2: 0.25, -1: 0.5},
        {-2: 1.0, -1: 1.0},
    ]
