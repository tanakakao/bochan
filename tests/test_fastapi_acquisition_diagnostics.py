from bochan.serving.fastapi.routers.candidates import _candidate_response
from bochan.serving.fastapi.schemas.responses import CandidateResponse


def test_candidate_response_diagnostics_remain_optional():
    response = CandidateResponse(
        model_id="model-1",
        candidates=[[0.1, 0.2]],
        acq_value=1.5,
    )

    assert response.diagnostics is None


def test_candidate_response_serializes_acquisition_diagnostics():
    response = _candidate_response(
        "model-1",
        [[0.1, 0.2]],
        1.5,
        diagnostics={
            "training_rows": 4,
            "baseline_rows": 3,
            "baseline_filtered": True,
            "pending_rows": 1,
        },
    )

    assert response.diagnostics == {
        "training_rows": 4,
        "baseline_rows": 3,
        "baseline_filtered": True,
        "pending_rows": 1,
    }
