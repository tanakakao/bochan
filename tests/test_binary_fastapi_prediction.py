from fastapi import FastAPI
from fastapi.testclient import TestClient

from bochan.api.fastapi import SessionStore, create_router
from bochan.serving.fastapi.dependencies import InMemoryOptimizerStore
from bochan.serving.fastapi.routers.predictions import predict as modular_predict
from bochan.serving.fastapi.schemas import PredictRequest as ModularPredictRequest
from tests.test_binary_api_prediction import _BinaryModel, make_optimizer


def test_modular_fastapi_returns_json_posterior_summary() -> None:
    optimizer = make_optimizer("binary", _BinaryModel())
    store = InMemoryOptimizerStore()
    model_id = store.add(optimizer)

    response = modular_predict(
        model_id,
        ModularPredictRequest(X=[[0.0], [1.0]], return_type="posterior"),
        store,
    )

    assert response.task_type == "binary"
    assert response.prediction_space == "probability"
    assert response.variance_kind == "bernoulli_observation"
    assert response.posterior["type"] == "_Posterior"
    assert response.posterior["mean"] == [[0.2], [0.8]]
    assert response.value == response.posterior


def test_old_fastapi_returns_json_posterior_summary() -> None:
    optimizer = make_optimizer("binary", _BinaryModel())
    store = SessionStore()
    session_id = store.create(optimizer)
    app = FastAPI()
    app.include_router(create_router(store))
    client = TestClient(app)

    response = client.post(
        f"/bochan/sessions/{session_id}/predict",
        json={"X": [[0.0], [1.0]], "return_type": "posterior"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_type"] == "binary"
    assert payload["prediction_space"] == "probability"
    assert payload["variance_kind"] == "bernoulli_observation"
    assert payload["posterior"]["type"] == "_Posterior"
    assert payload["posterior"]["mean"] == [[0.2], [0.8]]
