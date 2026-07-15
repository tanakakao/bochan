import pytest

pytest.importorskip("fastapi")

from bochan.serving.fastapi.routers.predictions import predict as modular_predict
from bochan.serving.fastapi.schemas import PredictRequest as ModularPredictRequest
from bochan.serving.fastapi.stores import InMemoryOptimizerStore
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
