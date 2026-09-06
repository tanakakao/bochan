from __future__ import annotations

import pytest

from bochan.serving.fastapi.services.candidates import _normalize_transport_cost_config


def test_fastapi_accepts_serializable_learned_gp_training_data():
    config = _normalize_transport_cost_config(
        {
            "kind": "learned_gp",
            "train_X": [[0.0, 0.2], [0.5, 0.6], [1.0, 1.0]],
            "train_cost": [1.0, 2.0, 5.0],
            "log_cost": True,
            "use_mean": False,
        }
    )

    assert config["kind"] == "learned_gp"
    assert config["train_cost"] == [1.0, 2.0, 5.0]
    assert config["use_mean"] is False


def test_fastapi_rejects_prebuilt_learned_cost_model():
    with pytest.raises(ValueError, match="Python API"):
        _normalize_transport_cost_config(
            {
                "kind": "learned_gp",
                "cost_model": "not-serializable",
            }
        )
