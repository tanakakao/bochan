from __future__ import annotations

import pytest

from bochan.serving.fastapi.services.candidates import _normalize_transport_cost_config


def test_fastapi_accepts_serializable_affine_and_fixed_cost_configs():
    affine = _normalize_transport_cost_config(
        {
            "kind": "affine",
            "fixed_cost": 1.0,
            "fidelity_weights": {-1: 2.0},
        }
    )
    fixed = _normalize_transport_cost_config(
        {
            "kind": "fixed",
            "fixed_cost": 3.0,
        }
    )

    assert affine["kind"] == "affine"
    assert affine["fidelity_weights"] == {-1: 2.0}
    assert fixed == {"kind": "fixed", "fixed_cost": 3.0}


def test_fastapi_rejects_callable_cost_mode():
    with pytest.raises(ValueError, match="Python API"):
        _normalize_transport_cost_config({"kind": "callable"})


def test_fastapi_rejects_callable_field_even_with_serializable_kind():
    with pytest.raises(ValueError, match="Python API"):
        _normalize_transport_cost_config(
            {
                "kind": "fixed",
                "fixed_cost": 2.0,
                "cost_callable": "not-json-callable",
            }
        )
