from bochan.serving.fastapi import create_api_router
from bochan.serving.fastapi.routers import create_api_router as owned_create_api_router


def test_fastapi_router_composition_has_one_owner() -> None:
    assert create_api_router is owned_create_api_router
    assert create_api_router.__module__ == "bochan.serving.fastapi.routers"
