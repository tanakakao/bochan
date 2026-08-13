from bochan.serving.fastapi import router


def test_fastapi_root_router_exports_nothing() -> None:
    assert router.__all__ == []
