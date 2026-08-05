from pathlib import Path


def test_composition_runtime_preserves_dataset_and_regression_adapters() -> None:
    source = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")

    assert 'url.pathname.endsWith("/datasets")' in source
    assert 'url.pathname.endsWith("/regression/run")' in source
    assert "web_composition: backendSettings(settings)" in source
    assert "payload.search_space" in source
    assert "formulaLikeColumn" in source
    assert "responseWithJson" in source
    assert "MutationObserver" not in source
    assert "document.querySelector" not in source
    assert "appendChild" not in source
