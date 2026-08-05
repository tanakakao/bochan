from pathlib import Path


def test_composition_runtime_preserves_dataset_and_regression_adapters() -> None:
    source = Path("web/src/compositionRuntime.ts").read_text(encoding="utf-8")

    assert 'url.pathname.endsWith("/datasets")' in source
    assert 'url.pathname.endsWith("/regression/run")' in source
    assert "web_composition: backendSettings(settings)" in source
    assert "payload.search_space" in source
    assert "formulaLikeColumn" in source
    assert "latestDataset = payload" in source
    assert "synchronizePrepareControls" in source
    assert ".feature-variable-choice" in source
