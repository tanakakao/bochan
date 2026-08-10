from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "src/bochan/models/classification/binary/base/models.py"
TEST_PATH = ROOT / "tests/test_model_contract_refactor.py"

OLD = "GPClassificationMixedModel"
NEW = "BinaryClassificationMixedGPModel"


def fix_model_message() -> None:
    text = MODEL_PATH.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(f"expected exactly one stale binary mixed name, got {count}")
    MODEL_PATH.write_text(text.replace(OLD, NEW), encoding="utf-8")


def strengthen_guard() -> None:
    text = TEST_PATH.read_text(encoding="utf-8")
    old = '''def test_binary_base_has_no_stale_compatibility_class_names() -> None:
    path = MODELS_ROOT / "classification" / "binary" / "base" / "models.py"
    source = path.read_text(encoding="utf-8")
    assert '"GPClassificationModel"' not in source
    assert '"GPClassificationMixedModel"' not in source
'''
    new = '''def test_binary_base_has_no_stale_compatibility_class_names() -> None:
    path = MODELS_ROOT / "classification" / "binary" / "base" / "models.py"
    source = path.read_text(encoding="utf-8")
    assert "GPClassificationModel" not in source
    assert "GPClassificationMixedModel" not in source
'''
    if old not in text:
        raise RuntimeError("binary compatibility-name guard block not found")
    TEST_PATH.write_text(text.replace(old, new, 1), encoding="utf-8")


def validate() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    if OLD in source:
        raise RuntimeError(f"stale binary mixed class name remains: {OLD}")


if __name__ == "__main__":
    fix_model_message()
    strengthen_guard()
    validate()
