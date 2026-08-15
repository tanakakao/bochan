from __future__ import annotations

from pathlib import Path

from bochan.serving.webapp.schemas.regression import RegressionRunRequest


ROOT = Path(__file__).resolve().parents[1]


def test_web_input_perturbation_is_disabled_unless_user_enables_it() -> None:
    """The Web defaults and request schema must not silently enable perturbation."""

    settings = (ROOT / "web/src/context/useWorkbenchRunSettings.ts").read_text(
        encoding="utf-8"
    )
    api = (ROOT / "web/src/api.ts").read_text(encoding="utf-8")

    assert "const [inputPerturbation, setInputPerturbation] = useState(false);" in settings
    assert "setInputPerturbation(false);" in settings
    assert "setInputPerturbation(true)" not in settings
    assert "input_perturbation: input.inputPerturbation" in api

    request = RegressionRunRequest(
        dataset_id="dataset",
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
    )
    assert request.input_perturbation is False
