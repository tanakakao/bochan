from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
file_path = ROOT / "tests/test_webapp_input_perturbation_prediction_shape.py"
text = file_path.read_text(encoding="utf-8")
old = '''def test_web_workflow_keeps_display_mean_and_installs_risk_baseline_adapter() -> None:
    assert target_settings._as_2d is normalize_prediction_rows
    assert target_results._as_2d is normalize_prediction_rows
    assert web_workflows._workflows_tabular._as_2d is not normalize_prediction_rows
'''
new = '''def test_web_workflow_uses_source_level_prediction_row_normalization() -> None:
    values = torch.tensor(
        [[1.0], [3.0], [10.0], [14.0]],
        dtype=torch.double,
    )
    expected = torch.tensor([[2.0], [12.0]], dtype=torch.double)

    torch.testing.assert_close(target_settings._as_2d(values, n_rows=2), expected)
    torch.testing.assert_close(target_results._as_2d(values, n_rows=2), expected)
    assert target_results._as_2d is target_settings._as_2d
    assert not hasattr(web_workflows._workflows_tabular, "_as_2d")
'''
if old in text:
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise RuntimeError("Expected runtime-patch assertion was not found")

print("Prediction-shape test updated for source-level wiring")
