from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(file_name: str, old: str, new: str) -> None:
    file_path = ROOT / file_name
    text = file_path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected snippet not found in {file_name}: {old!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "web/src/api.ts",
    '''          web_family: input.acquisitionFamily,
          web_level_set_parameter: input.acquisitionFamily === "level_set_estimation" ? input.beta : null,
          web_risk_type: effectiveRiskType,''',
    '''          web_family: input.acquisitionFamily,
          ...(input.acquisitionFamily === "level_set_estimation"
            ? { web_level_set_parameter: input.beta }
            : {}),
          web_risk_type: effectiveRiskType,''',
)

replace_once(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''    acqf_kwargs.pop("web_risk_type", None)
    acqf_kwargs.pop("web_risk_alpha", None)
    if family not in {''',
    '''    acqf_kwargs.pop("web_risk_type", None)
    acqf_kwargs.pop("web_risk_alpha", None)
    if family != "level_set_estimation":
        acqf_kwargs.pop("web_level_set_parameter", None)
    if family not in {''',
)

print("LSE marker guard applied successfully")
