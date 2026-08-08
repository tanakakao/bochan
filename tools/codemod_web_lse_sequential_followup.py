from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
file_path = ROOT / "web/src/api.ts"
text = file_path.read_text(encoding="utf-8")
old = '''        sequential:
          Boolean(input.sequential ?? true) ||
          input.searchSpace.some((variable) => variable.type === "categorical") ||
          searchMethod === "cmaes",
'''
new = '''        sequential:
          Boolean(input.sequential ?? true) ||
          input.searchSpace.some((variable) => variable.type === "categorical") ||
          searchMethod === "cmaes" ||
          (input.acquisitionFamily === "level_set_estimation" && input.inputPerturbation),
'''
if old in text:
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise RuntimeError("Expected optimizer sequential payload block was not found")

print("Perturbed LSE sequential payload guard applied")
