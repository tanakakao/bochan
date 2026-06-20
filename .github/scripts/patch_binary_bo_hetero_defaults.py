from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/acquisition/binary/bayesian_optimization/hetero_single_output.py"
text = path.read_text(encoding="utf-8")
old = "        samples_are_probs: bool = True,\n        apply_sigmoid_if_needed: bool = False,\n"
new = "        samples_are_probs: bool = False,\n        apply_sigmoid_if_needed: bool = True,\n"
if text.count(old) != 1:
    raise RuntimeError("hetero BO default block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
