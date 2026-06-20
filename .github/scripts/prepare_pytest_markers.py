from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "pyproject.toml"
text = path.read_text(encoding="utf-8")
anchor = 'pythonpath = ["src"]\n'
addition = 'markers = [\n  "slow: marks computationally expensive model tests",\n]\n'
if addition not in text:
    if anchor not in text:
        raise RuntimeError("pytest configuration anchor was not found")
    path.write_text(text.replace(anchor, anchor + addition, 1), encoding="utf-8")
