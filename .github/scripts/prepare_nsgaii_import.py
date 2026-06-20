from pathlib import Path

root = Path(__file__).resolve().parents[2]

nsgaii_path = root / "src/bochan/optim/nsgaii.py"
nsgaii_text = nsgaii_path.read_text(encoding="utf-8")
old_import = "from botorch.utils.multi_objective.optimize import optimize_with_nsgaii"
new_import = '''try:
    from botorch.utils.multi_objective.optimize import optimize_with_nsgaii
except ImportError as exc:  # pragma: no cover - depends on BoTorch version
    _NSGAII_IMPORT_ERROR = exc

    def optimize_with_nsgaii(*args, **kwargs):
        raise ImportError(
            "optimize_with_nsgaii is unavailable in the installed BoTorch version."
        ) from _NSGAII_IMPORT_ERROR
'''
if old_import in nsgaii_text:
    nsgaii_path.write_text(
        nsgaii_text.replace(old_import, new_import, 1),
        encoding="utf-8",
    )

pyproject_path = root / "pyproject.toml"
pyproject_text = pyproject_path.read_text(encoding="utf-8")
marker_anchor = 'pythonpath = ["src"]\n'
marker_config = 'markers = [\n  "slow: marks computationally expensive model tests",\n]\n'
if marker_config not in pyproject_text:
    if marker_anchor not in pyproject_text:
        raise RuntimeError("pytest configuration anchor was not found")
    pyproject_path.write_text(
        pyproject_text.replace(marker_anchor, marker_anchor + marker_config, 1),
        encoding="utf-8",
    )
