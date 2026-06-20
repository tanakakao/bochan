from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/bochan/optim/nsgaii.py"
text = path.read_text(encoding="utf-8")
old = "from botorch.utils.multi_objective.optimize import optimize_with_nsgaii"
new = '''try:
    from botorch.utils.multi_objective.optimize import optimize_with_nsgaii
except ImportError as exc:  # pragma: no cover - depends on BoTorch version
    _NSGAII_IMPORT_ERROR = exc

    def optimize_with_nsgaii(*args, **kwargs):
        raise ImportError(
            "optimize_with_nsgaii is unavailable in the installed BoTorch version."
        ) from _NSGAII_IMPORT_ERROR
'''
if old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
