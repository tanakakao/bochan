from pathlib import Path
from runpy import run_path

_TEST_DIR = Path(__file__).parent
_PUBLIC_API = run_path(_TEST_DIR / "test_multiclass_nparego_public_api.py")
_SIGNATURE = run_path(_TEST_DIR / "test_multiclass_nparego_signature.py")

test_multiclass_multitask_nparego_uses_training_baseline = _PUBLIC_API[
    "test_multiclass_multitask_nparego_uses_training_baseline"
]
test_multiclass_nparego_keeps_baseline_context = _SIGNATURE[
    "test_multiclass_nparego_keeps_baseline_context"
]
