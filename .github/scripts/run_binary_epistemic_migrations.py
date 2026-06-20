from pathlib import Path
import runpy

root = Path(__file__).resolve().parent
names = [
    "patch_binary_bo_single.py",
    "patch_binary_bo_utils.py",
    "patch_binary_bo_multi_imports.py",
    "patch_binary_bo_multi_ehvi.py",
    "patch_binary_bo_hetero_multi.py",
    "patch_binary_bo_hetero_defaults.py",
    "patch_binary_active_single.py",
    "patch_binary_ipv.py",
    "patch_binary_active_multi.py",
    "patch_binary_active_hetero_probability.py",
    "patch_binary_active_hetero_unified.py",
    "patch_binary_active_hetero_multi.py",
    "patch_binary_visualization_epistemic.py",
]
for name in names:
    print(f"RUN {name}", flush=True)
    runpy.run_path(str(root / name), run_name="__main__")
