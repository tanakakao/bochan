from pathlib import Path
import runpy
import subprocess
import traceback

root = Path(__file__).resolve().parent
repo = root.parents[1]
names = [
    "patch_binary_bo_single_current.py",
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
    try:
        runpy.run_path(str(root / name), run_name="__main__")
    except Exception:
        failure = repo / "binary-epistemic-migration-failure.log"
        failure.write_text(
            f"FAILED {name}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(
            [
                "git",
                "config",
                "user.email",
                "41898282+github-actions[bot]@users.noreply.github.com",
            ],
            check=True,
        )
        subprocess.run(["git", "add", str(failure)], check=True)
        subprocess.run(
            ["git", "commit", "-m", "Record epistemic migration failure"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "push",
                "origin",
                "HEAD:fix/binary-epistemic-uncertainty",
            ],
            check=True,
        )
        raise
