from __future__ import annotations

from pathlib import Path
import runpy

SCRIPT = Path(__file__).with_name("fix_classification_duplicate_controls.py")
text = SCRIPT.read_text(encoding="utf-8")
old = '''    text = replace_once(
        text,
        "        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()\\n",
        "        self.X_observed = None\\n",
        label="multiclass multi observed init",
    )
'''
new = '''    text = replace_once(
        text,
        "        self.X_observed = None if X_observed is None else torch.as_tensor(X_observed).detach()\\n"
        "        self.apply_softmax_if_needed = bool(apply_softmax_if_needed)\\n",
        "        self.X_observed = None\\n"
        "        self.apply_softmax_if_needed = bool(apply_softmax_if_needed)\\n",
        label="multiclass multi observed init",
    )
'''
count = text.count(old)
if count != 1:
    raise RuntimeError(
        "Could not refine multiclass multi-output observed initialization: "
        f"expected one codemod block, found {count}."
    )
SCRIPT.write_text(text.replace(old, new, 1), encoding="utf-8")
runpy.run_path(str(SCRIPT), run_name="__main__")
