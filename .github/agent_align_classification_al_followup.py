from pathlib import Path

p = Path("src/bochan/serving/webapp/workflows_tabular.py")
text = p.read_text(encoding="utf-8")
needle = '''    acqf_kwargs.setdefault("X_observed", train_x)\n\n\ndef _request_with_constraints'''
replacement = '''    acqf_kwargs.setdefault("X_observed", train_x)\n\n\ndef _set_active_learning_reference_kwargs(\n    acqf_kwargs: dict[str, object],\n    *,\n    acq_key: str,\n    train_x: object,\n) -> None:\n    """Backward-compatible Regression single-output AL reference helper."""\n    _set_active_learning_kwargs(\n        acqf_kwargs,\n        acq_key=acq_key,\n        train_x=train_x,\n        task_type="regression",\n        multi_output=False,\n    )\n\n\ndef _request_with_constraints'''
if needle not in text:
    raise RuntimeError("new Web AL helper insertion point not found")
p.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
print("follow-up compatibility patch applied")
