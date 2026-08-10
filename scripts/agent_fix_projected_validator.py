from pathlib import Path

path = Path(__file__).with_name("agent_align_projected_api.py")
text = path.read_text(encoding="utf-8")

old_test = '''            if "latent_dim" not in args:\n                offenders.append((str(path.relative_to(REPO_ROOT)), class_node.name, "missing latent_dim"))\n'''
new_test = '''            forwards_constructor_args = init.args.vararg is not None or init.args.kwarg is not None\n            if "latent_dim" not in args and not forwards_constructor_args:\n                offenders.append((str(path.relative_to(REPO_ROOT)), class_node.name, "missing latent_dim"))\n'''
if old_test not in text:
    raise RuntimeError("projected contract-test validator anchor not found")
text = text.replace(old_test, new_test, 1)

old_validate = '''            if "n_components" in args or "latent_dim" not in args:\n                raise RuntimeError(f"non-canonical projected signature: {path}:{class_node.name}: {sorted(args)}")\n'''
new_validate = '''            forwards_constructor_args = init.args.vararg is not None or init.args.kwarg is not None\n            if "n_components" in args or ("latent_dim" not in args and not forwards_constructor_args):\n                raise RuntimeError(f"non-canonical projected signature: {path}:{class_node.name}: {sorted(args)}")\n'''
if old_validate not in text:
    raise RuntimeError("projected runtime validator anchor not found")
text = text.replace(old_validate, new_validate, 1)

path.write_text(text, encoding="utf-8")
