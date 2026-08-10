from pathlib import Path

path = Path(__file__).with_name("agent_align_projected_api.py")
text = path.read_text(encoding="utf-8")

# Public PCA/REMBO wrappers that forward *args/**kwargs inherit latent_dim from
# their projected base class. Do not require the forwarding wrapper itself to
# redeclare latent_dim; only forbid an explicit legacy n_components argument.
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

# The contract concerns public projected *models*, not internal PCA / REMBO
# transformer or config objects. Skip those internal classes in both the
# generated regression test and the one-shot validator.
loop_anchor = '''            if not class_node.name.startswith(("PCA", "REMBO")):\n                continue\n            init = next(\n'''
loop_replacement = '''            if not class_node.name.startswith(("PCA", "REMBO")):\n                continue\n            if class_node.name.endswith(("Transformer", "Config")):\n                continue\n            init = next(\n'''
if text.count(loop_anchor) != 2:
    raise RuntimeError(f"expected 2 projected class-loop anchors, got {text.count(loop_anchor)}")
text = text.replace(loop_anchor, loop_replacement)

# Ordinal projected models previously used n_components as their public API,
# while PCAConfig / REMBOConfig correctly use n_components internally. Narrow
# the migration so only model constructor references become latent_dim.
old_ordinal = '''            text = text.replace("n_components=n_components,", "n_components=latent_dim,")\n            text = text.replace("``n_components``", "``latent_dim``")\n            text = text.replace("n_components の指定", "latent_dim の指定")\n'''
new_ordinal = '''            text = text.replace(\n                "            n_components=n_components,\\n            default=2,",\n                "            n_components=latent_dim,\\n            default=2,",\n            )\n            text = text.replace(\n                "- 外部 API は ``n_components`` に統一する。\\n"\n                "    - 旧 API の ``latent_dim`` は ``__init__`` 引数から削除する。",\n                "- 外部 API は ``latent_dim`` に統一する。\\n"\n                "    - PCAConfig / REMBOConfig 内部では ``n_components`` を使う。",\n            )\n'''
if old_ordinal not in text:
    raise RuntimeError("ordinal projected migration anchor not found")
text = text.replace(old_ordinal, new_ordinal, 1)

# After the public n_components parameter is removed, all calls to the shared
# dimension resolver must use its canonical `(latent_dim, default)` signature.
old_write = '''        if path.parts[-3:] == ("ordinal", "high_dim", "decomposition.py"):\n            text = text.replace("        self.n_components = self.projected_dim\\n", "")\n            text = text.replace(\n                "        # 内部互換用。外部 API からは latent_dim を削除する。\\n",\n                "",\n            )\n        write(path, text)\n\n\ndef simplify_multiclass_resolver() -> None:\n'''
new_write = '''        text = re.sub(\n            r"_resolve_latent_dim\\(latent_dim=latent_dim,\\s*n_components=n_components,\\s*default=(?P<default>\\d+)\\)",\n            lambda match: f"_resolve_latent_dim(latent_dim=latent_dim, default={match.group('default')})",\n            text,\n        )\n        if path.parts[-3:] == ("ordinal", "high_dim", "decomposition.py"):\n            text = text.replace("        self.n_components = self.projected_dim\\n", "")\n            text = text.replace(\n                "        # 内部互換用。外部 API からは latent_dim を削除する。\\n",\n                "",\n            )\n        write(path, text)\n\n\ndef simplify_multiclass_resolver() -> None:\n'''
if old_write not in text:
    raise RuntimeError("projected resolver-call cleanup anchor not found")
text = text.replace(old_write, new_write, 1)

# Keep generated regression tests at exactly one trailing newline so
# `git diff --check` does not reject the one-shot migration.
old_eof = '    write(path, text.rstrip() + addition + "\\n")\n'
new_eof = '    write(path, text.rstrip() + addition.rstrip() + "\\n")\n'
if old_eof not in text:
    raise RuntimeError("projected contract-test EOF anchor not found")
text = text.replace(old_eof, new_eof, 1)

# Add a validator guard for stale calls to the removed resolver argument.
validate_anchor = '''    projected = ast.parse(read(MODELS / "components" / "projected.py"))\n'''
validate_replacement = '''    for path in PROJECTED_FILES:\n        source = read(path)\n        if re.search(\n            r"_resolve_latent_dim\\([^)]*n_components\\s*=",\n            source,\n            flags=re.DOTALL,\n        ):\n            raise RuntimeError(f"stale n_components resolver call remains: {path}")\n\n    projected = ast.parse(read(MODELS / "components" / "projected.py"))\n'''
if validate_anchor not in text:
    raise RuntimeError("projected resolver validator anchor not found")
text = text.replace(validate_anchor, validate_replacement, 1)

path.write_text(text, encoding="utf-8")
