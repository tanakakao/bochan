from pathlib import Path

path = Path("src/bochan/api/observation/state.py")
text = path.read_text(encoding="utf-8")
old = '''            if not torch.is_floating_point(Yvar):
                Yvar = Yvar.to(dtype=Y.dtype)
            else:
                Yvar = Yvar.to(dtype=Y.dtype)
'''
new = '''            Yvar = Yvar.to(dtype=Y.dtype)
'''
if old not in text:
    raise RuntimeError("Expected Phase 4 Yvar dtype block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
