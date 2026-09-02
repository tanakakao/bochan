from pathlib import Path

path = Path("src/bochan/models/components/layers/kernel_layers.py")
text = path.read_text(encoding="utf-8")

class_marker = "class _PartialObservationMultitaskKernel(MultitaskKernel):"
if class_marker not in text:
    raise SystemExit("partial-observation multitask kernel class not found")

marker = "\n\n\nclass DeepKernel(ExactGP):"
if marker not in text:
    raise SystemExit("DeepKernel class boundary not found")

compatibility = '''\n\n# FastAPI material metadata historically exposes the semantic GPyTorch kernel\n# name via ``covar_module.__class__.__name__``. Keep that public contract stable\n# while retaining the Phase 6 subclass internally for masked exact prediction.\n_PartialObservationMultitaskKernel.__name__ = "MultitaskKernel"\n'''

if '_PartialObservationMultitaskKernel.__name__ = "MultitaskKernel"' not in text:
    text = text.replace(marker, compatibility + "\n\nclass DeepKernel(ExactGP):", 1)

path.write_text(text, encoding="utf-8")
