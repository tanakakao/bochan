from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "src/bochan/models/components/robust.py"

ORIGINAL = '''    @property
    def dense_delta(self) -> Tensor:
        dense = torch.zeros(self.dim, dtype=self.raw_delta.dtype, device=self.raw_delta.device)
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense
'''

HIDDEN_ORDINAL = '''    @property
    def dense_delta(self) -> Tensor:
        # Temporary migration marker for the ordinal implementation.
        dense = torch.zeros(self.dim, dtype=self.raw_delta.dtype, device=self.raw_delta.device)
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense
'''


def hide_ordinal() -> None:
    text = PATH.read_text(encoding="utf-8")
    class_pos = text.index("class SparseOutlierOrdinalLogitLikelihood")
    before = text[:class_pos]
    ordinal = text[class_pos:]
    if ordinal.count(ORIGINAL) != 1:
        raise RuntimeError(
            "Expected one ordinal dense_delta implementation, "
            f"found {ordinal.count(ORIGINAL)}."
        )
    PATH.write_text(
        before + ordinal.replace(ORIGINAL, HIDDEN_ORDINAL, 1),
        encoding="utf-8",
    )


def restore_ordinal() -> None:
    text = PATH.read_text(encoding="utf-8")
    if text.count(HIDDEN_ORDINAL) != 1:
        raise RuntimeError(
            "Expected one temporary ordinal marker, "
            f"found {text.count(HIDDEN_ORDINAL)}."
        )
    PATH.write_text(text.replace(HIDDEN_ORDINAL, ORIGINAL, 1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hide", "restore"])
    args = parser.parse_args()
    hide_ordinal() if args.mode == "hide" else restore_ordinal()
