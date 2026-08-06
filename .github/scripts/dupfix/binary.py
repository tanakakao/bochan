from __future__ import annotations

from .common import BINARY_PENDING_SECTION, ROOT, read, replace_once, replace_regex_once, write


def _insert_joint_duplicate_penalties(source: str) -> str:
    """Insert hard same-batch penalties while preserving each line's indentation."""

    marker = "out = self._apply_roi_weight_aggregated(out, mean_prob_x, Xt)"
    penalty = "out = out - self._same_batch_duplicate_penalty_per_point(Xt).sum(dim=-1)"
    lines = source.splitlines(keepends=True)
    updated: list[str] = []

    for index, line in enumerate(lines):
        updated.append(line)
        stripped = line.lstrip()
        if stripped.rstrip("\r\n") != marker:
            continue

        indent = line[: len(line) - len(stripped)]
        expected = f"{indent}{penalty}"
        if index + 1 < len(lines) and lines[index + 1].rstrip("\r\n") == expected:
            continue

        newline = "\r\n" if line.endswith("\r\n") else "\n"
        updated.append(f"{expected}{newline}")

    return "".join(updated)


def patch_binary() -> None:
    path = "src/bochan/acquisition/binary/base.py"
    text = read(path)
    text = replace_once(
        text,
        "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n",
        "from bochan.acquisition._duplicate_exclusion import (\n"
        "    hard_reference_duplicate_penalty_per_point,\n"
        "    hard_same_batch_duplicate_penalty_per_point,\n"
        ")\n"
        "from bochan.acquisition.binary._likelihood import latent_samples_to_binary_probabilities\n",
        label="binary imports",
    )
    text = replace_once(
        text,
        "        pending_penalty_weight: float = 0.0,\n"
        "        pending_penalty_beta: float = 10.0,\n"
        "        eps: float = 1e-6,\n",
        "        pending_penalty_weight: float = 0.0,\n"
        "        pending_penalty_beta: float = 10.0,\n"
        "        hard_duplicate_tol: float = 1e-8,\n"
        "        exclude_same_batch_duplicates: bool = True,\n"
        "        exclude_pending_duplicates: bool = True,\n"
        "        eps: float = 1e-6,\n",
        label="binary signature",
    )
    text = replace_once(
        text,
        "        self.pending_penalty_weight = float(pending_penalty_weight)\n"
        "        self.pending_penalty_beta = float(pending_penalty_beta)\n"
        "        self.eps = float(eps)\n",
        "        self.pending_penalty_weight = float(pending_penalty_weight)\n"
        "        self.pending_penalty_beta = float(pending_penalty_beta)\n"
        "        self.hard_duplicate_tol = float(hard_duplicate_tol)\n"
        "        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)\n"
        "        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)\n"
        "        if self.hard_duplicate_tol < 0.0:\n"
        "            raise ValueError(\"hard_duplicate_tol must be non-negative.\")\n"
        "        self.eps = float(eps)\n",
        label="binary attributes",
    )
    text = replace_regex_once(
        text,
        r"    # =========================================================\n    # pending penalty\n    # =========================================================\n.*?(?=    # =========================================================\n    # ROI helpers)",
        BINARY_PENDING_SECTION,
        label="binary pending section",
    )
    write(path, text)

    for target in (ROOT / "src/bochan/acquisition/binary").rglob("*.py"):
        if target == ROOT / path:
            continue
        source = target.read_text(encoding="utf-8")
        source = source.replace(
            "self._pending_penalty_per_point(",
            "self._candidate_penalty_per_point(",
        )
        source = source.replace(
            "self._pending_penalty_aggregated(",
            "self._candidate_penalty_aggregated(",
        )
        if target.name == "single_output.py" and "active_learning" in target.parts:
            source = _insert_joint_duplicate_penalties(source)
        target.write_text(source, encoding="utf-8")
