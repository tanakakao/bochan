"""Study data builders for visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..utils import ensure_2d, infer_target_cols


def study_target_dataframe(
    study: Any,
    *,
    target: str,
    target_cols: Sequence[str] | None = None,
    cycle_col: str = "cycle",
) -> pd.DataFrame:
    """完了 trial から cycle-target DataFrame を作る。"""

    trials = study.completed_trials()
    if not trials:
        return pd.DataFrame(columns=[cycle_col, target])

    targets = ensure_2d([trial.y for trial in trials])
    columns = infer_target_cols(study, target_cols, targets.shape[1])
    if target not in columns:
        raise ValueError(f"target must be one of {columns}.")

    target_index = columns.index(target)
    cycles = [
        trial.metadata.get(cycle_col, index)
        for index, trial in enumerate(trials)
    ]
    return pd.DataFrame({cycle_col: cycles, target: targets[:, target_index]})


__all__ = ["study_target_dataframe"]
