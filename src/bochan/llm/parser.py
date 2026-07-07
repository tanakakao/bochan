"""Parsing and validation utilities for LLM-generated candidates."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_payload(payload: Any) -> Any:
    """LLM 応答を JSON 互換 object に変換する。"""

    if isinstance(payload, (Mapping, list, tuple)):
        return payload
    if not isinstance(payload, str):
        raise TypeError(f"Expected str or JSON-like payload. Got {type(payload).__name__}.")
    return json.loads(_strip_json_fence(payload))


def _extract_x(item: Any) -> Any:
    if isinstance(item, Mapping):
        if "x" in item:
            return item["x"]
        if "candidate" in item:
            return item["candidate"]
    return item


def _candidate_rows(payload: Any) -> list[Any]:
    data = parse_json_payload(payload)
    if isinstance(data, Mapping):
        for key in ("candidates", "candidate_set", "points", "xs", "X"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, (list, tuple)):
        raise ValueError("LLM response must contain a list of candidates.")
    return [_extract_x(item) for item in data]


def candidates_to_tensor(
    payload: Any,
    *,
    bounds: Any,
    variable_names: Sequence[str] | None = None,
) -> Any:
    """LLM 応答または明示 candidate_set を ``[n, d]`` tensor に変換する。"""

    import torch

    if torch.is_tensor(payload):
        candidates = payload.to(dtype=bounds.dtype, device=bounds.device)
    else:
        rows = _candidate_rows(payload)
        values: list[list[float]] = []
        names = list(variable_names or [])
        for row in rows:
            if isinstance(row, Mapping):
                if not names:
                    raise ValueError("variable_names are required when candidates are returned as objects.")
                values.append([float(row[name]) for name in names])
            else:
                values.append([float(value) for value in row])
        candidates = torch.as_tensor(values, dtype=bounds.dtype, device=bounds.device)

    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    if candidates.ndim != 2:
        raise ValueError(f"Candidates must have shape [n, d]. Got {tuple(candidates.shape)}.")
    if candidates.shape[-1] != bounds.shape[-1]:
        raise ValueError(
            f"Candidate dimension mismatch. Expected d={bounds.shape[-1]}, got {candidates.shape[-1]}."
        )
    return candidates


def clip_to_bounds(candidates: Any, bounds: Any) -> Any:
    """候補を bounds 内に clip する。"""

    return candidates.clamp(min=bounds[0], max=bounds[1])


def remove_nonfinite(candidates: Any) -> Any:
    """NaN / inf を含む候補を取り除く。"""

    import torch

    if candidates.numel() == 0:
        return candidates
    mask = torch.isfinite(candidates).all(dim=-1)
    return candidates[mask]


def remove_duplicate_rows(candidates: Any, *, tolerance: float = 1e-9) -> Any:
    """重複候補を取り除く。"""

    import torch

    if candidates.shape[0] <= 1:
        return candidates
    kept = []
    for row in candidates:
        if not kept:
            kept.append(row)
            continue
        stack = torch.stack(kept, dim=0)
        if not torch.isclose(stack, row, atol=tolerance, rtol=0.0).all(dim=-1).any():
            kept.append(row)
    return torch.stack(kept, dim=0) if kept else candidates[:0]
