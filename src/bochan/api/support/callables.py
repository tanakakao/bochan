"""Callable signature helpers shared by API responsibility modules."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def _has_var_keyword(signature: inspect.Signature) -> bool:
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def _filter_kwargs_for_callable(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if _has_var_keyword(signature):
        return kwargs
    allowed = set(signature.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


