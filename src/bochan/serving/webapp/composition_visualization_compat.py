"""Temporary import shim for the remaining multielement ternary adapter.

Composition visualization routing itself is now explicit and no longer installed
through this module. Remove this shim together with the multielement ternary
runtime adapter in the follow-up cleanup.
"""

from __future__ import annotations

from .composition_visualization_dispatch import _unavailable_payload

__all__ = ["_unavailable_payload"]
