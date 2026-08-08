from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
file_path = ROOT / "src/bochan/serving/webapp/risk_settings.py"
text = file_path.read_text(encoding="utf-8")
old = '''from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Iterator
'''
new = '''from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any
'''
if old in text:
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise RuntimeError("Expected risk_settings import block was not found")

print("Web LSE lint import fixed")
