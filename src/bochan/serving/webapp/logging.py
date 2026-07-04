"""Structured logging utilities for the bochan web application."""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_REQUEST_ID: ContextVar[str | None] = ContextVar("bochan_web_request_id", default=None)
_LOG_PATH: Path | None = None

_STANDARD_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLogFormatter(logging.Formatter):
    """Render each log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or _REQUEST_ID.get()
        if request_id:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key.startswith("_"):
                continue
            if key == "request_id" and request_id:
                continue
            payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ConsoleLogFormatter(logging.Formatter):
    """Human-readable formatter for local development output."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None) or _REQUEST_ID.get() or "-"
        event = getattr(record, "event", "log")
        base = (
            f"{datetime.fromtimestamp(record.created).astimezone().isoformat(timespec='seconds')} "
            f"{record.levelname:<8} [{request_id}] {event}: {record.getMessage()}"
        )
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(
    *,
    log_dir: str | Path | None = None,
    level: str | int | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> Path:
    """Configure console and rotating JSONL file handlers.

    Environment variables:
        BOCHAN_LOG_LEVEL: Logging level. Default is ``INFO``.
        BOCHAN_LOG_DIR: Directory for log files. Default is ``.bochan/logs``.
        BOCHAN_LOG_MAX_BYTES: Maximum size of one log file. Default is 10 MiB.
        BOCHAN_LOG_BACKUP_COUNT: Number of rotated files retained. Default is 5.
    """

    global _LOG_PATH

    resolved_level = level or os.getenv("BOCHAN_LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level.upper(), logging.INFO)

    resolved_dir = Path(log_dir or os.getenv("BOCHAN_LOG_DIR", ".bochan/logs"))
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_dir / "bochan-web.jsonl"

    logger = logging.getLogger("bochan.web")
    logger.setLevel(resolved_level)
    logger.propagate = False

    if not getattr(logger, "_bochan_configured", False):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(ConsoleLogFormatter())

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes or int(os.getenv("BOCHAN_LOG_MAX_BYTES", str(10 * 1024 * 1024))),
            backupCount=backup_count or int(os.getenv("BOCHAN_LOG_BACKUP_COUNT", "5")),
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(JsonLogFormatter())

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        setattr(logger, "_bochan_configured", True)
    else:
        for handler in logger.handlers:
            handler.setLevel(resolved_level)

    _LOG_PATH = log_path
    return log_path


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger within the ``bochan.web`` namespace."""

    return logging.getLogger("bochan.web" if not name else f"bochan.web.{name}")


def set_request_id(request_id: str) -> Token[str | None]:
    """Bind a request ID to the current execution context."""

    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous request ID context."""

    _REQUEST_ID.reset(token)


def current_request_id() -> str | None:
    """Return the request ID bound to the current context."""

    return _REQUEST_ID.get()


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    """Write one structured event without requiring callers to build ``extra``."""

    logger.log(level, message, extra={"event": event, **fields})


def read_recent_logs(
    *,
    limit: int = 200,
    level: str | None = None,
    event: str | None = None,
    request_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read and filter the most recent records from the active JSONL file."""

    path = _LOG_PATH or configure_logging()
    bounded_limit = min(max(int(limit), 1), 1000)
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: list[dict[str, Any]] = []
    target_level = level.upper() if level else None

    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if target_level and str(entry.get("level", "")).upper() != target_level:
            continue
        if event and entry.get("event") != event:
            continue
        if request_id and entry.get("request_id") != request_id:
            continue
        entries.append(entry)
        if len(entries) >= bounded_limit:
            break
    entries.reverse()
    return entries


def log_file_path() -> Path:
    """Return the configured active log file path."""

    return _LOG_PATH or configure_logging()


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


__all__ = [
    "JsonLogFormatter",
    "configure_logging",
    "current_request_id",
    "get_logger",
    "log_event",
    "log_file_path",
    "read_recent_logs",
    "reset_request_id",
    "set_request_id",
]
