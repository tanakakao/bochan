"""Command-line entrypoint for provisioning TabPFN Web runtime checkpoints.

This module intentionally lives outside ``bochan.serving.webapp`` so invoking
``python -m bochan.tabpfn_preload`` does not initialize the Web application or
import the wider BoTorch model stack.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from bochan.tabpfn_assets import preload_tabpfn_assets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the TabPFN v3 classifier/regressor checkpoints required by "
            "the bochan Web runtime."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory. When omitted, TabPFN's configured cache directory "
            "is used (including TABPFN_MODEL_CACHE_DIR)."
        ),
    )
    parser.add_argument(
        "--allow-browser-auth",
        action="store_true",
        help=(
            "Allow TabPFN to open its interactive Prior Labs browser login flow. "
            "Deployment/CI jobs should instead provide TABPFN_TOKEN as a secret."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Provision all TabPFN checkpoints required by bochan Web."""

    args = _parser().parse_args(argv)
    status = preload_tabpfn_assets(
        args.cache_dir,
        allow_browser_auth=bool(args.allow_browser_auth),
    )
    print(f"TabPFN Web assets ready: {status['cache_dir']}")
    for which, model in status["models"].items():
        print(f"  {which}: {model['filename']}")
    print(
        "Configure TABPFN_MODEL_CACHE_DIR to this same persistent directory "
        "for the bochan Web runtime."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI invocation
    raise SystemExit(main())
