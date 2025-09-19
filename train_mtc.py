"""Compatibility wrapper for legacy training entry point.

This script delegates to the new config-driven pipeline implemented in src.main.
Use `python -m src.main --stages train` directly for full control.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.main import main as pipeline_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-task training via the unified pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return pipeline_main(["--config", str(args.config), "--stages", "train"])


if __name__ == "__main__":
    raise SystemExit(main())
