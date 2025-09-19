"""Compatibility wrapper for legacy Optuna tuning entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.main import main as pipeline_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna tuning via the unified pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return pipeline_main(["--config", str(args.config), "--stages", "tune"])


if __name__ == "__main__":
    raise SystemExit(main())
