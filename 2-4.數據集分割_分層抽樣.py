"""Compatibility wrapper that delegates dataset splitting to the unified pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.main import main as pipeline_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data splitting via the unified pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Ensure the split stage is enabled before running",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.enable:
        print("Warning: data.split.enabled is controlled via the config file. Pass --enable after updating the config if needed.")
    return pipeline_main(["--config", str(args.config), "--stages", "split"])


if __name__ == "__main__":
    raise SystemExit(main())
