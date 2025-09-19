from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .pipeline.stages import (
    PipelineContext,
    run_split_stage,
    run_stability_stage,
    run_training_stage,
    run_tuning_stage,
)
from .utils.config import load_project_config
from .utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)

STAGE_HANDLERS = {
    "split": run_split_stage,
    "train": run_training_stage,
    "tune": run_tuning_stage,
    "stability": run_stability_stage,
}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified pipeline entrypoint for multi-task classification")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGE_HANDLERS.keys()),
        help="Optional list of pipeline stages to run (overrides config order)",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available stages and exit",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.list_stages:
        print("Available stages:")
        for name in STAGE_HANDLERS:
            print(f" - {name}")
        return 0

    config = load_project_config(args.config)
    config.project.output_root.mkdir(parents=True, exist_ok=True)
    configure_logging(config.project.logging)

    LOGGER.info("Loaded configuration from %s", config.config_path)

    stages = args.stages or config.pipeline.stages
    invalid_stages = [stage for stage in stages if stage not in STAGE_HANDLERS]
    if invalid_stages:
        raise ValueError(f"Unknown stages requested: {invalid_stages}")

    LOGGER.info("Executing stages: %s", ", ".join(stages))
    context = PipelineContext(config=config)

    for stage in stages:
        handler = STAGE_HANDLERS[stage]
        LOGGER.info("--- Stage: %s ---", stage)
        handler(context)

    LOGGER.info("Pipeline execution completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


