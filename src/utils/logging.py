from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml

from .config import LoggingSettings

LOGGER = logging.getLogger(__name__)


def _load_yaml_config(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        LOGGER.warning("Logging configuration file not found: %%s", path)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Failed to read logging configuration %%s: %%s", path, exc)
    return None


def configure_logging(settings: LoggingSettings) -> None:
    if settings.config:
        config_data = _load_yaml_config(settings.config)
        if config_data:
            logging.config.dictConfig(config_data)
            LOGGER.debug("Applied logging configuration from %%s", settings.config)
            return

    level_name = settings.level.upper() if settings.level else "INFO"
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.debug("Initialized basic logging at level %%s", level_name)
