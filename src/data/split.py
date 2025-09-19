from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.io import load_records, write_records
from ..utils.config import DataSettings, SplitSettings

LOGGER = logging.getLogger(__name__)


def split_dataset(data_settings: DataSettings, split_settings: SplitSettings) -> Tuple[Path, Path]:
    if not split_settings.enabled:
        raise ValueError("SplitSettings.enabled must be true to perform dataset split")
    if split_settings.input_path is None:
        raise ValueError("SplitSettings.input_path must be provided when splitting is enabled")

    input_path = split_settings.input_path
    LOGGER.info("Loading raw dataset from %s", input_path)
    rows = load_records(input_path)
    if not rows:
        raise ValueError(f"No records loaded from {input_path}")

    df = pd.DataFrame(rows)
    if data_settings.primary_field not in df.columns:
        raise KeyError(f"Primary label field '{data_settings.primary_field}' missing from dataset")

    stratify_series = df[data_settings.primary_field]
    if stratify_series.isnull().any():
        raise ValueError("Primary label field contains null values; cannot perform stratified split")

    test_size = split_settings.test_size
    if not 0 < test_size < 1:
        raise ValueError("split_settings.test_size must be between 0 and 1")

    LOGGER.info("Performing stratified split with test_size=%s", test_size)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=split_settings.random_seed,
        stratify=stratify_series,
        shuffle=True,
    )

    timestamp = datetime.now().strftime(split_settings.timestamp_format)
    output_root = split_settings.output_root or data_settings.root_dir
    train_name = split_settings.train_template.format(timestamp=timestamp)
    test_name = split_settings.test_template.format(timestamp=timestamp)
    train_path = (output_root / train_name) if isinstance(output_root, Path) else Path(output_root) / train_name
    test_path = (output_root / test_name) if isinstance(output_root, Path) else Path(output_root) / test_name

    LOGGER.info("Saving %s training rows to %s", len(train_df), train_path)
    write_records(train_path, train_df.to_dict(orient="records"))
    LOGGER.info("Saving %s evaluation rows to %s", len(test_df), test_path)
    write_records(test_path, test_df.to_dict(orient="records"))

    return train_path, test_path
