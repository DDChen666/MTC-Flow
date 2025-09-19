from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


def load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_records(path: Path, rows: Sequence[Dict[str, Any]], indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=indent)


def write_dataframe_as_json(path: Path, df: pd.DataFrame, indent: int = 2) -> None:
    records = df.to_dict(orient="records")
    write_records(path, records, indent=indent)


def write_dataframe(path: Path, df: pd.DataFrame, *, fmt: str = "csv", **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False, encoding=kwargs.pop("encoding", "utf-8-sig"), **kwargs)
    elif fmt == "json":
        df.to_json(path, orient="records", force_ascii=False, indent=kwargs.pop("indent", 2), **kwargs)
    else:
        raise ValueError(f"Unsupported dataframe format: {fmt}")
