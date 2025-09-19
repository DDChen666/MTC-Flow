from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

from ..data.io import load_records
from ..mtc.dataset import DatasetFields, ReviewDataset
from ..mtc.labels import TaskLabelSchema
from ..mtc.modeling import MultiTaskClassifier
from ..mtc.training import EvaluationOutput, evaluate_dataset


@dataclass
class InferenceArtifacts:
    model: MultiTaskClassifier
    tokenizer: AutoTokenizer
    label_schema: TaskLabelSchema
    dataset_fields: DatasetFields
    device: torch.device


def load_label_schema(model_dir: Path) -> TaskLabelSchema:
    label_map_path = model_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing label_map.json in {model_dir}")
    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    primary = payload["PRIMARY_LABELS"]
    secondary = payload["SECONDARY_LABELS"]
    return TaskLabelSchema.from_lists(primary, secondary)


def load_model(
    model_dir: Path,
    base_model_name: str,
    label_schema: TaskLabelSchema,
    device: torch.device,
    dropout: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    focal_gamma: float = 1.5,
) -> MultiTaskClassifier:
    model_path = model_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find pytorch_model.bin in {model_dir}")

    model = MultiTaskClassifier(
        model_name=base_model_name,
        num_primary=len(label_schema.primary.labels),
        num_secondary=len(label_schema.secondary.labels),
        primary_weight=None,
        secondary_weight=None,
        use_focal=False,
        focal_gamma=focal_gamma,
        dropout=dropout,
        alpha=alpha,
        beta=beta,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_inference_artifacts(
    model_dir: Path,
    base_model_name: str,
    text_field: str,
    primary_field: str,
    secondary_field: str,
    device: torch.device | None = None,
) -> InferenceArtifacts:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_schema = load_label_schema(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    dataset_fields = DatasetFields(
        text_field=text_field,
        primary_field=primary_field,
        secondary_field=secondary_field,
    )
    model = load_model(
        model_dir=model_dir,
        base_model_name=base_model_name,
        label_schema=label_schema,
        device=device,
    )
    return InferenceArtifacts(
        model=model,
        tokenizer=tokenizer,
        label_schema=label_schema,
        dataset_fields=dataset_fields,
        device=device,
    )


def load_uploaded_records(data_path: Path) -> List[dict]:
    suffix = data_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_records(data_path)
    if suffix == ".csv":
        df = pd.read_csv(data_path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported file format: {suffix}")


def run_evaluation(
    artifacts: InferenceArtifacts,
    rows: Iterable[dict],
    max_length: int,
    batch_size: int,
) -> Tuple[EvaluationOutput, ReviewDataset]:
    records = list(rows)
    dataset = ReviewDataset(
        rows=records,
        tokenizer=artifacts.tokenizer,
        max_length=max_length,
        fields=artifacts.dataset_fields,
        label_schema=artifacts.label_schema,
    )
    if len(dataset) == 0:
        raise ValueError("No valid rows to evaluate. Check that labels exist and match the label map.")
    evaluation = evaluate_dataset(
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        dataset=dataset,
        batch_size=batch_size,
        device=artifacts.device,
        label_schema=artifacts.label_schema,
    )
    return evaluation, dataset
