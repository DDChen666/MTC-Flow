from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import __version__ as transformers_version

from ..data.io import load_records
from ..utils.config import EvaluationSettings, TrainingSettings
from .dataset import DatasetFields, ReviewDataset
from .labels import TaskLabelSchema
from .modeling import MultiTaskClassifier

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationOutput:
    metrics: Dict[str, Any]
    gold_primary: List[int]
    pred_primary: List[int]
    gold_secondary: List[int]
    pred_secondary: List[int]


@dataclass
class TrainingResult:
    run_name: str
    output_dir: Path
    evaluation: EvaluationOutput
    trial_params: Optional[Dict[str, Any]] = None

    @property
    def macro_avg(self) -> float:
        primary = self.evaluation.metrics["primary"]["macro_f1"]
        secondary = self.evaluation.metrics["secondary"]["macro_f1"]
        return 0.5 * (primary + secondary)


def compute_class_weights(int_labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(int_labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = 1.0 / np.log1p(counts + 1.0)
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


class MTTrainer(Trainer):
    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any):  # type: ignore[override]
        labels = inputs.get("labels")
        labels_secondary = inputs.get("labels_secondary")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=labels,
            labels_secondary=labels_secondary,
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def evaluate_dataset(
    model: nn.Module,
    tokenizer,
    dataset: ReviewDataset,
    batch_size: int,
    device: torch.device,
    label_schema: TaskLabelSchema,
) -> EvaluationOutput:
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    all_primary: List[np.ndarray] = []
    all_secondary: List[np.ndarray] = []
    gold_primary: List[np.ndarray] = []
    gold_secondary: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            outputs = model(
                input_ids=batch_tensors.get("input_ids"),
                attention_mask=batch_tensors.get("attention_mask"),
                token_type_ids=batch_tensors.get("token_type_ids"),
                labels=None,
                labels_secondary=None,
            )
            all_primary.append(outputs["logits_primary"].cpu().numpy())
            all_secondary.append(outputs["logits_secondary"].cpu().numpy())
            gold_primary.append(batch_tensors["labels"].cpu().numpy())
            gold_secondary.append(batch_tensors["labels_secondary"].cpu().numpy())

    logits_primary = np.concatenate(all_primary, axis=0)
    logits_secondary = np.concatenate(all_secondary, axis=0)
    gold_primary_arr = np.concatenate(gold_primary, axis=0)
    gold_secondary_arr = np.concatenate(gold_secondary, axis=0)
    pred_primary_arr = np.argmax(logits_primary, axis=-1)
    pred_secondary_arr = np.argmax(logits_secondary, axis=-1)

    primary_report = _build_report(gold_primary_arr, pred_primary_arr, label_schema.primary.labels)
    secondary_report = _build_report(gold_secondary_arr, pred_secondary_arr, label_schema.secondary.labels)

    metrics = {
        "primary": primary_report,
        "secondary": secondary_report,
    }

    return EvaluationOutput(
        metrics=metrics,
        gold_primary=gold_primary_arr.tolist(),
        pred_primary=pred_primary_arr.tolist(),
        gold_secondary=gold_secondary_arr.tolist(),
        pred_secondary=pred_secondary_arr.tolist(),
    )


def _build_report(gold: np.ndarray, pred: np.ndarray, labels_list: List[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    label_indices = list(range(len(labels_list)))
    accuracy = float(accuracy_score(gold, pred))
    macro_f1 = float(f1_score(gold, pred, average="macro"))
    weighted_f1 = float(f1_score(gold, pred, average="weighted"))
    report = classification_report(
        gold,
        pred,
        labels=label_indices,
        target_names=labels_list,
        digits=4,
        output_dict=True,
    )
    matrix = confusion_matrix(gold, pred, labels=label_indices).tolist()
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": matrix,
        "labels": labels_list,
    }


def train_and_evaluate(
    train_path: Path,
    test_path: Path,
    dataset_fields: DatasetFields,
    label_schema: TaskLabelSchema,
    training_settings: TrainingSettings,
    evaluation_settings: EvaluationSettings,
    seed: int,
    output_dir: Path,
    trial_params: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
) -> TrainingResult:
    LOGGER.info("Starting training using transformers %s", transformers_version)
    LOGGER.info("Train data: %s", train_path)
    LOGGER.info("Test data: %s", test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {test_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_records(train_path)
    test_rows = load_records(test_path)
    if not train_rows or not test_rows:
        raise ValueError("Training or evaluation dataset is empty")

    tokenizer = AutoTokenizer.from_pretrained(training_settings.model_name, use_fast=True)

    train_dataset = ReviewDataset(train_rows, tokenizer, training_settings.max_length, dataset_fields, label_schema)
    eval_dataset = ReviewDataset(test_rows, tokenizer, training_settings.max_length, dataset_fields, label_schema)

    if len(train_dataset) == 0:
        raise ValueError("No valid training examples after filtering")
    if len(eval_dataset) == 0:
        raise ValueError("No valid evaluation examples after filtering")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    primary_labels = [label_schema.primary.label2id[item["primary"]] for item in train_dataset.items]
    secondary_labels = [label_schema.secondary.label2id[item["secondary"]] for item in train_dataset.items]
    primary_weight = compute_class_weights(primary_labels, len(label_schema.primary.labels)) if training_settings.weight_primary else None
    secondary_weight = compute_class_weights(secondary_labels, len(label_schema.secondary.labels)) if training_settings.weight_secondary else None

    model = MultiTaskClassifier(
        model_name=training_settings.model_name,
        num_primary=len(label_schema.primary.labels),
        num_secondary=len(label_schema.secondary.labels),
        primary_weight=primary_weight,
        secondary_weight=secondary_weight,
        use_focal=training_settings.use_focal,
        focal_gamma=training_settings.focal_gamma,
        dropout=training_settings.dropout,
        alpha=training_settings.alpha,
        beta=training_settings.beta,
    ).to(device)

    collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=training_settings.batch_size,
        per_device_eval_batch_size=training_settings.eval_batch_size,
        learning_rate=training_settings.learning_rate,
        num_train_epochs=training_settings.epochs,
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available() and training_settings.fp16_auto,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)]

    trainer = MTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    if training_settings.weighted_sampler:
        class_counts = np.bincount(primary_labels, minlength=len(label_schema.primary.labels)).astype(float)
        inv_freq = 1.0 / np.maximum(class_counts, 1.0)
        sample_weights = np.array([inv_freq[label_schema.primary.label2id[item["primary"]]] for item in train_dataset.items], dtype=np.float64)
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)

        def _train_dataloader() -> DataLoader:
            return DataLoader(train_dataset, batch_size=training_settings.batch_size, sampler=sampler, collate_fn=collator)

        trainer.get_train_dataloader = _train_dataloader  # type: ignore[assignment]

    trainer.train()

    evaluation = evaluate_dataset(trainer.model, tokenizer, eval_dataset, training_settings.eval_batch_size, device, label_schema)

    if trial_params:
        evaluation.metrics["trial_params"] = trial_params

    stamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGGER.info("Training completed; macro_avg=%.4f", 0.5 * (
        evaluation.metrics["primary"]["macro_f1"] + evaluation.metrics["secondary"]["macro_f1"]
    ))

    return TrainingResult(
        run_name=stamp,
        output_dir=output_dir,
        evaluation=evaluation,
        trial_params=trial_params,
    )






