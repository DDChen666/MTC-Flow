from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.split import split_dataset
from ..mtc.dataset import DatasetFields
from ..mtc.labels import TaskLabelSchema
from ..mtc.training import TrainingResult, train_and_evaluate
from ..utils.config import ProjectConfig, TrainingSettings
from ..visualization.metrics import plot_confusion_matrix, plot_f1_bars
if TYPE_CHECKING:
    import optuna


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    config: ProjectConfig
    train_data_path: Optional[Path] = None
    test_data_path: Optional[Path] = None
    label_schema: TaskLabelSchema = field(init=False)
    dataset_fields: DatasetFields = field(init=False)
    last_training_result: Optional[TrainingResult] = None

    def __post_init__(self) -> None:
        self.label_schema = TaskLabelSchema.from_lists(
            self.config.labels.primary,
            self.config.labels.secondary,
        )
        self.dataset_fields = DatasetFields(
            text_field=self.config.data.text_field,
            primary_field=self.config.data.primary_field,
            secondary_field=self.config.data.secondary_field,
        )
        if self.train_data_path is None:
            self.train_data_path = self.config.data.train_path
        if self.test_data_path is None:
            self.test_data_path = self.config.data.test_path


def run_split_stage(context: PipelineContext) -> None:
    split_settings = context.config.data.split
    if not split_settings.enabled:
        LOGGER.info("Split stage disabled; skipping")
        return
    LOGGER.info("Running dataset split stage")
    train_path, test_path = split_dataset(context.config.data, split_settings)
    context.train_data_path = train_path
    context.test_data_path = test_path
    context.config.update_data_paths(train_path, test_path)
    LOGGER.info("Dataset split complete: train=%s test=%s", train_path, test_path)


def run_training_stage(context: PipelineContext) -> Optional[TrainingResult]:
    settings = context.config.training
    if not settings.enabled:
        LOGGER.info("Training stage disabled; skipping")
        return None

    if context.train_data_path is None or context.test_data_path is None:
        raise ValueError("Training and evaluation dataset paths must be provided before training stage")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = context.config.resolve_output_dir(settings.output_subdir, run_name)

    LOGGER.info("Starting training run %s", run_name)
    result = train_and_evaluate(
        train_path=context.train_data_path,
        test_path=context.test_data_path,
        dataset_fields=context.dataset_fields,
        label_schema=context.label_schema,
        training_settings=context.config.training,
        evaluation_settings=context.config.training.evaluation,
        seed=context.config.project.seed,
        output_dir=run_dir,
        trial_params=None,
        run_name=run_name,
    )

    context.last_training_result = result
    save_training_artifacts(result, context)
    LOGGER.info(
        "Training completed: run=%s macro_avg=%.4f primary_macro_f1=%.4f secondary_macro_f1=%.4f",
        result.run_name,
        result.macro_avg,
        result.evaluation.metrics["primary"]["macro_f1"],
        result.evaluation.metrics["secondary"]["macro_f1"],
    )
    return result


def save_training_artifacts(result: TrainingResult, context: PipelineContext) -> None:
    eval_config = context.config.training.evaluation
    run_dir = result.output_dir
    run_name = result.run_name
    metrics = result.evaluation.metrics

    if eval_config.save_metrics:
        metrics_path = run_dir / f"metrics_{run_name}.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        LOGGER.info("Saved metrics to %s", metrics_path)

    if eval_config.save_confusion:
        primary_matrix = np.array(metrics["primary"]["confusion_matrix"], dtype=int)
        secondary_matrix = np.array(metrics["secondary"]["confusion_matrix"], dtype=int)
        primary_labels = context.label_schema.primary.labels
        secondary_labels = context.label_schema.secondary.labels

        primary_df = pd.DataFrame(primary_matrix, index=primary_labels, columns=primary_labels)
        primary_path = run_dir / f"confusion_primary_{run_name}.csv"
        primary_df.to_csv(primary_path, encoding="utf-8-sig")

        secondary_df = pd.DataFrame(secondary_matrix, index=secondary_labels, columns=secondary_labels)
        secondary_path = run_dir / f"confusion_secondary_{run_name}.csv"
        secondary_df.to_csv(secondary_path, encoding="utf-8-sig")

        LOGGER.info("Saved confusion matrices to %s and %s", primary_path, secondary_path)

    if eval_config.save_predictions:
        preds_path = run_dir / f"predictions_{run_name}.csv"
        primary_id2label = context.label_schema.primary.id2label
        secondary_id2label = context.label_schema.secondary.id2label
        df_preds = pd.DataFrame({
            "gold_primary_id": result.evaluation.gold_primary,
            "gold_primary": [primary_id2label[i] for i in result.evaluation.gold_primary],
            "pred_primary_id": result.evaluation.pred_primary,
            "pred_primary": [primary_id2label[i] for i in result.evaluation.pred_primary],
            "gold_secondary_id": result.evaluation.gold_secondary,
            "gold_secondary": [secondary_id2label[i] for i in result.evaluation.gold_secondary],
            "pred_secondary_id": result.evaluation.pred_secondary,
            "pred_secondary": [secondary_id2label[i] for i in result.evaluation.pred_secondary],
        })
        df_preds.to_csv(preds_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved predictions to %s", preds_path)

    if eval_config.save_label_map:
        label_map_path = run_dir / "label_map.json"
        with label_map_path.open("w", encoding="utf-8") as fh:
            json.dump(context.label_schema.to_serializable(), fh, ensure_ascii=False, indent=2)
        LOGGER.info("Saved label map to %s", label_map_path)

    if eval_config.generate_visuals:
        metrics_primary = metrics["primary"]
        metrics_secondary = metrics["secondary"]
        plot_confusion_matrix(
            metrics_primary["confusion_matrix"],
            context.label_schema.primary.labels,
            "Primary Confusion Matrix",
            run_dir / f"confusion_primary_{run_name}.png",
        )
        plot_confusion_matrix(
            metrics_secondary["confusion_matrix"],
            context.label_schema.secondary.labels,
            "Secondary Confusion Matrix",
            run_dir / f"confusion_secondary_{run_name}.png",
        )
        plot_f1_bars(
            metrics_primary["report"],
            context.label_schema.primary.labels,
            "Primary Class F1-score",
            run_dir / f"f1_primary_{run_name}.png",
        )
        plot_f1_bars(
            metrics_secondary["report"],
            context.label_schema.secondary.labels,
            "Secondary Class F1-score",
            run_dir / f"f1_secondary_{run_name}.png",
        )


if TYPE_CHECKING:
    import optuna

def run_tuning_stage(context: PipelineContext) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('Optuna is required for the tuning stage. Install it or disable the stage in the config.') from exc

    tuning_cfg = context.config.tuning
    if not tuning_cfg.enabled:
        LOGGER.info("Tuning stage disabled; skipping")
        return
    if context.train_data_path is None or context.test_data_path is None:
        raise ValueError("Training and evaluation dataset paths must be set before tuning stage")

    tuning_root = context.config.resolve_output_dir(tuning_cfg.output_subdir)
    LOGGER.info("Starting hyperparameter tuning: trials=%d", tuning_cfg.trials)

    trial_records: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        sampled_params = _sample_search_space(trial, tuning_cfg.search_space)
        normalized_params = _normalize_params(sampled_params, context.config.training)
        trial_dir = context.config.resolve_output_dir(
            tuning_cfg.output_subdir,
            f"trial_{trial.number:03d}",
        )

        trial_settings = replace(context.config.training, **normalized_params)
        result = train_and_evaluate(
            train_path=context.train_data_path,
            test_path=context.test_data_path,
            dataset_fields=context.dataset_fields,
            label_schema=context.label_schema,
            training_settings=trial_settings,
            evaluation_settings=trial_settings.evaluation,
            seed=context.config.project.seed,
            output_dir=trial_dir,
            trial_params=normalized_params,
            run_name=f"trial_{trial.number:03d}",
        )
        save_training_artifacts(result, context)

        primary_macro = result.evaluation.metrics["primary"]["macro_f1"]
        secondary_macro = result.evaluation.metrics["secondary"]["macro_f1"]
        trial.set_user_attr("primary_macro_f1", primary_macro)
        trial.set_user_attr("secondary_macro_f1", secondary_macro)

        record = {
            "number": trial.number,
            "macro_avg": result.macro_avg,
            "primary_macro_f1": primary_macro,
            "secondary_macro_f1": secondary_macro,
        }
        record.update(normalized_params)
        trial_records.append(record)
        return result.macro_avg

    study = optuna.create_study(direction=tuning_cfg.direction)
    study.optimize(objective, n_trials=tuning_cfg.trials)

    best = {
        "best_value_macro_avg": study.best_value,
        "best_params": study.best_params,
        "best_primary_macro_f1": study.best_trial.user_attrs.get("primary_macro_f1"),
        "best_secondary_macro_f1": study.best_trial.user_attrs.get("secondary_macro_f1"),
        "trials": len(study.trials),
    }

    best_path = tuning_root / "optuna_best.json"
    with best_path.open("w", encoding="utf-8") as fh:
        json.dump(best, fh, ensure_ascii=False, indent=2)

    df_trials = pd.DataFrame(trial_records)
    trials_path = tuning_root / "optuna_trials.csv"
    df_trials.to_csv(trials_path, index=False, encoding="utf-8-sig")

    LOGGER.info("Tuning completed. Best macro_avg=%.4f", study.best_value)


def run_stability_stage(context: PipelineContext) -> None:
    stability_cfg = context.config.stability
    if not stability_cfg.enabled:
        LOGGER.info("Stability stage disabled; skipping")
        return
    if context.train_data_path is None or context.test_data_path is None:
        raise ValueError("Training and evaluation dataset paths must be set before stability stage")

    stability_root = context.config.resolve_output_dir(stability_cfg.output_subdir)
    LOGGER.info("Evaluating training stability for %d runs", stability_cfg.runs)

    rows: List[Dict[str, Any]] = []

    for run_idx in range(stability_cfg.runs):
        seed = stability_cfg.base_seed + run_idx
        run_name = f"run_{run_idx + 1:02d}"
        run_dir = context.config.resolve_output_dir(stability_cfg.output_subdir, run_name)
        result = train_and_evaluate(
            train_path=context.train_data_path,
            test_path=context.test_data_path,
            dataset_fields=context.dataset_fields,
            label_schema=context.label_schema,
            training_settings=context.config.training,
            evaluation_settings=context.config.training.evaluation,
            seed=seed,
            output_dir=run_dir,
            trial_params={"seed": seed},
            run_name=run_name,
        )
        save_training_artifacts(result, context)
        rows.append({
            "run": run_idx + 1,
            "seed": seed,
            "macro_avg": result.macro_avg,
            "primary_macro_f1": result.evaluation.metrics["primary"]["macro_f1"],
            "secondary_macro_f1": result.evaluation.metrics["secondary"]["macro_f1"],
        })

    df = pd.DataFrame(rows)
    summary = {
        "macro_avg_mean": float(df["macro_avg"].mean()),
        "macro_avg_std": float(df["macro_avg"].std(ddof=1) if len(df) > 1 else 0.0),
        "primary_macro_f1_mean": float(df["primary_macro_f1"].mean()),
        "primary_macro_f1_std": float(df["primary_macro_f1"].std(ddof=1) if len(df) > 1 else 0.0),
        "secondary_macro_f1_mean": float(df["secondary_macro_f1"].mean()),
        "secondary_macro_f1_std": float(df["secondary_macro_f1"].std(ddof=1) if len(df) > 1 else 0.0),
        "runs": len(df),
    }

    runs_path = stability_root / "stability_runs.csv"
    df.to_csv(runs_path, index=False, encoding="utf-8-sig")
    summary_path = stability_root / "stability_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    LOGGER.info("Stability evaluation complete. Results written to %s", stability_root)


def _sample_search_space(trial: "optuna.trial.Trial", search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        dist = spec.get("distribution", "categorical")
        if dist == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif dist == "uniform":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif dist == "loguniform":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        elif dist in {"int", "int_uniform"}:
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif dist in {"int_loguniform"}:
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)), log=True)
        else:
            raise ValueError(f"Unsupported distribution type: {dist}")
    return params


def _normalize_params(params: Dict[str, Any], base_settings: TrainingSettings) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in params.items():
        if not hasattr(base_settings, key):
            normalized[key] = value
            continue
        current = getattr(base_settings, key)
        if isinstance(current, bool):
            normalized[key] = bool(value)
        elif isinstance(current, int):
            normalized[key] = int(value)
        elif isinstance(current, float):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized




