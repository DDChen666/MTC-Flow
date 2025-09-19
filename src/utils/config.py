from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass
class LoggingSettings:
    config: Optional[Path] = None
    level: str = "INFO"


@dataclass
class ProjectSettings:
    seed: int = 42
    output_root: Path = Path("artifacts")
    logging: LoggingSettings = field(default_factory=LoggingSettings)


@dataclass
class PipelineSettings:
    stages: List[str] = field(default_factory=lambda: ["split", "train", "tune", "stability"])


@dataclass
class SplitSettings:
    enabled: bool = False
    input_path: Optional[Path] = None
    output_root: Optional[Path] = None
    train_template: str = "train_set_{timestamp}.json"
    test_template: str = "test_set_{timestamp}.json"
    test_size: float = 0.2
    stratify_key: str = "primary"
    random_seed: int = 42
    timestamp_format: str = "%Y%m%d_%H%M%S"


@dataclass
class DataSettings:
    root_dir: Path = Path("input_train_and_test_json")
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    text_field: str = "text"
    primary_field: str = "primary"
    secondary_field: str = "secondary"
    split: SplitSettings = field(default_factory=SplitSettings)


@dataclass
class LabelSettings:
    primary: List[str] = field(default_factory=lambda: [
        "BUG",
        "UI/UX",
        "FEATURE_REQUEST",
        "PERFORMANCE",
        "POSITIVE",
        "INVALID",
    ])
    secondary: List[str] = field(default_factory=lambda: [
        "ACCOUNT",
        "TRANSACTION",
        "CREDIT_CARD",
        "GENERAL",
    ])


@dataclass
class EvaluationSettings:
    save_metrics: bool = True
    save_confusion: bool = True
    save_predictions: bool = True
    save_label_map: bool = True
    generate_visuals: bool = True


@dataclass
class TrainingSettings:
    enabled: bool = True
    model_name: str = "IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese"
    output_subdir: str = "training"
    max_length: int = 256
    epochs: int = 5
    batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    dropout: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    use_focal: bool = False
    focal_gamma: float = 1.5
    weight_primary: bool = True
    weight_secondary: bool = True
    weighted_sampler: bool = True
    fp16_auto: bool = True
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)


@dataclass
class TuningSettings:
    enabled: bool = False
    trials: int = 12
    output_subdir: str = "tuning"
    direction: str = "maximize"
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class StabilitySettings:
    enabled: bool = False
    runs: int = 3
    output_subdir: str = "stability"
    base_seed: int = 42


@dataclass
class ProjectConfig:
    config_path: Path
    config_dir: Path
    project_root: Path
    project: ProjectSettings = field(default_factory=ProjectSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    data: DataSettings = field(default_factory=DataSettings)
    labels: LabelSettings = field(default_factory=LabelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    tuning: TuningSettings = field(default_factory=TuningSettings)
    stability: StabilitySettings = field(default_factory=StabilitySettings)

    def resolve_output_dir(self, subdir: str, suffix: Optional[str] = None) -> Path:
        base = self.project.output_root / subdir
        if suffix:
            base = base / suffix
        base.mkdir(parents=True, exist_ok=True)
        return base

    def update_data_paths(self, train_path: Path, test_path: Path) -> None:
        self.data.train_path = train_path
        self.data.test_path = test_path


def _resolve_path(
    value: Optional[str],
    *,
    config_dir: Path,
    project_root: Path,
    default: Optional[str] = None,
    prefer_config_dir: bool = False,
) -> Optional[Path]:
    if value is None:
        value = default
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    text = str(path)
    if text.startswith(("./", ".\\", "../", "..\\")):
        return (config_dir / path).resolve()
    base = config_dir if prefer_config_dir else project_root
    return (base / path).resolve()


def _resolve_relative(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def load_project_config(path: Path) -> ProjectConfig:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    config_dir = path.parent
    project_root = config_dir.parent

    project_raw = raw.get("project", {})
    logging_raw = project_raw.get("logging", {})
    project = ProjectSettings(
        seed=int(project_raw.get("seed", ProjectSettings.seed)),
        output_root=_resolve_path(
            project_raw.get("output_root"),
            config_dir=config_dir,
            project_root=project_root,
            default=str(ProjectSettings.output_root),
        ),
        logging=LoggingSettings(
            config=_resolve_path(
                logging_raw.get("config"),
                config_dir=config_dir,
                project_root=project_root,
                prefer_config_dir=True,
            ),
            level=logging_raw.get("level", LoggingSettings.level),
        ),
    )

    pipeline_raw = raw.get("pipeline", {})
    pipeline = PipelineSettings(
        stages=list(pipeline_raw.get("stages", PipelineSettings().stages))
    )

    data_raw = raw.get("data", {})
    root_dir = _resolve_path(
        data_raw.get("root_dir"),
        config_dir=config_dir,
        project_root=project_root,
        default=str(DataSettings.root_dir),
    )
    if root_dir is None:
        raise ValueError("data.root_dir must be provided")
    split_raw = data_raw.get("split", {})
    split_settings = SplitSettings(
        enabled=bool(split_raw.get("enabled", SplitSettings.enabled)),
        input_path=_resolve_path(
            split_raw.get("input_path"),
            config_dir=config_dir,
            project_root=project_root,
        ),
        output_root=_resolve_path(
            split_raw.get("output_root"),
            config_dir=config_dir,
            project_root=project_root,
        ) or root_dir,
        train_template=split_raw.get("train_template", SplitSettings.train_template),
        test_template=split_raw.get("test_template", SplitSettings.test_template),
        test_size=float(split_raw.get("test_size", SplitSettings.test_size)),
        stratify_key=split_raw.get("stratify_key", SplitSettings.stratify_key),
        random_seed=int(split_raw.get("random_seed", SplitSettings.random_seed)),
        timestamp_format=split_raw.get("timestamp_format", SplitSettings.timestamp_format),
    )
    data = DataSettings(
        root_dir=root_dir,
        train_path=_resolve_relative(root_dir, data_raw.get("train_file")),
        test_path=_resolve_relative(root_dir, data_raw.get("test_file")),
        text_field=data_raw.get("text_field", DataSettings.text_field),
        primary_field=data_raw.get("primary_field", DataSettings.primary_field),
        secondary_field=data_raw.get("secondary_field", DataSettings.secondary_field),
        split=split_settings,
    )

    labels_raw = raw.get("labels", {})
    labels = LabelSettings(
        primary=list(labels_raw.get("primary", LabelSettings().primary)),
        secondary=list(labels_raw.get("secondary", LabelSettings().secondary)),
    )

    training_raw = raw.get("training", {})
    evaluation_raw = training_raw.get("evaluation", {})
    evaluation = EvaluationSettings(
        save_metrics=bool(evaluation_raw.get("save_metrics", EvaluationSettings.save_metrics)),
        save_confusion=bool(evaluation_raw.get("save_confusion", EvaluationSettings.save_confusion)),
        save_predictions=bool(evaluation_raw.get("save_predictions", EvaluationSettings.save_predictions)),
        save_label_map=bool(evaluation_raw.get("save_label_map", EvaluationSettings.save_label_map)),
        generate_visuals=bool(evaluation_raw.get("generate_visuals", EvaluationSettings.generate_visuals)),
    )
    training = TrainingSettings(
        enabled=bool(training_raw.get("enabled", TrainingSettings.enabled)),
        model_name=training_raw.get("model_name", TrainingSettings.model_name),
        output_subdir=training_raw.get("output_subdir", TrainingSettings.output_subdir),
        max_length=int(training_raw.get("max_length", TrainingSettings.max_length)),
        epochs=int(training_raw.get("epochs", TrainingSettings.epochs)),
        batch_size=int(training_raw.get("batch_size", TrainingSettings.batch_size)),
        eval_batch_size=int(training_raw.get("eval_batch_size", TrainingSettings.eval_batch_size)),
        learning_rate=float(training_raw.get("learning_rate", TrainingSettings.learning_rate)),
        dropout=float(training_raw.get("dropout", TrainingSettings.dropout)),
        alpha=float(training_raw.get("alpha", TrainingSettings.alpha)),
        beta=float(training_raw.get("beta", TrainingSettings.beta)),
        use_focal=bool(training_raw.get("use_focal", TrainingSettings.use_focal)),
        focal_gamma=float(training_raw.get("focal_gamma", TrainingSettings.focal_gamma)),
        weight_primary=bool(training_raw.get("weight_primary", TrainingSettings.weight_primary)),
        weight_secondary=bool(training_raw.get("weight_secondary", TrainingSettings.weight_secondary)),
        weighted_sampler=bool(training_raw.get("weighted_sampler", TrainingSettings.weighted_sampler)),
        fp16_auto=bool(training_raw.get("fp16_auto", TrainingSettings.fp16_auto)),
        evaluation=evaluation,
    )

    tuning_raw = raw.get("tuning", {})
    tuning = TuningSettings(
        enabled=bool(tuning_raw.get("enabled", TuningSettings.enabled)),
        trials=int(tuning_raw.get("trials", TuningSettings.trials)),
        output_subdir=tuning_raw.get("output_subdir", TuningSettings.output_subdir),
        direction=tuning_raw.get("direction", TuningSettings.direction),
        search_space=dict(tuning_raw.get("search_space", {})),
    )

    stability_raw = raw.get("stability", {})
    stability = StabilitySettings(
        enabled=bool(stability_raw.get("enabled", StabilitySettings.enabled)),
        runs=int(stability_raw.get("runs", StabilitySettings.runs)),
        output_subdir=stability_raw.get("output_subdir", StabilitySettings.output_subdir),
        base_seed=int(stability_raw.get("base_seed", StabilitySettings.base_seed)),
    )

    cfg = ProjectConfig(
        config_path=path,
        config_dir=config_dir,
        project_root=project_root,
        project=project,
        pipeline=pipeline,
        data=data,
        labels=labels,
        training=training,
        tuning=tuning,
        stability=stability,
    )

    LOGGER.debug("Loaded project configuration from %s", path)
    return cfg
