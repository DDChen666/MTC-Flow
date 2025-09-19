from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import pandas as pd
import torch

from src.mtc.inference import (
    load_uploaded_records,
    prepare_inference_artifacts,
    run_evaluation,
)
from src.mtc.labels import TaskLabelSchema

DEFAULT_BASE_MODEL = "IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_PRIMARY_FIELD = "primary"
DEFAULT_SECONDARY_FIELD = "secondary"
ARTIFACT_ROOT = Path("artifacts") / "training"


def _list_training_runs() -> List[str]:
    if not ARTIFACT_ROOT.exists():
        return []
    runs = [p for p in ARTIFACT_ROOT.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in runs]


def _build_summary(
    metrics: dict,
    total_rows: int,
    kept_rows: int,
    label_schema: TaskLabelSchema,
) -> str:
    primary = metrics["primary"]
    secondary = metrics["secondary"]
    skipped = total_rows - kept_rows
    lines = [
        "### Performance Summary",
        f"- Uploaded rows: **{total_rows}**",
        f"- Evaluated rows: **{kept_rows}**"
        + (f" (filtered out {skipped})" if skipped else ""),
        "",
        "**Primary Task**",
        f"• Accuracy: `{primary['accuracy']:.4f}`",
        f"• Macro F1: `{primary['macro_f1']:.4f}`",
        f"• Weighted F1: `{primary['weighted_f1']:.4f}`",
        "",
        "**Secondary Task**",
        f"• Accuracy: `{secondary['accuracy']:.4f}`",
        f"• Macro F1: `{secondary['macro_f1']:.4f}`",
        f"• Weighted F1: `{secondary['weighted_f1']:.4f}`",
        "",
        "Label sets:",
        f"- Primary labels: {', '.join(label_schema.primary.labels)}",
        f"- Secondary labels: {', '.join(label_schema.secondary.labels)}",
    ]
    return "\n".join(lines)


def _report_to_dataframe(report: dict) -> pd.DataFrame:
    df = pd.DataFrame(report).T
    df.index.name = "class"
    return df.reset_index()


def _confusion_to_dataframe(matrix: List[List[int]], labels: List[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _build_predictions_dataframe(dataset, evaluation, label_schema: TaskLabelSchema) -> pd.DataFrame:
    records = []
    for idx, item in enumerate(dataset.items):
        records.append({
            "text": item["text"],
            "gold_primary": item["primary"],
            "pred_primary": label_schema.primary.id2label[evaluation.pred_primary[idx]],
            "gold_secondary": item["secondary"],
            "pred_secondary": label_schema.secondary.id2label[evaluation.pred_secondary[idx]],
        })
    return pd.DataFrame(records)


def evaluate(
    run_choice: str,
    custom_run_dir: str,
    base_model_name: str,
    text_field: str,
    primary_field: str,
    secondary_field: str,
    max_length: int,
    batch_size: int,
    data_file,
):
    if data_file is None:
        raise gr.Error("Please upload a dataset file (JSON, JSONL, or CSV).")

    run_path_str = custom_run_dir.strip() or run_choice.strip()
    if not run_path_str:
        raise gr.Error("Select a training run or provide a custom path.")

    model_dir = Path(run_path_str).expanduser().resolve()
    if not model_dir.exists():
        raise gr.Error(f"Model directory not found: {model_dir}")

    uploaded_path = Path(data_file.name)
    records = load_uploaded_records(uploaded_path)
    if not records:
        raise gr.Error("Uploaded file contains no records.")

    artifacts = prepare_inference_artifacts(
        model_dir=model_dir,
        base_model_name=base_model_name.strip() or DEFAULT_BASE_MODEL,
        text_field=text_field.strip() or DEFAULT_TEXT_FIELD,
        primary_field=primary_field.strip() or DEFAULT_PRIMARY_FIELD,
        secondary_field=secondary_field.strip() or DEFAULT_SECONDARY_FIELD,
    )

    evaluation, dataset = run_evaluation(
        artifacts=artifacts,
        rows=records,
        max_length=max_length,
        batch_size=batch_size,
    )

    summary = _build_summary(
        metrics=evaluation.metrics,
        total_rows=len(records),
        kept_rows=len(dataset),
        label_schema=artifacts.label_schema,
    )

    primary_report_df = _report_to_dataframe(evaluation.metrics["primary"]["report"])
    secondary_report_df = _report_to_dataframe(evaluation.metrics["secondary"]["report"])

    primary_cm_df = _confusion_to_dataframe(
        evaluation.metrics["primary"]["confusion_matrix"],
        evaluation.metrics["primary"]["labels"],
    )
    secondary_cm_df = _confusion_to_dataframe(
        evaluation.metrics["secondary"]["confusion_matrix"],
        evaluation.metrics["secondary"]["labels"],
    )

    predictions_df = _build_predictions_dataframe(dataset, evaluation, artifacts.label_schema)

    tmp_dir = Path(tempfile.mkdtemp(prefix="gradio_mtc_"))
    predictions_path = tmp_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    return (
        summary,
        primary_report_df,
        secondary_report_df,
        primary_cm_df,
        secondary_cm_df,
        predictions_df,
        str(predictions_path),
    )


def build_interface() -> gr.Blocks:
    runs = _list_training_runs()
    with gr.Blocks(title="MTC Flow Evaluation") as demo:
        gr.Markdown(
            """# Multi-Task Classification Evaluation
Upload a labelled dataset to measure how a trained run performs.\nSelect an existing training artifact or provide a custom directory containing `pytorch_model.bin` and `label_map.json`.
"""
        )

        with gr.Row():
            run_dropdown = gr.Dropdown(
                choices=runs,
                value=runs[0] if runs else "",
                label="Saved training runs",
                info="Located under artifacts/training",
            )
            custom_run = gr.Textbox(
                label="Custom model directory (optional)",
                placeholder="E.g. /path/to/run",
            )

        with gr.Row():
            base_model = gr.Textbox(
                label="Base model name",
                value=DEFAULT_BASE_MODEL,
            )
            max_length = gr.Slider(
                minimum=16,
                maximum=512,
                step=8,
                value=256,
                label="Max sequence length",
            )
            batch_size = gr.Slider(
                minimum=1,
                maximum=64,
                step=1,
                value=8,
                label="Batch size",
            )

        with gr.Row():
            text_field = gr.Textbox(label="Text field", value=DEFAULT_TEXT_FIELD)
            primary_field = gr.Textbox(label="Primary label field", value=DEFAULT_PRIMARY_FIELD)
            secondary_field = gr.Textbox(label="Secondary label field", value=DEFAULT_SECONDARY_FIELD)

        data_file = gr.File(label="Upload evaluation dataset", file_types=[".json", ".jsonl", ".csv"], file_count="single")

        evaluate_button = gr.Button("Run evaluation", variant="primary")

        summary = gr.Markdown()
        with gr.Row():
            primary_report = gr.Dataframe(label="Primary classification report")
            secondary_report = gr.Dataframe(label="Secondary classification report")
        with gr.Row():
            primary_cm = gr.Dataframe(label="Primary confusion matrix")
            secondary_cm = gr.Dataframe(label="Secondary confusion matrix")
        predictions = gr.Dataframe(label="Predictions preview")
        download = gr.File(label="Download predictions CSV")

        evaluate_button.click(
            fn=evaluate,
            inputs=[
                run_dropdown,
                custom_run,
                base_model,
                text_field,
                primary_field,
                secondary_field,
                max_length,
                batch_size,
                data_file,
            ],
            outputs=[
                summary,
                primary_report,
                secondary_report,
                primary_cm,
                secondary_cm,
                predictions,
                download,
            ],
        )
    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
