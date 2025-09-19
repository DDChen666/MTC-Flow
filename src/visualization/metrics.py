from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

LOGGER = logging.getLogger(__name__)


def _ensure_available() -> bool:
    if plt is None:
        LOGGER.warning("matplotlib is not installed; skipping visualization generation")
        return False
    return True


def plot_confusion_matrix(matrix, labels, title: str, output_path: Path) -> None:
    if not _ensure_available():
        return
    data = np.array(matrix, dtype=float)
    if data.size == 0:
        LOGGER.warning("Empty confusion matrix for %s; skipping", title)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels) + 2), max(5, 0.8 * len(labels) + 2)))
    im = ax.imshow(data, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    vmax = data.max() if data.size else 0
    for (i, j), val in np.ndenumerate(data):
        display = f"{int(val)}" if vmax > 5 else f"{val:.1f}"
        ax.text(j, i, display, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved confusion matrix to %s", output_path)


def plot_f1_bars(report: Dict[str, Dict[str, float]], labels, title: str, output_path: Path) -> None:
    if not _ensure_available():
        return
    scores = []
    for label in labels:
        metrics = report.get(label) or {}
        scores.append(float(metrics.get("f1-score", 0.0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels) + 2), 4))
    bars = ax.bar(labels, scores, color="#4c72b0")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for rect, value in zip(bars, scores):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved F1 bar chart to %s", output_path)
