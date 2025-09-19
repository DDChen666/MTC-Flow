"""Deprecated legacy module preserved for backward compatibility.

The multi-task classification workflow has been migrated to the config-driven
architecture under the `src` package. Please invoke `python -m src.main` or the
thin wrappers (train_mtc.py / optuna_tune.py / eval_stability.py) instead of
directly importing this module.
"""
from __future__ import annotations

raise RuntimeError(
    "mtc_core.py has been replaced by the new config-driven pipeline. "
    "Use `python -m src.main` with an explicit config to execute the workflow."
)
