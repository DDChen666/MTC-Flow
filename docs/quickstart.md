# Quick Start Guide

This document walks through the project from zero to an initial training run. Complete these checkpoints and you will understand where to place data, how to edit the configuration file, and how to orchestrate the stages.

## 0. Prerequisites
- **Python 3.9+** (64‑bit recommended). Install CUDA-enabled PyTorch if you plan to train on GPU.
- **git + SSH keys** if you intend to push your work to GitHub.
- **Labelled data** in JSON/JSONL format with at least `text`, `primary`, and `secondary` fields.

## 1. Create a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```
Remember to run `deactivate` when you want to leave the environment.

## 2. Prepare the dataset directory
1. The default location is `input_train_and_test_json/`.
2. Drop two files inside it, for example:
   - `train_set_20250919_181918.json`
   - `test_set_20250919_181918.json`
3. Files can be JSON array or JSONL. The field names must match the ones declared in the config (`data.text_field`, `data.primary_field`, `data.secondary_field`).
4. Prefer a different folder? Update `data.root_dir` in your config.

## 3. Configure the pipeline
- Copy `configs/default.yaml` to a new file if you want to keep multiple experiment configs.
- Stages are controlled by `enabled` flags. `false` means “skip this stage”.
- Handy parameters:
  | Field | Description | When to change |
  |-------|-------------|----------------|
  | `training.model_name` | HuggingFace model checkpoint | Switch to another backbone or LoRA base |
  | `training.epochs` | Number of epochs | Increase for small datasets, decrease when overfitting |
  | `training.batch_size` | Train batch size | Bounded by GPU/CPU memory |
  | `tuning.search_space` | Optuna search ranges | Adjust when exploring new hyperparameters |

## 4. Run the pipeline
1. Preview available stages:
   ```powershell
   python -m src.main --config configs/default.yaml --list-stages
   ```
2. Run only the training stage (common first step):
   ```powershell
   python -m src.main --config configs/default.yaml --stages train
   ```
3. Execute all stages listed in `pipeline.stages`:
   ```powershell
   python -m src.main --config configs/default.yaml
   ```
4. Check the `artifacts/` folder for outputs (metrics, confusion matrices, predictions, etc.).

## 5. Optional extras
- **Dataset split**: set `data.split.enabled: true` and provide `split.input_path`; the pipeline will emit fresh train/test files.
- **Hyperparameter search**: set `tuning.enabled: true` and run `python -m src.main --stages tune`.
- **Stability check**: set `stability.enabled: true` or run `python -m src.main --stages stability`.

## 6. Troubleshooting
| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError` during training | Config paths don’t match actual files | Double-check `data.root_dir`, `train_file`, and `test_file` |
| `ImportError: No module named optuna` | Optuna not installed | `pip install optuna` or disable the tune stage |
| CUDA OOM / RAM exhaustion | Batch size too big | Reduce `training.batch_size`, disable `fp16_auto`, or switch to CPU |

## 7. Next steps
- **Version control**: keep configs under `configs/` and let `.gitignore` hide local data/venv.
- **Automation**: schedule `python -m src.main --config ... --stages train` in CI to keep models fresh.
- **Extension**: customise `src/mtc/` modules to add new heads, losses, or preprocessing for other tasks.

Happy experimenting!
