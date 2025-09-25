# MTC-Flow 多任務文本分類流程

<p align="right">
  <strong>語言 / Language:</strong>
  <a href="#readme-zh">中文</a> ·
  <a href="#readme-en">English</a>
</p>

---

<div id="readme-zh">

## 🚀 專案總覽
MTC-Flow 是一個以 **設定檔驅動 (config-driven)** 的多任務文本分類訓練流程。透過單一入口 `python -m src.main`，即可串連資料切分、模型訓練、超參數搜尋、穩定性評估與視覺化輸出，並可搭配 Gradio 介面快速檢視推論結果。

> 完整逐步教學請參考 [`docs/quickstart.md`](docs/quickstart.md)。

## ✨ 核心特性
- **模組化流程控制**：以 YAML 設定檔決定要執行的 stage（split/train/tune/stability），也可用命令列覆蓋。
- **多任務標籤支援**：同時處理 Primary/Secondary 標籤，並輸出精確度、Macro/Weighted F1、混淆矩陣等指標。
- **豐富的訓練產物**：自動保存指標、預測、標籤對應、圖表，方便後續分析。
- **超參數搜尋與穩定性分析**：內建 Optuna 搜尋與多次重訓統計，快速掌握模型表現分佈。
- **互動式評估**：`gradio_app.py` 可載入訓練產物，直接上傳新資料檔進行可視化評估。

## 🗂️ 目錄速覽
```
.
├── configs/            # YAML 設定檔範例
├── docs/               # 使用教學與補充說明
├── src/                # 核心程式碼（資料、訓練、推論、pipeline）
├── artifacts/          # 預設輸出目錄（訓練結果、圖表、搜尋紀錄）
├── input_train_and_test_json/
│   ├── train_*.json    # 訓練資料
│   └── test_*.json     # 測試資料
├── gradio_app.py       # Gradio 互動式評估介面
└── train_mtc.py 等工具腳本
```

## 📦 環境需求
- Python 3.9 以上
- (選用) CUDA 環境與對應的 PyTorch 版本
- 建議使用虛擬環境（`venv`、`conda` 皆可）

## ⚙️ 安裝與初始化
```bash
python -m venv venv
source venv/bin/activate      # Windows 使用 .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .              # 以開發模式安裝依賴
```

## 📁 準備資料
1. 將標註資料放入 `input_train_and_test_json/` 目錄，檔名需與設定檔一致。
2. 支援 JSON / JSONL，欄位至少包含 `text`、`primary`、`secondary`。
3. 若要自動切分資料，請於設定檔將 `data.split.enabled` 改為 `true` 並指定來源檔案。

## 🛠️ 設定檔重點
`configs/default.yaml` 為模板，建議複製後依需求調整。

| 區塊 | 常用欄位 | 說明 |
|------|----------|------|
| `project` | `seed`, `output_root`, `logging` | 控制隨機種子、輸出目錄與紀錄設定 |
| `pipeline` | `stages` | 預設執行順序，可改由命令列指定 |
| `data` | `root_dir`, `train_file`, `test_file`, `split.*` | 指定資料來源與是否進行切分 |
| `labels` | `primary`, `secondary` | 定義兩層標籤集合與順序 |
| `training` | `model_name`, `epochs`, `batch_size`, `learning_rate`, `evaluation.*` | 訓練相關超參數與輸出控制 |
| `tuning` | `enabled`, `trials`, `search_space` | Optuna 搜尋設定 |
| `stability` | `runs`, `base_seed` | 多次重訓評估設定 |

## 🔁 Pipeline 操作範例
```bash
# 1. 檢視可用 stage
python -m src.main --list-stages

# 2. 僅執行訓練
python -m src.main --config configs/default.yaml --stages train

# 3. 依設定檔順序執行（預設：split → train → tune → stability）
python -m src.main --config configs/default.yaml
```

### Stage 詳細說明
| Stage | 目的 | 產出 |
|-------|------|------|
| `split` | 依設定切分原始資料為 train/test | 生成新的 train/test JSON 檔案 |
| `train` | 進行模型訓練與評估 | 指標 JSON、預測 CSV、標籤對照、混淆矩陣與 F1 圖 |
| `tune` | 使用 Optuna 搜尋超參數 | `optuna_best.json`, `optuna_trials.csv` |
| `stability` | 多次重訓評估模型穩定度 | `stability_runs.csv`, `stability_summary.json` |

## 📊 產出物位置
- `artifacts/training/<run_name>/`：單次訓練所有輸出。
- `artifacts/tuning/`：超參數搜尋紀錄與最佳結果摘要。
- `artifacts/stability/`：穩定性統計報告。

## 🖥️ 互動式評估 (Gradio)
```bash
python gradio_app.py
```
1. 選擇或輸入訓練結果資料夾（需含 `pytorch_model.bin` 與 `label_map.json`）。
2. 上傳含標籤的新資料檔（JSON/JSONL/CSV）。
3. 介面會即時顯示 Primary/Secondary 指標、混淆矩陣、預測結果並可匯出。

## 🧪 常用腳本
- `train_mtc.py`：以指令列快速觸發訓練（舊版流程兼容）。
- `optuna_tune.py`：獨立啟動超參數搜尋。
- `eval_stability.py`：單獨進行穩定性評估。
- `check_env.py`：檢查環境依賴與 GPU 狀態。

## ❓ 疑難排解
| 問題 | 可能原因 | 解法 |
|------|----------|------|
| 找不到資料檔 | 路徑或檔名未與設定檔一致 | 檢查 `data.root_dir`、`train_file`、`test_file` |
| `ImportError: No module named optuna` | 尚未安裝 Optuna | `pip install optuna` 或停用 `tuning.enabled` |
| CUDA / 記憶體不足 | batch size 過大或 FP16 不適用 | 調整 `training.batch_size`、關閉 `fp16_auto` 或改用 CPU |

## 📚 延伸閱讀
- [`docs/quickstart.md`](docs/quickstart.md)：完整新手指南。
- `超詳細專案教學.md`、`簡易操作指令.md`：中文詳解與備忘筆記。

</div>

---

<div id="readme-en">

## 🚀 Overview
MTC-Flow is a **config-driven** pipeline for multi-task text classification. A single entry point `python -m src.main` orchestrates dataset splitting, model training, hyperparameter search, stability evaluation, and reporting. The optional Gradio app lets you validate trained runs on fresh labelled data with one upload.

> See [`docs/quickstart.md`](docs/quickstart.md) for a step-by-step walkthrough.

## ✨ Highlights
- **Modular pipeline** — control which stages (`split`, `train`, `tune`, `stability`) run via YAML or CLI overrides.
- **Dual-label support** — evaluate primary and secondary tasks simultaneously with accuracy, macro/weighted F1, confusion matrices, and charts.
- **Rich artifacts** — automatically persist metrics, predictions, label maps, and visualisations for every run.
- **Hyperparameter & stability tooling** — built-in Optuna search and repeated training statistics help you understand performance variance.
- **Interactive evaluation** — `gradio_app.py` loads training artifacts and visualises predictions for uploaded datasets.

## 🗂️ Directory at a Glance
```
.
├── configs/            # Sample YAML configs
├── docs/               # Guides and supplementary notes
├── src/                # Core modules (data, training, inference, pipeline)
├── artifacts/          # Default output root (runs, charts, search logs)
├── input_train_and_test_json/
│   ├── train_*.json    # Training data
│   └── test_*.json     # Evaluation data
├── gradio_app.py       # Interactive evaluation interface
└── train_mtc.py etc.   # Helper scripts
```

## 📦 Requirements
- Python 3.9+
- (Optional) CUDA-enabled environment with the matching PyTorch build
- Virtual environment is recommended (`venv` or `conda`)

## ⚙️ Installation
```bash
python -m venv venv
source venv/bin/activate      # On Windows: .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

## 📁 Dataset Preparation
1. Place labelled files under `input_train_and_test_json/` and ensure filenames match the config.
2. JSON and JSONL are supported; each record must include `text`, `primary`, and `secondary` fields.
3. To auto-split raw data, enable `data.split.enabled` and provide the source path in the config.

## 🛠️ Configuration Cheat Sheet
Use `configs/default.yaml` as a template and clone it per experiment.

| Section | Key fields | Notes |
|---------|------------|-------|
| `project` | `seed`, `output_root`, `logging` | Global seed, artifact root, logging config |
| `pipeline` | `stages` | Default execution order (override via CLI) |
| `data` | `root_dir`, `train_file`, `test_file`, `split.*` | Dataset locations and optional splitting |
| `labels` | `primary`, `secondary` | Define label sets and ordering |
| `training` | `model_name`, `epochs`, `batch_size`, `learning_rate`, `evaluation.*` | Training hyperparameters and artifact toggles |
| `tuning` | `enabled`, `trials`, `search_space` | Optuna study settings |
| `stability` | `runs`, `base_seed` | Repeated training analysis |

## 🔁 Pipeline Commands
```bash
# 1. List available stages
python -m src.main --list-stages

# 2. Train only
python -m src.main --config configs/default.yaml --stages train

# 3. Run the full pipeline (default: split → train → tune → stability)
python -m src.main --config configs/default.yaml
```

### Stage Reference
| Stage | Purpose | Outputs |
|-------|---------|---------|
| `split` | Create new train/test sets from raw data | Fresh train/test JSON files |
| `train` | Train the model and evaluate it | Metrics JSON, predictions CSV, label map, confusion/F1 visuals |
| `tune` | Explore hyperparameters with Optuna | `optuna_best.json`, `optuna_trials.csv` |
| `stability` | Measure variance across repeated runs | `stability_runs.csv`, `stability_summary.json` |

## 📊 Artifacts
- `artifacts/training/<run_name>/` — all outputs for a single training run.
- `artifacts/tuning/` — search logs and the best-trial summary.
- `artifacts/stability/` — aggregated statistics for repeated runs.

## 🖥️ Gradio App
```bash
python gradio_app.py
```
1. Select or provide the training run directory containing `pytorch_model.bin` and `label_map.json`.
2. Upload labelled data (JSON/JSONL/CSV).
3. Inspect primary/secondary metrics, confusion matrices, detailed predictions, and export CSVs.

## 🧪 Helper Scripts
- `train_mtc.py` — legacy-compatible CLI to trigger training quickly.
- `optuna_tune.py` — run hyperparameter search standalone.
- `eval_stability.py` — execute stability assessment independently.
- `check_env.py` — confirm environment prerequisites and GPU availability.

## ❓ Troubleshooting
| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Data files not found | Paths in the config do not match actual files | Double-check `data.root_dir`, `train_file`, and `test_file` |
| `ImportError: No module named optuna` | Optuna is missing | `pip install optuna` or disable `tuning.enabled` |
| CUDA / RAM exhaustion | Batch size too large or FP16 unsuitable | Lower `training.batch_size`, disable `fp16_auto`, or fall back to CPU |

## 📚 Further Reading
- [`docs/quickstart.md`](docs/quickstart.md) — full walkthrough for newcomers.
- `超詳細專案教學.md`, `簡易操作指令.md` — additional Chinese notes and cheat sheets.

</div>
