# 多任務分類專案重構指南

本專案已全面重構為以設定檔驅動的模組化架構，整合資料切割、訓練、超參數搜尋與穩定性評估等流程，並提供單一入口 `main.py` 控制執行步驟。以下說明涵蓋資料結構、設定檔欄位、產出物與擴充方式，協助快速掌握新設計。

## Quick Start (for newcomers)

1. **Create a virtual environment and install dependencies**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -e .
   ```
   If you already have a venv, just activate it and run the install commands.

2. **Prepare the dataset folder**
   - Place `train_set_*.json` and `test_set_*.json` under `input_train_and_test_json/`.
   - If you want to keep the data elsewhere, change `data.root_dir` in your config file.

3. **Adjust the configuration file**
   - `configs/default.yaml` is a template; copy it and tweak labels, model name, epochs, etc.
   - Any stage with `enabled: false` will be skipped, so only turn on the stages you need.

4. **Run your first training cycle**
   ```powershell
   python -m src.main --config configs/default.yaml --stages train
   ```
   Use `python -m src.main --list-stages` whenever you want to double-check which stages will run.

> Need more hand-holding? See [docs/quickstart.md](docs/quickstart.md) for a step-by-step walkthrough.

## 安裝或啟動虛擬環境
```
.\venv\Scripts\Activate.ps1
pip install -e .
```


## 專案結構

```
.
├── configs/                 # 設定與記錄相關檔案
│   ├── default.yaml         # 主設定檔，控制所有流程與超參數
│   └── logging.yaml         # logging.dictConfig 設定
├── input_train_and_test_json/  # 目前的訓練/測試資料 (依設定檔引用)
├── src/                     # 新的 Python 套件 (以套件形式發佈)
│   ├── main.py              # 中央控制器 (python -m src.main)
│   ├── data/                # 資料處理與 I/O (JSON/JSONL 寫讀與切割)
│   ├── mtc/                 # 多任務分類模型相關模組 (Dataset / Model / Training)
│   ├── pipeline/            # 各個 stage 的執行邏輯 (split / train / tune / stability)
│   ├── utils/               # 設定檔載入、logging 工具等共用函式
│   └── visualization/       # 評估結果圖像化工具 (matplotlib)
├── artifacts/               # 預設輸出目錄 (首次執行會自動建立)
├── pyproject.toml           # 依賴與套件資訊 (pip install -e .)
├── train_mtc.py             # 舊入口兼容 wrapper → 呼叫新 pipeline train stage
├── optuna_tune.py           # 舊入口兼容 wrapper → 呼叫 new pipeline tune stage
├── eval_stability.py        # 舊入口兼容 wrapper → 呼叫 new pipeline stability stage
├── 2-4....py                # 舊資料切割腳本 wrapper → 呼叫 new pipeline split stage
└── mtc_core.py              # 已註明淘汰並引導使用新入口
```

> 備註：`legacy` 腳本（train_mtc.py 等）僅保留 CLI 參數 → 呼叫 `python -m src.main`，維持既有習慣但不再硬編碼任何路徑或超參數。

## 環境與安裝

1. 建議使用 Python 3.9+（與原虛擬環境相容）。
2. 安裝依賴：
   ```bash
   pip install -e .
   ```
   - `torch` / `transformers` 為核心訓練依賴。
   - `optuna`、`matplotlib` 可依需求啟用，若未安裝則自動跳過對應 stage 或顯示明確錯誤。
3. 若需 GPU 訓練，請確保對應的 CUDA/PyTorch 安裝妥善。

## 新的單一入口：`python -m src.main`

- `--config`: 指定設定檔，預設為 `configs/default.yaml`。
- `--stages`: 指定要執行的 stage 順序（覆寫設定檔）。可複數，例如：
  ```bash
  python -m src.main --stages split train
  ```
- `--list-stages`: 列出可用 stage (`split`, `train`, `tune`, `stability`)。

Stage 說明：

| Stage      | 說明 | 產出 | 需求 |
|------------|------|------|------|
| `split`    | 依設定檔從原始資料切出 train/test，並回寫路徑到 config context | `train_set_*.json`, `test_set_*.json` | `data.split.enabled = true` 且 `input_path` 指向原始資料 |
| `train`    | 執行模型訓練、推論與指標輸出 | metrics JSON、混淆矩陣 CSV/PNG、預測 CSV、label_map | `data.train_file`/`data.test_file` 可由 split stage 或手動給定 |
| `tune`     | Optuna 超參數搜尋 (每個 trial 都獨立訓練) | `optuna_best.json`, `optuna_trials.csv`，以及各 trial 的 metrics/predictions | 需安裝 `optuna`，並設定 `tuning.enabled = true` |
| `stability`| 重複訓練 N 次評估 macro F1 的平均/標準差 | `stability_runs.csv`, `stability_summary.json` + 各 run 指標檔案 | `stability.enabled = true` |

> 每個 stage 的啟用與參數皆在設定檔控制；即使列在 `pipeline.stages`，若該 stage 的 `enabled` 為 `false` 仍會自動略過。

## 設定檔（`configs/default.yaml`）詳解

### `project`
- `seed`: 隨機種子（統一傳遞到所有 stage）。
- `output_root`: 所有成果預設輸出根目錄（預設 `artifacts/`）。
- `logging`: 指定 logging 設定檔（預設為 `configs/logging.yaml`）；若該檔不存在則 fallback 至基本設定。

### `pipeline`
- `stages`: 執行順序，預設列出全部 stage。若想只跑部分流程，可：
  - 直接編輯此清單；或
  - 使用 CLI `--stages` 覆寫。

### `data`
- `root_dir`: 資料所在目錄。
- `train_file` / `test_file`: 已存在的資料檔名。當 `split` stage 執行後會更新 context，後續 stage 不需 `find_latest_*`。
- `text_field` / `primary_field` / `secondary_field`: 對應 JSON 欄位名稱，便於未來遷移到其他資料模式。
- `split`: 控制資料切割。
  - `enabled`: true 時執行資料切割並輸出新的 train/test。
  - `input_path`: 原始資料（支援 JSON/JSONL）。
  - `train_template` / `test_template`: 檔名樣板（包含 `{timestamp}`）。
  - `test_size`, `stratify_key`, `random_seed`: 對應 `train_test_split` 參數。

### `labels`
- `primary` / `secondary`: 標籤集合，系統會自動建立 label ↔︎ id 對映。
- 如需增減類別，只需在此調整即可，`ReviewDataset` 會依設定篩選合法記錄。

### `training`
- 核心模型與超參數設定 (`model_name`, `epochs`, `batch_size`, `learning_rate`, `max_length`, `dropout`, `alpha`, `beta`, `use_focal`, `focal_gamma`, `weight_*`, `weighted_sampler`, `fp16_auto` 等)。
- `output_subdir`: 訓練結果會寫入 `artifacts/training/<timestamp>/`。
- `evaluation`: 控制輸出項目：
  - `save_metrics`: `metrics_<timestamp>.json`
  - `save_confusion`: 混淆矩陣 CSV
  - `save_predictions`: 預測 CSV（含 label id 與 label 名稱）
  - `save_label_map`: label 對映表
  - `generate_visuals`: `matplotlib` 圖檔（若未安裝會跳警告並略過）

### `tuning`
- `enabled`: true 時才可透過 stage 執行 Optuna。
- `trials`: 迭代次數。
- `search_space`: 以欄位名稱對應 `TrainingSettings` 欄位，支援 `categorical` / `uniform` / `loguniform` / `int` 家族。
  - 範例：
    ```yaml
    tuning:
      enabled: true
      trials: 20
      search_space:
        learning_rate:
          distribution: loguniform
          low: 5e-6
          high: 3e-5
        batch_size:
          distribution: categorical
          choices: [2, 4, 6]
    ```

### `stability`
- `runs`: 執行幾次（每次的 seed = `base_seed + index`）。
- `output_subdir`: 預設 `artifacts/stability/`，每 run 會在其下產生 `run_XX/` 子目錄。

## 產出與日誌

- **日誌 (logging)**：依 `configs/logging.yaml` 設定，預設輸出到終端與 `artifacts/pipeline.log`。
- **訓練結果**：以 `<output_root>/<training.output_subdir>/<run_name>/` 存放。
  - `metrics_<run>.json`：含 primary/secondary accuracy、macro/weighted F1、classification report、混淆矩陣與 trial 參數。
  - `confusion_primary_<run>.csv / .png`、`confusion_secondary_<run>.csv / .png`
  - `predictions_<run>.csv`：包含 gold/pred id、label 字串。
  - `label_map.json`
- **Optuna**：`artifacts/tuning/` 下保存 `optuna_best.json` 與 `optuna_trials.csv`，每個 trial 也有對應子資料夾。
- **Stability**：`stability_runs.csv` 列出每次 seed 與 macro F1，`stability_summary.json` 提供平均與標準差。

## 擴充與客製化建議

1. **新增模型或任務**：
   - 在 `configs/default.yaml` 中增列新的 label 集合或資料欄位即可。
   - 如需 LoRA / 其他 adaptor，可於 `src/mtc/modeling.py` 擴增模型邏輯，同時在設定檔加上對應開關。
2. **資料前處理**：
   - 於 `src/data` 新增模組並在 pipeline stage 中串接。
   - 若涉及多個來源，可在設定檔加入自訂欄位（例如多路資料路徑）。
3. **更多 stage**：
   - 在 `src/pipeline/stages.py` 新增函式後，於 `src/main.py` 的 `STAGE_HANDLERS` 中註冊即可由設定檔呼叫。
4. **進階設定覆寫**：
   - 若需以 CLI 覆寫設定，建議在 `src/utils/config.py` 增加 merge/override 功能，再由 `main.py` 解析額外旗標。

## 與舊流程的對應

| 舊腳本 | 新執行方式 | 說明 |
|--------|------------|------|
| `2-4.數據...py` | `python -m src.main --stages split` | 不再依賴 `find_latest_*`，改由設定檔指定輸入與命名樣板 |
| `train_mtc.py` | `python -m src.main --stages train` | 完整訓練含評估、輸出與視覺化 |
| `optuna_tune.py` | `python -m src.main --stages tune` | 需預先在 `default.yaml` 啟用 tuning，或自行建立 tuning 專用設定檔 |
| `eval_stability.py` | `python -m src.main --stages stability` | 依設定檔的 runs/base_seed 執行多次訓練 |
| `mtc_core.py` | —— | 已移除核心實作，直接導向新架構 |

## 驗證建議

1. **列出 stage**：`python -m src.main --list-stages`（已可執行）。
2. **Dry run**：
   - 先將 `training.enabled=false`，只跑 `split` 以確認資料路徑整理正常。
3. **正式訓練**：
   - 依預設 `configs/default.yaml`，確保 `train_file` / `test_file` 指向現有資料後執行 `python -m src.main --stages train`。
4. **超參數搜尋或穩定性測試**：
   - 編輯設定檔啟用相應 stage，再執行 `python -m src.main --stages tune stability` 等組合。

## 常見問題

- **Optuna 未安裝**：執行 tune stage 會收到明確例外，安裝後重試即可，不影響其他 stage。
- **Matplotlib 未安裝**：會顯示警告並略過圖像輸出，文字指標仍會產出。
- **自訂訓練/測試檔名稱**：直接在 `configs/default.yaml` 修改 `train_file` / `test_file`，或啟用 `split` stage 重新切割。
- **切換語言或模型**：更新 `training.model_name`，並視需求調整 tokenizer 長度 `max_length`、標籤集合。

---

此 README 提供完整結構與操作手冊，後續只需透過設定檔即可擴充為其他任務（如 LoRA 微調、更多任務 Head 等），無需再依賴時間排序或手動修改腳本。若需進一步自動化（例如整合 CI/CD 觸發訓練），可在 `src/pipeline/stages.py` 擴增新 stage 並透過設定檔啟用。
