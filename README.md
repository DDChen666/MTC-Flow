# 多任務分類專案重構指南

本專案已全面重構為以設定檔驅動的模組化架構，整合資料切割、訓練、超參數搜尋與穩定性評估等流程，並提供單一入口 `main.py` 控制執行步驟。以下說明涵蓋資料結構、設定檔欄位、產出物與擴充方式，協助快速掌握新設計。

## Quick Start (for newcomers)

1. **Create a virtual environment and install dependencies**
   ```powershell
   python -m venv venv
   .\\venv\\Scripts\\Activate.ps1
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

## Interactive Evaluation (Gradio)

Launch the Gradio dashboard to test a trained run on new labelled data:

```powershell
python gradio_app.py
```

- Select a saved model directory under `artifacts/training/` or enter a custom path containing `pytorch_model.bin` and `label_map.json`.
- Upload a dataset (`.json`, `.jsonl`, `.csv`) where each row包含 text、primary、secondary 標籤。
- 立即查看 Primary/Secondary Accuracy、Macro F1、混淆矩陣與完整預測，並可下載 CSV。

