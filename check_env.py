# python check_env.py
import sys
import transformers
import torch

print("--- Python 环境诊断报告 ---")
print(f"正在使用的 Python 解释器: {sys.executable}")
print("-" * 25)
print(f"Transformers 库版本: {transformers.__version__}")
print(f"Transformers 库路径: {transformers.__file__}")
print("-" * 25)
print(f"PyTorch 库版本: {torch.__version__}")
print(f"PyTorch 库路径: {torch.__file__}")
print("-" * 25)
print("诊断结束。")