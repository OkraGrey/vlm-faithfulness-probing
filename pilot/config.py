"""Single source of truth for pilot + Phase 1 settings.

Change ONE place to swap models, devices, tasks, or sample counts.
"""
import os
from pathlib import Path
import torch

# --- Model ---
# 3B is the default. To run 7B WITHOUT editing code (keeps git clean across machines):
#   export QWEN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# All scripts read config.MODEL_NAME, and hooks read layer count/dim dynamically,
# so the same pipeline runs on either model (3B: 36L/2048d, 7B: 28L/3584d).
MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

# --- Device ---
# Auto-select: CUDA > MPS > CPU. fp16 on GPU, fp32 on CPU.
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# --- BLINK subset(s) ---
# Pilot (smoke): one task, 5 samples — used by pilot.py
BLINK_TASK = "Spatial_Relation"
NUM_SAMPLES = 5

# Phase 1 (real inference): all 14 BLINK task types.
# Each is visual-perception-grounded by BLINK's design (caption + LLM = random).
BLINK_TASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]
SAMPLES_PER_TASK = 10000    # FULL RUN: effectively "all available examples" — capped by split size
BLINK_SPLIT = "val"

# --- IO ---
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
INFERENCE_DIR = OUTPUTS_DIR / "inference"
INFERENCE_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    print(f"MODEL_NAME       = {MODEL_NAME}")
    print(f"DEVICE           = {DEVICE}")
    print(f"DTYPE            = {DTYPE}")
    print(f"BLINK_TASK       = {BLINK_TASK}            (pilot.py)")
    print(f"NUM_SAMPLES      = {NUM_SAMPLES}                       (pilot.py)")
    print(f"BLINK_TASKS      = {BLINK_TASKS}")
    print(f"SAMPLES_PER_TASK = {SAMPLES_PER_TASK}")
    print(f"BLINK_SPLIT      = {BLINK_SPLIT}")
    print(f"OUTPUTS_DIR      = {OUTPUTS_DIR}")
    print(f"INFERENCE_DIR    = {INFERENCE_DIR}")
