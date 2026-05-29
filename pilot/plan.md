# Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal end-to-end pipeline that loads BLINK, runs a Qwen2.5-VL model, and extracts last-token hidden states across LLM decoder layers — verifying the foundation works before any probing experiments.

**Architecture:** Single-process Python pipeline. HuggingFace `transformers` loads Qwen2.5-VL; PyTorch forward hooks attach to each LLM decoder layer to capture the last-token hidden state per sample. Hidden states + predictions are written to disk as a single `.npz` for downstream probing. Mac-friendly defaults (MPS, fp16, smaller 3B model); switching to 7B on a CUDA cluster is a one-line config change.

**Tech Stack:** Python 3.11 (venv), PyTorch (MPS on Mac / CUDA on cluster), `transformers >= 4.49`, `datasets`, `Pillow`, `numpy`. `scikit-learn` is included for probe work later but not used in the pilot.

**Hardware assumptions:** Apple Silicon Mac with ≥16 GB unified memory, or a single CUDA GPU with ≥16 GB. The 3B model in fp16 needs ~6 GB; tighten the sample count if memory is tight.

**What this pilot is NOT:** It does not train probes, run baselines, or compute selectivity. Those come after this foundation is verified. This pilot's sole job is: "does the inference + hidden-state-extraction pipeline produce sensibly-shaped outputs on real BLINK data?"

---

## File Structure

```
pilot/
├── env/                    # venv (gitignored)
├── .gitignore
├── README.md
├── plan.md                 # this file
├── requirements.txt
├── config.py               # all knobs in one place
├── data_loader.py          # BLINK from HuggingFace
├── model_setup.py          # model load + forward hooks
├── pilot.py                # entry point — runs the pipeline
├── smoke_test.py           # imports + config check, no model download
└── outputs/                # hidden states + predictions (gitignored)
```

Each file has one responsibility. `config.py` is the single source of truth for what model / how many samples / which layers / which device. The other files import from it; no magic numbers anywhere else.

---

## Task 1: Folder scaffolding

**Files:**
- Create: `pilot/.gitignore`
- Create: `pilot/requirements.txt`
- Create: `pilot/README.md`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
env/
__pycache__/
*.pyc
outputs/
.DS_Store
*.npz
*.npy
.ipynb_checkpoints/
```

- [ ] **Step 2: Write `requirements.txt`**

```
torch>=2.2
transformers>=4.49.0
accelerate>=0.34
datasets>=2.20
Pillow>=10
numpy>=1.26
scikit-learn>=1.4
qwen-vl-utils
```

- [ ] **Step 3: Write `README.md`**

```markdown
# Pilot

Minimal end-to-end pipeline to verify VLM hidden-state extraction on BLINK.

## Setup

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Smoke test (no model download)

```
python smoke_test.py
```

## Full pilot (downloads ~6 GB model on first run)

```
python pilot.py
```
```

---

## Task 2: Config

**Files:**
- Create: `pilot/config.py`

- [ ] **Step 1: Write config.py**

```python
"""Single source of truth for pilot settings.

Change ONE place to swap models, devices, or sample counts.
"""
from pathlib import Path
import torch

# --- Model ---
# 3B for Mac/pilot; switch to "Qwen/Qwen2.5-VL-7B-Instruct" on cluster.
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

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

# --- BLINK subset for pilot ---
BLINK_TASK = "Spatial_Relation"   # one of 14 BLINK task names; small, perception-grounded
NUM_SAMPLES = 5                    # keep tiny for pilot; scale later

# --- Probing locations (LLM decoder layer indices) ---
# Qwen2.5-VL-3B has 36 LLM decoder layers (hidden_size=2048).
# Qwen2.5-VL-7B has 28 LLM decoder layers (hidden_size=3584).
# We auto-detect at runtime; this is just the default sample if extraction succeeds.
# For pilot: extract ALL layers; reporting only verifies shape.

# --- IO ---
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
```

- [ ] **Step 2: Sanity-print config**

```python
# Append to config.py for ad-hoc inspection
if __name__ == "__main__":
    print(f"MODEL_NAME   = {MODEL_NAME}")
    print(f"DEVICE       = {DEVICE}")
    print(f"DTYPE        = {DTYPE}")
    print(f"BLINK_TASK   = {BLINK_TASK}")
    print(f"NUM_SAMPLES  = {NUM_SAMPLES}")
    print(f"OUTPUTS_DIR  = {OUTPUTS_DIR}")
```

Run: `python config.py` — expect printout with no errors.

---

## Task 3: BLINK loader

**Files:**
- Create: `pilot/data_loader.py`

- [ ] **Step 1: Write the loader**

```python
"""Load a small slice of BLINK from HuggingFace.

BLINK has 14 task types; we use one perception-grounded task for the pilot.
Each example yields (image, question, options_dict, answer_letter).
"""
from datasets import load_dataset
from PIL import Image
from typing import Iterator, Tuple, Dict

import config


def load_blink_samples(task: str = config.BLINK_TASK,
                       n: int = config.NUM_SAMPLES,
                       split: str = "val") -> Iterator[Tuple[Image.Image, str, Dict[str, str], str]]:
    """Yield n examples from BLINK[task][split].

    Returns tuples: (PIL image, question text, options dict {A:..., B:..., ...}, answer letter).
    """
    ds = load_dataset("BLINK-Benchmark/BLINK", task, split=split)

    for i in range(min(n, len(ds))):
        ex = ds[i]
        # BLINK schema: image_1, image_2, ..., question, choices, answer
        # We take only the first image for the pilot (simpler than multi-image flow).
        image = ex["image_1"]
        question = ex["question"]
        choices = ex["choices"]    # list like ["A. left", "B. right", ...]
        answer = ex["answer"]      # letter like "A"

        # Parse "A. text" into {"A": "text"}
        options = {}
        for c in choices:
            parts = c.split(". ", 1)
            if len(parts) == 2:
                options[parts[0].strip()] = parts[1].strip()
            else:
                options[c[0]] = c[2:].strip()

        yield image, question, options, answer


if __name__ == "__main__":
    print(f"Loading {config.NUM_SAMPLES} samples from BLINK[{config.BLINK_TASK}]...")
    for i, (img, q, opts, ans) in enumerate(load_blink_samples()):
        print(f"\n[{i}] Q: {q[:80]}")
        print(f"    options: {opts}")
        print(f"    answer:  {ans}")
        print(f"    image:   {img.size} {img.mode}")
```

- [ ] **Step 2: Smoke-run the loader**

Run: `python data_loader.py`
Expected: 5 examples printed with question text, options, answer letter, and image size. First run downloads the BLINK split (~MB, fast).

---

## Task 4: Model loader + forward hooks

**Files:**
- Create: `pilot/model_setup.py`

- [ ] **Step 1: Write the loader + hook machinery**

```python
"""Load Qwen2.5-VL and register forward hooks on every LLM decoder layer.

Hooks capture the last-token hidden state per sample into a dict keyed by layer index.
Always remove hooks in a try/finally — leaked hooks accumulate tensors and silently OOM.
"""
from contextlib import contextmanager
from typing import Dict, List
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import config


def load_model_and_processor():
    """Returns (model, processor). Model is in eval mode on config.DEVICE."""
    print(f"Loading {config.MODEL_NAME} on {config.DEVICE} ({config.DTYPE})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=config.DTYPE,
        device_map=config.DEVICE if config.DEVICE != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if config.DEVICE == "mps":
        model = model.to("mps")
    model.eval()

    processor = AutoProcessor.from_pretrained(config.MODEL_NAME)

    n_layers = len(model.model.language_model.layers) if hasattr(model.model, "language_model") else len(model.model.layers)
    print(f"  loaded. LLM decoder has {n_layers} layers.")
    return model, processor


@contextmanager
def capture_last_token_hidden_states(model, store: Dict[int, torch.Tensor]):
    """Context manager that hooks every LLM decoder layer.

    On exit, hooks are removed even if the forward pass raises. `store` is mutated
    in-place: store[layer_idx] = tensor of shape (batch, hidden_dim).
    """
    # Qwen2_5_VL exposes the LLM at model.model.language_model.layers in newer transformers,
    # and model.model.layers in older ones. Handle both.
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers

    hooks = []

    def make_hook(idx: int):
        def fn(module, inputs, output):
            # decoder layer output is a tuple; first element is hidden_states (B, T, D)
            hs = output[0] if isinstance(output, tuple) else output
            # Take last token of the sequence
            store[idx] = hs[:, -1, :].detach().to("cpu").float()
        return fn

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))
        yield store
    finally:
        for h in hooks:
            h.remove()


def run_single(model, processor, image, question: str) -> tuple[str, Dict[int, torch.Tensor]]:
    """One inference pass. Returns (generated_text, {layer_idx: hidden_state})."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

    hidden_states: Dict[int, torch.Tensor] = {}
    with capture_last_token_hidden_states(model, hidden_states), torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    # Strip the input prompt from the generated tokens before decoding
    gen_ids = out[:, inputs.input_ids.shape[1]:]
    generated = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return generated, hidden_states
```

- [ ] **Step 2: Verify the hook machinery compiles**

Run: `python -c "import model_setup; print('ok')"`
Expected: prints `ok` (no model download yet, just import check).

---

## Task 5: Pilot entry point

**Files:**
- Create: `pilot/pilot.py`

- [ ] **Step 1: Write the orchestrator**

```python
"""End-to-end pilot: load BLINK -> load model -> infer -> extract -> save.

This is the smallest run that proves the full pipeline works.
"""
import numpy as np
import torch

import config
from data_loader import load_blink_samples
from model_setup import load_model_and_processor, run_single


def main():
    print("=== Pilot run ===")
    print(f"Model:   {config.MODEL_NAME}")
    print(f"Device:  {config.DEVICE} ({config.DTYPE})")
    print(f"Task:    {config.BLINK_TASK}")
    print(f"Samples: {config.NUM_SAMPLES}")
    print()

    model, processor = load_model_and_processor()
    samples = list(load_blink_samples())

    all_layer_states: list[np.ndarray] = []     # final shape (N, num_layers, hidden_dim)
    records = []

    for i, (image, question, options, answer) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Q: {question[:70]}")
        try:
            generated, hidden = run_single(model, processor, image, question)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        layer_indices = sorted(hidden.keys())
        stacked = torch.stack([hidden[k] for k in layer_indices], dim=1).squeeze(0).numpy()
        all_layer_states.append(stacked)

        print(f"  generated: {generated[:80]!r}")
        print(f"  gold:      {answer}  options={options}")
        print(f"  hidden_states: {stacked.shape}  dtype={stacked.dtype}")

        records.append({
            "idx": i,
            "question": question,
            "options": options,
            "gold": answer,
            "generated": generated,
        })

    if not all_layer_states:
        print("\nNo samples succeeded. Aborting save.")
        return

    arr = np.stack(all_layer_states, axis=0)
    out_path = config.OUTPUTS_DIR / "pilot_hidden_states.npz"
    np.savez_compressed(
        out_path,
        hidden_states=arr,
        gold=np.array([r["gold"] for r in records]),
        generated=np.array([r["generated"] for r in records]),
        questions=np.array([r["question"] for r in records]),
    )
    print(f"\nSaved {arr.shape} -> {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lock in expected shape**

Run: `python pilot.py`
Expected output (for 3B model, 5 samples): each sample prints `hidden_states: (36, 2048)` and the final save shows `(5, 36, 2048)`. First run will download the model (~6 GB), which takes minutes. Subsequent runs load from cache.

---

## Task 6: Smoke test (no model download)

**Files:**
- Create: `pilot/smoke_test.py`

- [ ] **Step 1: Write smoke test**

```python
"""Lightweight checks that pass without downloading the model.

Verifies: imports, config, BLINK loader, device detection.
Fails fast if anything is broken before you wait for a model download.
"""
import sys


def main():
    print("--- Smoke test ---")

    # 1. Imports
    try:
        import torch
        import transformers
        import datasets
        import numpy
        from PIL import Image
        print(f"  torch:        {torch.__version__}")
        print(f"  transformers: {transformers.__version__}")
        print(f"  datasets:     {datasets.__version__}")
        print(f"  numpy:        {numpy.__version__}")
    except Exception as e:
        print(f"  IMPORT FAILED: {e}")
        return 1

    # 2. Config loads
    try:
        import config
        print(f"  device:       {config.DEVICE} ({config.DTYPE})")
        print(f"  model:        {config.MODEL_NAME}")
    except Exception as e:
        print(f"  CONFIG FAILED: {e}")
        return 1

    # 3. BLINK loader works (downloads task subset on first run)
    try:
        from data_loader import load_blink_samples
        samples = list(load_blink_samples(n=1))
        if not samples:
            print("  BLINK FAILED: no samples returned")
            return 1
        img, q, opts, ans = samples[0]
        print(f"  blink:        loaded 1 sample, image={img.size}, answer={ans}")
    except Exception as e:
        print(f"  BLINK FAILED: {e}")
        return 1

    # 4. model_setup imports (no model download)
    try:
        import model_setup
        print(f"  model_setup:  imports cleanly")
    except Exception as e:
        print(f"  model_setup FAILED: {e}")
        return 1

    print("--- All smoke checks passed. Run `python pilot.py` for full pilot. ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run smoke test**

Run: `source env/bin/activate && python smoke_test.py`
Expected: all four sections print OK; final line "All smoke checks passed."

---

## Verification Order (what we actually run)

1. `python3 -m venv env && source env/bin/activate`
2. `pip install -r requirements.txt`
3. `python smoke_test.py` — must pass before downloading the model.
4. `python pilot.py` — only when smoke test is green; this triggers the model download.

---

## What comes AFTER this pilot (NOT in scope here)

These are the real experiments described in `reqs.md`. Each becomes its own focused plan:

- **Probe training**: logistic regression with L2 across all layers and aggregation strategies.
- **Baseline suite**: majority class, layer-0, random-init model, random-labels selectivity, final-token vocabulary one-hot, DiM probe, etc.
- **Amnesic probing (INLP)**: Level 2 causal validation per Elazar et al. 2021.
- **OOD evaluation**: VisRes Bench.
- **Scaling**: switch `MODEL_NAME` to `Qwen2.5-VL-7B-Instruct` on the cluster; rerun.

The pilot's deliverable is a saved `.npz` of `(N, num_layers, hidden_dim)` hidden states plus the generated answers and gold labels — the input every downstream phase consumes.
