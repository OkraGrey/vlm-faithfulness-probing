"""Phase 1: Real inference run with checkpointing.

Runs Qwen2.5-VL across multiple BLINK tasks, saves per-sample .npz files
to outputs/inference/{task}/{idx:04d}.npz so the run is fully resumable.

Each .npz contains:
  - hidden_states: (num_layers, hidden_dim) float32
  - task:          str
  - idx:           int  (within-task index)
  - question:      str
  - options:       JSON-encoded dict
  - gold:          str  (gold letter)
  - generated:     str  (model's raw output)
  - parsed:        str or "" if unparseable
  - correct:       bool

On restart, samples that already have a saved file are skipped.
Per-sample exceptions are logged but do not abort the whole run.
"""
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import config
from data_loader import load_blink_multi
from model_setup import load_model_and_processor, run_single
from answer_parser import parse_answer, is_correct


def sample_path(task: str, idx: int) -> Path:
    """Where one sample's output lives."""
    return config.INFERENCE_DIR / task / f"{idx:04d}.npz"


def already_done(task: str, idx: int) -> bool:
    return sample_path(task, idx).exists()


def save_sample(task: str, idx: int, hidden_layers: dict, question: str,
                options: dict, gold: str, generated: str, parsed: str | None) -> None:
    """Stack hidden states from layer dict and write everything to one .npz."""
    out_path = sample_path(task, idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layer_indices = sorted(hidden_layers.keys())
    stacked = torch.stack([hidden_layers[k] for k in layer_indices], dim=1).squeeze(0).numpy()

    np.savez_compressed(
        out_path,
        hidden_states=stacked,
        task=task,
        idx=idx,
        question=question,
        options=json.dumps(options),
        gold=gold,
        generated=generated,
        parsed=parsed if parsed is not None else "",
        correct=bool(is_correct(parsed, gold)),
    )


def summarize_existing() -> dict:
    """Inspect the inference dir and report how many samples are already done per task."""
    summary = {}
    for task_dir in config.INFERENCE_DIR.iterdir() if config.INFERENCE_DIR.exists() else []:
        if task_dir.is_dir():
            summary[task_dir.name] = len(list(task_dir.glob("*.npz")))
    return summary


def main():
    print("=== Phase 1: Inference run ===")
    print(f"Model:            {config.MODEL_NAME}")
    print(f"Device:           {config.DEVICE} ({config.DTYPE})")
    print(f"Tasks:            {config.BLINK_TASKS}")
    print(f"Samples per task: {config.SAMPLES_PER_TASK}")
    print(f"Output dir:       {config.INFERENCE_DIR}")
    existing = summarize_existing()
    if existing:
        print(f"Existing samples: {existing}")
    print()

    model, processor = load_model_and_processor()

    n_done = 0
    n_skipped = 0
    n_errors = 0
    n_correct = 0
    n_parsed = 0
    # total_planned is approximate when SAMPLES_PER_TASK exceeds actual split sizes
    total_planned = "(determined by BLINK split sizes)"

    t_start = time.time()

    for task, idx, images, question, options, gold in load_blink_multi():
        if already_done(task, idx):
            n_skipped += 1
            continue

        seen = n_done + n_skipped + n_errors + 1
        print(f"[{seen}] {task}[{idx}] imgs={len(images)} Q: {question[:50]}")
        try:
            generated, hidden = run_single(model, processor, images, question, options=options)
        except Exception:
            n_errors += 1
            print(f"  ERROR:\n{traceback.format_exc().splitlines()[-1]}")
            continue

        parsed = parse_answer(generated, options)
        correct = is_correct(parsed, gold)

        save_sample(task, idx, hidden, question, options, gold, generated, parsed)
        n_done += 1
        if parsed is not None:
            n_parsed += 1
            if correct:
                n_correct += 1

        marker = "✓" if correct else ("?" if parsed is None else "✗")
        print(f"  {marker}  gold={gold}  parsed={parsed!r:<5}  raw={generated[:50]!r}")

    elapsed = time.time() - t_start
    print()
    print("=== Done ===")
    print(f"Newly completed: {n_done}")
    print(f"Skipped (resumed): {n_skipped}")
    print(f"Errors: {n_errors}")
    print(f"Parsed: {n_parsed}/{n_done}  ({100 * n_parsed / max(n_done, 1):.0f}%)")
    print(f"Correct (of parsed): {n_correct}/{n_parsed}  ({100 * n_correct / max(n_parsed, 1):.0f}%)")
    print(f"Wrong (contrastive set): {n_parsed - n_correct}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed / max(n_done, 1):.1f}s per new sample)")


if __name__ == "__main__":
    main()
