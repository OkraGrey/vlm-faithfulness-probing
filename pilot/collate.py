"""Combine all per-sample .npz files into one consolidated dataset.

Reading 1,901 separate .npz files for every probe experiment is slow and clumsy.
This script walks outputs/inference/ once and produces outputs/dataset.npz with:

  X            (N, num_layers, hidden_dim) float32  — hidden states
  y_correct    (N,)  bool   — model got the question right
  task         (N,)  str    — BLINK task name
  gold         (N,)  str    — gold answer letter
  parsed       (N,)  str    — model's parsed letter ('' if unparseable)
  generated    (N,)  str    — model's raw output
  question     (N,)  str
  options      (N,)  str    — JSON-encoded options dict
  sample_id    (N,)  str    — "TaskName/0042" — for joining back to per-sample files

Run once after run_inference.py completes.
"""
import json
from pathlib import Path

import numpy as np

import config


def collate(inference_dir: Path = config.INFERENCE_DIR,
            out_path: Path = config.OUTPUTS_DIR / "dataset.npz") -> Path:
    files = sorted(inference_dir.rglob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files in {inference_dir} — run run_inference.py first.")

    print(f"Collating {len(files)} per-sample files...")

    X_list = []
    y_correct = []
    task = []
    gold = []
    parsed = []
    generated = []
    question = []
    options = []
    sample_id = []

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        X_list.append(d["hidden_states"])
        y_correct.append(bool(d["correct"]))
        task.append(str(d["task"]))
        gold.append(str(d["gold"]))
        parsed.append(str(d["parsed"]))
        generated.append(str(d["generated"]))
        question.append(str(d["question"]))
        options.append(str(d["options"]))
        sample_id.append(f"{d['task']}/{int(d['idx']):04d}")

    X = np.stack(X_list, axis=0).astype(np.float32)

    print(f"  X.shape       = {X.shape}  ({X.nbytes / 1e6:.0f} MB)")
    print(f"  n_correct     = {sum(y_correct)}")
    print(f"  n_wrong       = {len(y_correct) - sum(y_correct)}")
    print(f"  unique tasks  = {len(set(task))}")

    np.savez_compressed(
        out_path,
        X=X,
        y_correct=np.array(y_correct, dtype=bool),
        task=np.array(task),
        gold=np.array(gold),
        parsed=np.array(parsed),
        generated=np.array(generated),
        question=np.array(question),
        options=np.array(options),
        sample_id=np.array(sample_id),
    )
    print(f"Saved -> {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB compressed)")
    return out_path


if __name__ == "__main__":
    collate()
