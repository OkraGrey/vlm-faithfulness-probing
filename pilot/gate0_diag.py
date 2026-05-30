"""Gate 0 diagnostic (REGENERABLE): does the probe read GOLD, or the model's OWN WRONG OUTPUT?

Reproduces the "gold-vs-output decomposition" that docs/03 (Gate 0) describes as *transient*.
It was computed once during the 2026-05-30 session, flipped the H1 candidate set, but was
never saved to disk. This script promotes it to a first-class, reproducible artifact so the
single most consequential Gate-0 number is backed by a regenerable file, not prose.

Method (identical honest protocol to gate0_cv.py):
  On the WRONG-output subset, train the difference-in-means (DiM) probe under nested 5-fold CV
  (Marks & Tegmark 2023, arXiv:2310.06824; layer chosen on inner folds only), read the probe's
  predicted answer letter for each wrong sample, and classify it:
    -> gold    : prediction == BLINK gold answer        (H1 "sees but can't say" candidate)
    -> output  : prediction == model's parsed wrong output (PAPO-consistent; the internal
                 state already agrees with the wrong answer -- arXiv:2507.06448)
    -> neither : prediction == some third option

Interpretation: a high ->gold fraction marks an H1 candidate task; a high ->output fraction
means there is no hidden correct grounding to recover (supports PAPO's premise for that task).

Run:  ./env/bin/python gate0_diag.py
Out:  outputs/gate0_diag_summary.json
"""
import json

import numpy as np
from sklearn.preprocessing import LabelEncoder

import config
import gate0_cv as g

SEEDS = [0, 1, 2, 3, 4]


def reading(frac_gold, frac_out, frac_neither):
    """Label a task the way docs/03 Gate 0 does, from the dominant destination."""
    if frac_gold >= 0.55:
        return "GOLD (H1)" if frac_gold < 0.85 else "GOLD (H1, strong)"
    if frac_out >= 0.55:
        return "OUTPUT (PAPO)"
    return "mixed/noise"


def process_task(data, task):
    mask = data["task"] == task
    X = data["X"][mask]
    gold = data["gold"][mask]
    parsed = data["parsed"][mask]
    yc = data["y_correct"][mask].astype(bool)

    X, gold, parsed, yc, n_drop = g.filter_nan(X, gold, parsed, yc)
    if len(gold) < g.MIN_TASK_N:
        return None

    le = LabelEncoder()
    y_enc = le.fit_transform(gold)
    if len(le.classes_) < 2:
        return None
    wrong = ~yc
    if not wrong.any():
        return None

    fg, fo, fn = [], [], []
    for seed in SEEDS:
        pdim, _, _ = g.nested_cv_once(X, y_enc, yc, seed)          # honest nested-CV DiM
        pred_letters = le.inverse_transform(pdim)
        pg, gw, ow = pred_letters[wrong], gold[wrong], parsed[wrong]
        to_gold = pg == gw
        to_out = (~to_gold) & (pg == ow)
        to_neither = ~(to_gold | to_out)
        fg.append(to_gold.mean()); fo.append(to_out.mean()); fn.append(to_neither.mean())

    frac_gold, frac_out, frac_neither = float(np.mean(fg)), float(np.mean(fo)), float(np.mean(fn))
    return {
        "task": task,
        "n_wrong": int(wrong.sum()),
        "options": int(len(le.classes_)),
        "frac_to_gold": round(frac_gold, 3),
        "frac_to_output": round(frac_out, 3),
        "frac_to_neither": round(frac_neither, 3),
        "reading": reading(frac_gold, frac_out, frac_neither),
    }


def main():
    data = g.load_dataset()
    tasks = sorted(set(data["task"]))
    summaries = [s for s in (process_task(data, t) for t in tasks) if s]
    # sort by the H1-relevant signal (gold first), like the docs/03 table
    summaries.sort(key=lambda s: -s["frac_to_gold"])

    out = config.OUTPUTS_DIR / "gate0_diag_summary.json"
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)

    print("=== Gate 0 diagnostic: probe reads GOLD vs model's OWN OUTPUT (wrong subset) ===")
    print(f"{'task':<26} {'n_w':>4} {'opt':>3} {'->gold':>7} {'->out':>7} {'->neither':>9}  reading")
    for s in summaries:
        print(f"{s['task']:<26} {s['n_wrong']:>4} {s['options']:>3} "
              f"{s['frac_to_gold']:>7.2f} {s['frac_to_output']:>7.2f} {s['frac_to_neither']:>9.2f}  {s['reading']}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
