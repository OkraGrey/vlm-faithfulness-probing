"""G3 analysis: is the GOLD-reading signal visual or a text/format artifact?

For each GOLD-reading candidate (Jigsaw, Multi-view_Reasoning, Art_Style) we compare,
on the SAME wrong-output subset (wrongness defined by the real-image run), the DiM
probe's gold-recovery from:
  - REAL-image hidden states  (outputs/inference/{task})
  - NOISE-image hidden states (outputs/inference_noise/{task}, from run_inference_noise.py)

visual_attribution = real_gold_acc_wrong - noise_gold_acc_wrong.

Decision (per task):
  real high AND noise ~ majority (collapses)  -> signal is VISUAL  -> H1 holds
  real high AND noise also high               -> signal is ARTIFACT -> H1 dies (cautionary)

Run AFTER run_inference_noise.py finishes:  ./env/bin/python gate3_construct.py
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

import config
import gate0_cv as g

# Default = 3B GOLD-reading candidates; override: export CANDIDATE_TASKS="taskA,taskB"
CANDIDATE_TASKS = os.environ.get(
    "CANDIDATE_TASKS", "Jigsaw,Multi-view_Reasoning,Art_Style").split(",")
NOISE_DIR = config.OUTPUTS_DIR / "inference_noise"
REAL_DIR = config.INFERENCE_DIR
SEEDS = [0, 1, 2, 3, 4]


def load_dir(task, base):
    """Load per-sample npz for a task into dict idx -> (hidden(36,2048), gold, correct, parsed)."""
    out = {}
    d = base / task
    if not d.exists():
        return out
    for fp in d.glob("*.npz"):
        z = np.load(fp, allow_pickle=True)
        out[int(z["idx"])] = (
            z["hidden_states"].astype(np.float32),
            str(z["gold"]),
            bool(z["correct"]),
            str(z["parsed"]),
        )
    return out


def matched_arrays(task):
    """Return X_real, X_noise, gold, correct_real for idx present in BOTH runs."""
    real = load_dir(task, REAL_DIR)
    noise = load_dir(task, NOISE_DIR)
    idxs = sorted(set(real) & set(noise))
    Xr = np.stack([real[i][0] for i in idxs])
    Xn = np.stack([noise[i][0] for i in idxs])
    gold = np.array([real[i][1] for i in idxs])
    correct = np.array([real[i][2] for i in idxs], dtype=bool)
    return Xr, Xn, gold, correct, len(idxs), len(real), len(noise)


def gold_recovery_wrong(X, y_enc, correct):
    """Mean (over seeds) DiM nested-CV gold accuracy on the wrong-output subset, + bootstrap CI."""
    wrong = ~correct
    accs = []
    canonical = None
    for seed in SEEDS:
        pdim, _, _ = g.nested_cv_once(X, y_enc, correct, seed=seed)
        accs.append(g.subset_acc(y_enc, pdim, wrong))
        if seed == 0:
            canonical = pdim
    ci = g.bootstrap_ci(y_enc[wrong], canonical[wrong])
    return float(np.nanmean(accs)), float(np.nanstd(accs)), ci, int(wrong.sum())


def main():
    print("=== G3: real vs noise gold-recovery on the wrong-output subset ===\n")
    results = []
    for task in CANDIDATE_TASKS:
        Xr, Xn, gold, correct, nmatch, nreal, nnoise = matched_arrays(task)
        if nmatch == 0:
            print(f"  {task}: no matched samples yet (real={nreal}, noise={nnoise}) — run inference first.")
            continue
        # NaN filter aligned across both runs
        keep = ~(np.isnan(Xr).any(axis=(1, 2)) | np.isnan(Xn).any(axis=(1, 2)))
        Xr, Xn, gold, correct = Xr[keep], Xn[keep], gold[keep], correct[keep]

        le = LabelEncoder()
        y_enc = le.fit_transform(gold)
        counts = np.bincount(y_enc)
        majority = float(counts.max() / counts.sum())

        real_acc, real_sd, real_ci, n_wrong = gold_recovery_wrong(Xr, y_enc, correct)
        noise_acc, noise_sd, noise_ci, _ = gold_recovery_wrong(Xn, y_enc, correct)
        attribution = real_acc - noise_acc

        # verdict
        noise_collapses = noise_ci[1] < real_ci[0]  # noise CI strictly below real CI
        near_majority = abs(noise_acc - majority) < 0.10
        if attribution > 0.15 and (noise_collapses or near_majority):
            verdict = "VISUAL (H1 holds)"
        elif attribution < 0.10:
            verdict = "ARTIFACT (H1 dies — non-visual)"
        else:
            verdict = "PARTIAL / inconclusive"

        print(f"  {task}  (n={len(y_enc)}, wrong={n_wrong}, options={len(le.classes_)}, majority={majority:.3f})")
        print(f"    real-image  gold-acc(wrong) = {real_acc:.3f} ± {real_sd:.3f}  CI[{real_ci[0]:.2f},{real_ci[1]:.2f}]")
        print(f"    noise-image gold-acc(wrong) = {noise_acc:.3f} ± {noise_sd:.3f}  CI[{noise_ci[0]:.2f},{noise_ci[1]:.2f}]")
        print(f"    visual_attribution = {attribution:+.3f}   -> {verdict}\n")

        results.append({
            "task": task, "n": int(len(y_enc)), "n_wrong": n_wrong,
            "majority": majority,
            "real_gold_acc_wrong": real_acc, "real_ci": real_ci,
            "noise_gold_acc_wrong": noise_acc, "noise_ci": noise_ci,
            "visual_attribution": attribution, "verdict": verdict,
        })

    if results:
        out = config.OUTPUTS_DIR / "gate3_construct_summary.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
