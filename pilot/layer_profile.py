"""Layer profile (REGENERABLE): per-layer DiM gold-recovery on the wrong-output subset.

Reproduces the Gate-3 ADDENDUM diagnostic (docs/03) that forced the weaker, honest claim:
the gold answer for Jigsaw / Multi-view_Reasoning is decodable at ~1.0 from the EARLIEST
layers (incl. L0), i.e. a shallow/early visual cue -- NOT a signal that emerges with depth
the way genuine internal "reason-then-suppress" grounding would (the warning sign flagged in
docs/03 sec 1 and the expectation pre-registered there). This was computed once and described
in prose; this script makes it a saved, regenerable artifact.

For each task and each decoder layer L (0..35), we run honest 5-fold CV with the
difference-in-means probe (Marks & Tegmark 2023, arXiv:2310.06824) at that FIXED layer,
pool predictions so every sample is scored once, and report gold accuracy on the wrong
subset (mean +/- sd over seeds). A profile that is high at L0 == layer-0 artifact signature;
one that rises with depth == emergent grounding.

Run:  ./env/bin/python layer_profile.py
Out:  outputs/layer_profile_summary.json
"""
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
import gate0_cv as g

# Default = 3B GOLD-reading tasks; override: export CANDIDATE_TASKS="taskA,taskB"
TASKS = os.environ.get("CANDIDATE_TASKS", "Jigsaw,Multi-view_Reasoning").split(",")
SEEDS = [0, 1, 2, 3, 4]


def cv_gold_acc_at_layer(XL, y_enc, wrong, seed):
    """Pooled 5-fold CV DiM gold-accuracy on the wrong subset, single fixed layer."""
    n_splits = g._safe_splits(y_enc, 5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.full(len(y_enc), -1)
    for tr, te in skf.split(XL, y_enc):
        sc = StandardScaler()
        xa = sc.fit_transform(XL[tr]); xb = sc.transform(XL[te])
        pred[te] = g.dim_predict(xa, y_enc[tr], xb)
    return g.subset_acc(y_enc, pred, wrong)


def process_task(data, task):
    mask = data["task"] == task
    X = data["X"][mask]
    gold = data["gold"][mask]
    yc = data["y_correct"][mask].astype(bool)
    X, gold, yc, n_drop = g.filter_nan(X, gold, yc)

    le = LabelEncoder()
    y_enc = le.fit_transform(gold)
    wrong = ~yc
    n_layers = X.shape[1]

    profile = []
    for L in range(n_layers):
        accs = [cv_gold_acc_at_layer(X[:, L, :], y_enc, wrong, seed) for seed in SEEDS]
        profile.append({"layer": L, "gold_acc_wrong": round(float(np.nanmean(accs)), 3),
                        "sd": round(float(np.nanstd(accs)), 3)})

    counts = np.bincount(y_enc)
    return {
        "task": task,
        "n_wrong": int(wrong.sum()),
        "majority": round(float(counts.max() / counts.sum()), 3),
        "gold_acc_L0": profile[0]["gold_acc_wrong"],
        "gold_acc_max": max(p["gold_acc_wrong"] for p in profile),
        "emerges_with_depth": bool(profile[0]["gold_acc_wrong"] < 0.70
                                   and max(p["gold_acc_wrong"] for p in profile) - profile[0]["gold_acc_wrong"] > 0.20),
        "profile": profile,
    }


def main():
    data = g.load_dataset()
    summaries = [process_task(data, t) for t in TASKS]

    out = config.OUTPUTS_DIR / "layer_profile_summary.json"
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)

    for s in summaries:
        print(f"\n=== {s['task']}  (n_wrong={s['n_wrong']}, majority={s['majority']}) ===")
        print(f"  gold-acc at L0 = {s['gold_acc_L0']:.2f} | max over layers = {s['gold_acc_max']:.2f} "
              f"| emerges-with-depth = {s['emerges_with_depth']}")
        line = " ".join(f"{p['gold_acc_wrong']:.2f}" for p in s["profile"])
        print(f"  L0..L{len(s['profile'])-1}: {line}")
    print(f"\nSaved -> {out}")
    print("Reading: high at L0 + flat == shallow/early cue (layer-0 artifact); "
          "low-then-rising == grounding that emerges with depth.")


if __name__ == "__main__":
    main()
