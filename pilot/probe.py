"""First probe experiment — does any layer encode the gold answer in hidden states?

Per task, layer-wise, we train a linear probe (logistic regression with L2) on hidden
states to predict the GOLD letter. Then split test set by whether the model itself was
correct, and report:

  acc_all     — probe accuracy on all test samples
  acc_correct — probe accuracy on test samples where the MODEL was right
  acc_wrong   — probe accuracy on test samples where the MODEL was wrong   ← the headline

The hypothesis: if `acc_wrong` significantly exceeds the majority-class baseline at any
layer, the model internally encodes the right answer even when its output is wrong.

This is a first-look sanity check, not the final analysis. Selectivity, random-init
baseline, multiple seeds, bootstrap CIs come in later phases.
"""
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def load_dataset():
    d = np.load(config.OUTPUTS_DIR / "dataset.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def probe_one(X_layer, y, y_correct, seed=0, C=1.0):
    """Train one logistic-regression probe; return per-subset accuracies."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_classes = len(np.unique(y_enc))
    if n_classes < 2:
        return None

    # Stratify by gold so both classes are in train+test
    idx = np.arange(len(y_enc))
    idx_tr, idx_te = train_test_split(idx, test_size=0.25, random_state=seed,
                                      stratify=y_enc)

    # Standardize features (helps L2 regularization behave consistently across layers)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_layer[idx_tr])
    X_te = scaler.transform(X_layer[idx_te])

    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="lbfgs")
    clf.fit(X_tr, y_enc[idx_tr])

    pred = clf.predict(X_te)
    y_te = y_enc[idx_te]
    yc_te = y_correct[idx_te]

    acc_all = accuracy_score(y_te, pred)
    acc_correct = accuracy_score(y_te[yc_te], pred[yc_te]) if yc_te.any() else float("nan")
    acc_wrong = accuracy_score(y_te[~yc_te], pred[~yc_te]) if (~yc_te).any() else float("nan")

    return {
        "n_classes": int(n_classes),
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
        "n_test_correct": int(yc_te.sum()),
        "n_test_wrong": int((~yc_te).sum()),
        "acc_all": float(acc_all),
        "acc_correct": float(acc_correct),
        "acc_wrong": float(acc_wrong),
        "majority_baseline": float(max(np.bincount(y_enc)) / len(y_enc)),
    }


def probe_task_layerwise(data, task, num_layers=36, seed=0):
    """Run the probe at every layer for one task; return list of result dicts."""
    mask = data["task"] == task
    X = data["X"][mask]            # (n, 36, 2048)
    y = data["gold"][mask]
    yc = data["y_correct"][mask]

    # Filter out samples that contain NaN at any layer (MPS fp16 numerical instability
    # produced NaN in some hidden states; ~1-2% of samples typically).
    nan_mask = np.isnan(X).any(axis=(1, 2))
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        X = X[~nan_mask]
        y = y[~nan_mask]
        yc = yc[~nan_mask]
        print(f"  {task}: dropped {n_dropped} NaN sample(s)")

    n = len(y)
    if n < 30:
        print(f"  {task}: only {n} samples after NaN filter, skipping")
        return []

    n_wrong = int((~yc).sum())
    n_correct = int(yc.sum())
    print(f"  {task}: n={n}  correct={n_correct}  wrong={n_wrong}  options={len(set(y))}")

    rows = []
    for L in range(num_layers):
        r = probe_one(X[:, L, :], y, yc, seed=seed)
        if r is None:
            continue
        r["task"] = task
        r["layer"] = L
        rows.append(r)
    return rows


def print_task_table(task, rows):
    print(f"\n=== {task} ===")
    print(f"{'layer':>5} {'acc_all':>8} {'acc_corr':>9} {'acc_wrong':>10} {'majority':>9}")
    for r in rows:
        print(f"{r['layer']:>5d} {r['acc_all']:>8.3f} {r['acc_correct']:>9.3f} "
              f"{r['acc_wrong']:>10.3f} {r['majority_baseline']:>9.3f}")
    # peak layers
    best_all = max(rows, key=lambda r: r["acc_all"])
    best_wrong = max(rows, key=lambda r: (r["acc_wrong"] if not np.isnan(r["acc_wrong"]) else -1))
    print(f"  peak overall: layer {best_all['layer']}  acc_all={best_all['acc_all']:.3f}")
    print(f"  peak wrong:   layer {best_wrong['layer']}  acc_wrong={best_wrong['acc_wrong']:.3f}  "
          f"(vs majority={best_wrong['majority_baseline']:.3f})")


def main():
    print("Loading dataset...")
    data = load_dataset()
    print(f"  X.shape = {data['X'].shape}")
    print(f"  tasks   = {sorted(set(data['task']))}")
    print()

    all_rows = []
    for task in sorted(set(data["task"])):
        rows = probe_task_layerwise(data, task)
        all_rows.extend(rows)
        if rows:
            print_task_table(task, rows)

    # Save full results
    out = config.OUTPUTS_DIR / "probe_results_v1.json"
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved {len(all_rows)} probe rows -> {out}")

    # Cross-task headline
    print("\n=== Headline: peak wrong-subset accuracy per task ===")
    print(f"{'task':<28s} {'best_layer':>10} {'acc_wrong':>10} {'majority':>10} {'gap':>8}")
    for task in sorted(set(r["task"] for r in all_rows)):
        rows = [r for r in all_rows if r["task"] == task]
        valid = [r for r in rows if not np.isnan(r["acc_wrong"])]
        if not valid:
            continue
        best = max(valid, key=lambda r: r["acc_wrong"])
        gap = best["acc_wrong"] - best["majority_baseline"]
        marker = "←" if gap > 0.10 else ""
        print(f"{task:<28s} {best['layer']:>10d} {best['acc_wrong']:>10.3f} "
              f"{best['majority_baseline']:>10.3f} {gap:>+8.3f} {marker}")


if __name__ == "__main__":
    main()
