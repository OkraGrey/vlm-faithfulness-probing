"""Gate 0: Honest re-confirmation of Gate 1 survivors under nested cross-validation.

Gate 1 (baselines.py) selected, per task, the best of 36 layers by wrong-subset
accuracy on a single 25% test split (4-25 wrong-test samples), then reported that
layer's CI uncorrected. That is a winner's-curse / selection-bias estimate.

This module re-estimates the signal HONESTLY:

  1. Nested k-fold CV. Outer folds give the evaluation; the probe layer is selected
     on INNER folds of the outer-train only (the held-out fold never influences
     layer choice). Pooled across outer folds, every sample is predicted exactly
     once -> the wrong-subset accuracy is computed on the FULL wrong set (57-91),
     not a 15-23 sample slice.

  2. Difference-in-means is the PRIMARY probe (Marks & Tegmark 2023, arXiv:2310.06824):
     low-capacity, overfits less at 2048-dim x ~70 samples, and is the direction we
     carry into the causal stage (activation patching, G4). L2 logistic regression
     is reported as a secondary, complementary probe.

  3. Selectivity (Hewitt & Liang 2019, arXiv:1909.03368) is recomputed under the same
     nested-CV protocol with fixed-shuffled labels.

We also report, for comparability with Gate 1, a FIXED-LAYER CV estimate at the
layer baselines_summary.json originally named for each task, and the distribution of
inner-CV-selected layers (a scattered distribution => the original peak was noise).

passes_gate0 (the honest analogue of passes_gate1):
  - pooled wrong-subset bootstrap CI excludes the majority baseline, AND
  - selectivity_wrong > 0.10

Run:  ./env/bin/python gate0_cv.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 10_000
OUTER_SPLITS = 5
INNER_SPLITS = 4
PROBE_C = 1.0
PROBE_MAX_ITER = 1000
SELECTIVITY_THRESHOLD = 0.10
MIN_TASK_N = 30


# ---------------- data ----------------

def load_dataset():
    d = np.load(config.OUTPUTS_DIR / "dataset.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def filter_nan(X, *aligned):
    """Drop samples where ANY layer of X contains NaN."""
    nan_mask = np.isnan(X).any(axis=tuple(range(1, X.ndim)))
    keep = ~nan_mask
    return (X[keep], *(a[keep] for a in aligned), int(nan_mask.sum()))


def load_original_best_layers():
    """Read the Gate-1 best_layer per task for the fixed-layer comparison."""
    p = config.OUTPUTS_DIR / "baselines_summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        rows = json.load(f)
    return {r["task"]: r["best_layer"] for r in rows}


# ---------------- probes ----------------

def dim_predict(X_train, y_train, X_test):
    """Difference-in-means: predict the class whose train mean is nearest (L2)."""
    classes = np.unique(y_train)
    means = np.stack([X_train[y_train == c].mean(axis=0) for c in classes])
    d = np.linalg.norm(X_test[:, None, :] - means[None, :, :], axis=2)
    return classes[d.argmin(axis=1)]


def logreg_predict(X_train, y_train, X_test, seed):
    clf = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=seed,
                             solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def _safe_splits(y, max_splits):
    """Largest n_splits <= max_splits that StratifiedKFold can satisfy."""
    min_class = min(Counter(y).values())
    return max(2, min(max_splits, min_class))


# ---------------- nested CV ----------------

def select_layer_inner(Xtr_all, ytr, seed):
    """Pick the layer with best mean inner-CV DiM accuracy, using outer-train only.

    Xtr_all: (n_train, n_layers, dim). Returns the chosen layer index.
    Layer choice NEVER sees the outer-test fold => no selection leakage.
    """
    n_layers = Xtr_all.shape[1]
    n_inner = _safe_splits(ytr, INNER_SPLITS)
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 7)
    layer_acc = np.zeros(n_layers)
    splits = list(skf.split(Xtr_all[:, 0, :], ytr))
    for L in range(n_layers):
        accs = []
        for itr, iva in splits:
            sc = StandardScaler()
            xa = sc.fit_transform(Xtr_all[itr, L, :])
            xb = sc.transform(Xtr_all[iva, L, :])
            pred = dim_predict(xa, ytr[itr], xb)
            accs.append((pred == ytr[iva]).mean())
        layer_acc[L] = np.mean(accs)
    return int(layer_acc.argmax())


def nested_cv_once(X, y, yc, seed, fixed_layer=None):
    """One nested-CV pass. Returns pooled outer-test predictions and selected layers.

    If fixed_layer is not None, skip inner selection and use that layer everywhere.
    """
    n = len(y)
    n_outer = _safe_splits(y, OUTER_SPLITS)
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    pred_dim = np.full(n, -1, dtype=int)
    pred_lr = np.full(n, -1, dtype=int)
    chosen_layers = []
    seen = np.zeros(n, dtype=bool)

    for otr, ote in skf.split(X[:, 0, :], y):
        # leakage guard: train/test index sets must be disjoint
        assert not (set(otr) & set(ote)), "train/test overlap"
        seen[ote] = True

        L = fixed_layer if fixed_layer is not None else select_layer_inner(X[otr], y[otr], seed)
        chosen_layers.append(L)

        sc = StandardScaler()
        xa = sc.fit_transform(X[otr, L, :])
        xb = sc.transform(X[ote, L, :])

        pred_dim[ote] = dim_predict(xa, y[otr], xb)
        pred_lr[ote] = logreg_predict(xa, y[otr], xb, seed)

    assert seen.all(), "every sample must be tested exactly once"
    return pred_dim, pred_lr, chosen_layers


def subset_acc(y, pred, mask):
    if not mask.any():
        return float("nan")
    return float((y[mask] == pred[mask]).mean())


def bootstrap_ci(y_true, y_pred, seed=0, n=N_BOOTSTRAP, alpha=0.05):
    rng = np.random.default_rng(seed)
    m = len(y_true)
    if m == 0:
        return float("nan"), float("nan")
    accs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        accs[i] = (y_true[idx] == y_pred[idx]).mean()
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


# ---------------- per task ----------------

def process_task(data, task, orig_best):
    mask = data["task"] == task
    X = data["X"][mask]
    y = data["gold"][mask]
    yc = data["y_correct"][mask].astype(bool)

    X, y, yc, n_dropped = filter_nan(X, y, yc)
    if len(y) < MIN_TASK_N:
        return None

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if len(le.classes_) < 2:
        return None
    counts = np.bincount(y_enc)
    majority = float(counts.max() / counts.sum())
    maj_class = int(counts.argmax())
    wrong = ~yc

    print(f"  {task}: n={len(y)} wrong={int(wrong.sum())} options={len(le.classes_)} "
          f"majority={majority:.3f} (dropped {n_dropped} NaN)")

    # --- nested CV with internal layer selection, across seeds ---
    dim_wrong, dim_all, lr_wrong, sel_wrong, all_chosen = [], [], [], [], []
    canonical = None
    for seed in SEEDS:
        pdim, plr, chosen = nested_cv_once(X, y_enc, yc, seed)
        # shuffled-label control (selectivity), same protocol
        rng = np.random.default_rng(seed + 10_000)
        y_shuf = y_enc.copy()
        rng.shuffle(y_shuf)
        pdim_ctrl, _, _ = nested_cv_once(X, y_shuf, yc, seed)

        dim_wrong.append(subset_acc(y_enc, pdim, wrong))
        dim_all.append(subset_acc(y_enc, pdim, np.ones_like(wrong)))
        lr_wrong.append(subset_acc(y_enc, plr, wrong))
        sel_wrong.append(subset_acc(y_enc, pdim, wrong) - subset_acc(y_shuf, pdim_ctrl, wrong))
        all_chosen.extend(chosen)
        if seed == 0:
            canonical = (pdim, plr)

    # bootstrap CI on the pooled seed-0 DiM predictions, wrong subset
    pdim0, plr0 = canonical
    ci_dim = bootstrap_ci(y_enc[wrong], pdim0[wrong])
    ci_lr = bootstrap_ci(y_enc[wrong], plr0[wrong])

    # --- fixed-layer CV at the Gate-1-reported layer (comparability) ---
    fixed = orig_best.get(task)
    fixed_res = None
    if fixed is not None:
        fdim, flr, _ = nested_cv_once(X, y_enc, yc, 0, fixed_layer=fixed)
        fixed_res = {
            "layer": int(fixed),
            "dim_wrong_acc": subset_acc(y_enc, fdim, wrong),
            "lr_wrong_acc": subset_acc(y_enc, flr, wrong),
            "dim_wrong_ci": bootstrap_ci(y_enc[wrong], fdim[wrong]),
        }

    layer_hist = dict(sorted(Counter(all_chosen).items(), key=lambda kv: -kv[1]))
    dim_wrong_mean = float(np.nanmean(dim_wrong))
    sel_wrong_mean = float(np.nanmean(sel_wrong))
    ci_excludes_majority = (not np.isnan(ci_dim[0])) and ci_dim[0] > majority
    sel_pass = sel_wrong_mean > SELECTIVITY_THRESHOLD
    passes = bool(ci_excludes_majority and sel_pass)

    return {
        "task": task,
        "n": int(len(y)),
        "n_wrong": int(wrong.sum()),
        "majority": majority,
        "dim_wrong_acc_mean": dim_wrong_mean,
        "dim_wrong_acc_std": float(np.nanstd(dim_wrong)),
        "dim_wrong_ci": ci_dim,
        "dim_all_acc_mean": float(np.nanmean(dim_all)),
        "lr_wrong_acc_mean": float(np.nanmean(lr_wrong)),
        "lr_wrong_ci": ci_lr,
        "selectivity_wrong_mean": sel_wrong_mean,
        "selected_layer_hist": layer_hist,
        "fixed_layer_cv": fixed_res,
        "selectivity_pass": sel_pass,
        "ci_excludes_majority": ci_excludes_majority,
        "passes_gate0": passes,
    }


def print_table(summaries):
    print("\n=== Gate 0: nested-CV honest re-estimate (DiM primary) ===")
    print(f"{'task':<26} {'n_w':>4} {'maj':>5} {'dimW':>6} {'CI':>14} "
          f"{'selW':>6} {'lrW':>6} {'fixL/acc':>12} {'pass':>5}")
    for s in summaries:
        ci = f"[{s['dim_wrong_ci'][0]:.2f},{s['dim_wrong_ci'][1]:.2f}]"
        fx = s["fixed_layer_cv"]
        fxs = f"L{fx['layer']}/{fx['dim_wrong_acc']:.2f}" if fx else "-"
        flag = "PASS" if s["passes_gate0"] else "."
        print(f"{s['task']:<26} {s['n_wrong']:>4} {s['majority']:>5.3f} "
              f"{s['dim_wrong_acc_mean']:>6.3f} {ci:>14} {s['selectivity_wrong_mean']:>+6.3f} "
              f"{s['lr_wrong_acc_mean']:>6.3f} {fxs:>12} {flag:>5}")
    n_pass = sum(s["passes_gate0"] for s in summaries)
    print(f"\nTasks passing Gate 0: {n_pass}/{len(summaries)} "
          f"(CI-excludes-majority AND selectivity_wrong>{SELECTIVITY_THRESHOLD})")
    print("Layer-selection stability for the headline tasks:")
    for s in summaries:
        if s["task"] in ("Forensic_Detection", "Object_Localization") or s["passes_gate0"]:
            print(f"  {s['task']:<26} selected layers -> {s['selected_layer_hist']}")


def main():
    data = load_dataset()
    orig_best = load_original_best_layers()
    tasks = sorted(set(data["task"]))
    print(f"Loaded X={data['X'].shape}, {len(tasks)} tasks. Original best layers: {orig_best}\n")

    summaries = []
    for task in tasks:
        s = process_task(data, task, orig_best)
        if s:
            summaries.append(s)

    out = config.OUTPUTS_DIR / "gate0_cv_summary.json"
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved -> {out}")
    print_table(summaries)


# ---------------- self-test (leakage / fold integrity) ----------------

def _selftest():
    rng = np.random.default_rng(0)
    n, nl, d = 120, 6, 16
    y = rng.integers(0, 2, size=n)
    # layer 3 carries the signal; others are noise
    X = rng.normal(size=(n, nl, d))
    X[:, 3, :] += y[:, None] * 3.0
    yc = rng.random(n) > 0.5

    pdim, plr, chosen = nested_cv_once(X, y, yc, seed=0)
    assert (pdim >= 0).all() and (plr >= 0).all(), "all samples predicted"
    acc = (pdim == y).mean()
    assert acc > 0.9, f"signal layer should be recovered, got {acc:.2f}"
    # inner selection should usually pick the signal layer 3
    assert Counter(chosen).most_common(1)[0][0] == 3, f"expected layer 3, got {Counter(chosen)}"
    print(f"[selftest] OK  acc={acc:.3f}  chosen_layers={Counter(chosen)}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
