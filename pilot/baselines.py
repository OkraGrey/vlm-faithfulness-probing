"""Gate 1: Unified baselines + controls for VLM faithfulness probing.

Runs all Gate 1 experiments on the collated dataset in one batch:

  real_probe        Logistic regression on hidden states (5 seeds, bootstrap CIs)
  random_label_ctrl Same probe, labels shuffled (Hewitt & Liang 2019 selectivity)
  dim_probe         Difference-in-means, zero training (Marks & Tegmark 2023)
  vocab_probe       Logistic regression on the model's parsed-letter one-hot
                    (no hidden state — rules out lexical shortcut)

Per (task, layer) we report:
  acc_all, acc_correct, acc_wrong   for each baseline type
  selectivity = real_acc - random_label_acc
  bootstrap 95% CI on real probe (from seed=0 test set)
  McNemar p-value: real probe vs majority class
  passes_gate1 flag using the criteria documented in methodology_log.md

CRITICAL: train/test splits and feature standardization are identical across
all four baselines for a given (task, layer, seed). Otherwise differences
between baselines reflect split variation, not the experimental manipulation.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import config


SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 10_000
TEST_FRACTION = 0.25
PROBE_C = 1.0
PROBE_MAX_ITER = 2000

# Gate 1 decision thresholds (per methodology_log.md section 2)
SELECTIVITY_THRESHOLD = 0.10
DIM_GAP_THRESHOLD = 0.05


# ---------------- data utilities ----------------

def load_dataset():
    d = np.load(config.OUTPUTS_DIR / "dataset.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def filter_nan(X, *aligned):
    """Drop samples where ANY layer of X contains NaN."""
    nan_mask = np.isnan(X).any(axis=tuple(range(1, X.ndim)))
    keep = ~nan_mask
    return (X[keep], *(a[keep] for a in aligned), int(nan_mask.sum()))


def stratified_split(y_enc, seed):
    """Return (idx_train, idx_test) — stratified by label, deterministic per seed."""
    idx = np.arange(len(y_enc))
    return train_test_split(idx, test_size=TEST_FRACTION, random_state=seed,
                            stratify=y_enc)


# ---------------- probes ----------------

def fit_logreg(X_train, y_train, X_test, seed):
    """Standardize, fit logistic regression with L2, predict test."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    clf = LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER, random_state=seed,
                             solver="lbfgs")
    clf.fit(X_tr, y_train)
    return clf.predict(X_te)


def dim_probe(X_train, y_train, X_test):
    """Zero-trained difference-in-means probe (Marks & Tegmark 2023).

    Per class c, compute mean(X_train[y==c]). Predict the class whose mean is
    closest in Euclidean distance to each test point.
    """
    classes = np.unique(y_train)
    means = np.stack([X_train[y_train == c].mean(axis=0) for c in classes])  # (C, D)
    # broadcasted distances: (n_test, C)
    d = np.linalg.norm(X_test[:, None, :] - means[None, :, :], axis=2)
    return classes[d.argmin(axis=1)]


def vocab_onehot_features(parsed):
    """One-hot encode parsed letters (the model's output) into a small dense matrix.

    Empty/unparseable strings become an all-zeros row (treated as an extra category).
    """
    parsed_clean = np.array([p if p else "_" for p in parsed]).reshape(-1, 1)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return enc.fit_transform(parsed_clean)


# ---------------- per-baseline runners (shared splits) ----------------

def run_one_layer(X_layer, y_enc, y_correct, vocab_feats, seed):
    """Run all four baselines on one (task, layer) with one seed using ONE split.

    Returns dict of per-baseline metrics on test set:
      acc_all, acc_correct, acc_wrong  for each of: real, random_label, dim, vocab.
    Also returns the real-probe test predictions (for downstream McNemar / bootstrap).
    """
    idx_tr, idx_te = stratified_split(y_enc, seed)
    y_te = y_enc[idx_te]
    yc_te = y_correct[idx_te]

    # Shuffled labels for random-label control: same train/test split, shuffled labels
    rng = np.random.default_rng(seed + 10_000)
    y_shuffled = y_enc.copy()
    rng.shuffle(y_shuffled)
    y_shuffled_te = y_shuffled[idx_te]

    out = {}

    # Real probe
    pred_real = fit_logreg(X_layer[idx_tr], y_enc[idx_tr], X_layer[idx_te], seed)
    out["real"] = subset_accuracies(y_te, pred_real, yc_te)

    # Random-label control — note: y_te here is the SHUFFLED test labels (matches what
    # the probe was trained against). This is the Hewitt & Liang protocol.
    pred_ctrl = fit_logreg(X_layer[idx_tr], y_shuffled[idx_tr], X_layer[idx_te], seed)
    out["random_label"] = subset_accuracies(y_shuffled_te, pred_ctrl, yc_te)

    # Difference-in-means probe (zero training; uses real labels)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_layer[idx_tr])
    X_te_s = scaler.transform(X_layer[idx_te])
    pred_dim = dim_probe(X_tr_s, y_enc[idx_tr], X_te_s)
    out["dim"] = subset_accuracies(y_te, pred_dim, yc_te)

    # Vocab one-hot probe (no hidden state at all)
    pred_vocab = fit_logreg(vocab_feats[idx_tr], y_enc[idx_tr], vocab_feats[idx_te], seed)
    out["vocab"] = subset_accuracies(y_te, pred_vocab, yc_te)

    return {
        "metrics": out,
        "pred_real": pred_real,
        "y_te": y_te,
        "yc_te": yc_te,
        "idx_te": idx_te,
    }


def subset_accuracies(y_te, pred, yc_te):
    """Return acc on all-test / correct-output subset / wrong-output subset."""
    return {
        "acc_all": float(accuracy_score(y_te, pred)),
        "acc_correct": float(accuracy_score(y_te[yc_te], pred[yc_te])) if yc_te.any() else float("nan"),
        "acc_wrong": float(accuracy_score(y_te[~yc_te], pred[~yc_te])) if (~yc_te).any() else float("nan"),
        "n_test": int(len(y_te)),
        "n_test_correct": int(yc_te.sum()),
        "n_test_wrong": int((~yc_te).sum()),
    }


# ---------------- statistical tests ----------------

def bootstrap_ci(y_true, y_pred, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=0):
    """Bootstrap 95% CI on accuracy."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    accs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        accs[i] = (y_true[idx] == y_pred[idx]).mean()
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def mcnemar_pvalue(pred_a, pred_b, y_true):
    """McNemar's test comparing pred_a vs pred_b on paired test set.

    Tests whether the two classifiers have different error rates.
    Returns p-value (chi-square, continuity-corrected).
    """
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b_count = int(np.sum(a_correct & ~b_correct))  # a right, b wrong
    c_count = int(np.sum(~a_correct & b_correct))  # b right, a wrong
    if b_count + c_count == 0:
        return 1.0
    chi2_stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    return float(1 - chi2.cdf(chi2_stat, 1))


# ---------------- task processing ----------------

def process_task(data, task, num_layers=36):
    """Run all Gate 1 experiments for one task across all layers and seeds.

    Returns:
      rows: list of per-(layer, seed, baseline) result dicts.
      summary: per-task headline computed at the best layer.
    """
    mask = data["task"] == task
    X = data["X"][mask]
    y = data["gold"][mask]
    yc = data["y_correct"][mask]
    parsed = data["parsed"][mask]

    X, y, yc, parsed, n_dropped = filter_nan(X, y, yc, parsed)
    n = len(y)
    if n < 30:
        return [], None

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return [], None

    # Majority-class baseline (over the FULL task data, not just test).
    counts = np.bincount(y_enc)
    majority_baseline = float(counts.max() / counts.sum())

    vocab_feats = vocab_onehot_features(parsed)

    n_wrong = int((~yc).sum())
    n_correct = int(yc.sum())
    print(f"  {task}: n={n}  correct={n_correct}  wrong={n_wrong}  "
          f"options={n_classes}  majority={majority_baseline:.3f}  "
          f"(dropped {n_dropped} NaN)")

    rows = []
    for L in range(num_layers):
        per_seed = {"real": [], "random_label": [], "dim": [], "vocab": []}
        canonical_results = None

        for seed in SEEDS:
            r = run_one_layer(X[:, L, :], y_enc, yc, vocab_feats, seed)
            for k in per_seed:
                per_seed[k].append(r["metrics"][k])
            if seed == 0:
                canonical_results = r

        # Aggregate across seeds: mean and SD on each accuracy field.
        agg = {}
        for bname, seed_list in per_seed.items():
            agg[bname] = {}
            for field in ["acc_all", "acc_correct", "acc_wrong"]:
                vals = np.array([s[field] for s in seed_list], dtype=float)
                agg[bname][field + "_mean"] = float(np.nanmean(vals))
                agg[bname][field + "_std"] = float(np.nanstd(vals))
            for field in ["n_test", "n_test_correct", "n_test_wrong"]:
                agg[bname][field] = seed_list[0][field]

        # Bootstrap CI + McNemar on the canonical (seed=0) real probe.
        y_te = canonical_results["y_te"]
        yc_te = canonical_results["yc_te"]
        pred_real = canonical_results["pred_real"]

        majority_pred = np.full_like(y_te, counts.argmax())
        mcnemar_p = mcnemar_pvalue(pred_real, majority_pred, y_te)

        ci_all_lo, ci_all_hi = bootstrap_ci(y_te, pred_real)
        if (~yc_te).any():
            ci_wrong_lo, ci_wrong_hi = bootstrap_ci(y_te[~yc_te], pred_real[~yc_te])
        else:
            ci_wrong_lo = ci_wrong_hi = float("nan")

        # Selectivity is computed seed-by-seed then averaged.
        sel_seedwise = [s["acc_all"] - c["acc_all"]
                        for s, c in zip(per_seed["real"], per_seed["random_label"])]
        selectivity_all = float(np.mean(sel_seedwise))
        sel_wrong_seedwise = [
            (s["acc_wrong"] - c["acc_wrong"]) for s, c in zip(per_seed["real"], per_seed["random_label"])
            if not (np.isnan(s["acc_wrong"]) or np.isnan(c["acc_wrong"]))
        ]
        selectivity_wrong = float(np.mean(sel_wrong_seedwise)) if sel_wrong_seedwise else float("nan")

        rows.append({
            "task": task,
            "layer": L,
            "n_classes": n_classes,
            "majority_baseline": majority_baseline,
            "n_test_correct": agg["real"]["n_test_correct"],
            "n_test_wrong": agg["real"]["n_test_wrong"],
            "agg": agg,
            "selectivity_all": selectivity_all,
            "selectivity_wrong": selectivity_wrong,
            "ci_all": [ci_all_lo, ci_all_hi],
            "ci_wrong": [ci_wrong_lo, ci_wrong_hi],
            "mcnemar_p": mcnemar_p,
        })

    summary = summarize_task(rows, majority_baseline)
    return rows, summary


def summarize_task(rows, majority_baseline):
    """Compute per-task headlines and the passes_gate1 flag."""
    # Best layer by mean wrong-subset accuracy
    valid = [r for r in rows if not np.isnan(r["agg"]["real"]["acc_wrong_mean"])]
    if not valid:
        return None
    best = max(valid, key=lambda r: r["agg"]["real"]["acc_wrong_mean"])
    real = best["agg"]["real"]
    vocab = best["agg"]["vocab"]
    dim_a = best["agg"]["dim"]

    # Decision criteria (methodology_log.md §2)
    sel_pass = best["selectivity_wrong"] > SELECTIVITY_THRESHOLD
    vocab_gap = real["acc_wrong_mean"] - vocab["acc_wrong_mean"]
    vocab_pass = vocab_gap > 0
    dim_gap = real["acc_wrong_mean"] - dim_a["acc_wrong_mean"]
    dim_pass = dim_gap > DIM_GAP_THRESHOLD
    ci_lo, ci_hi = best["ci_wrong"]
    ci_excludes_majority = not (np.isnan(ci_lo) or np.isnan(ci_hi)) and (ci_lo > majority_baseline)
    passes_gate1 = bool(sel_pass and vocab_pass and dim_pass and ci_excludes_majority)

    return {
        "task": best["task"],
        "best_layer": best["layer"],
        "majority_baseline": majority_baseline,
        "n_test_wrong": best["n_test_wrong"],
        "real_acc_wrong_mean": real["acc_wrong_mean"],
        "real_acc_wrong_std": real["acc_wrong_std"],
        "real_acc_wrong_ci": best["ci_wrong"],
        "vocab_acc_wrong_mean": vocab["acc_wrong_mean"],
        "dim_acc_wrong_mean": dim_a["acc_wrong_mean"],
        "selectivity_wrong": best["selectivity_wrong"],
        "selectivity_all": best["selectivity_all"],
        "mcnemar_p": best["mcnemar_p"],
        "vocab_gap": vocab_gap,
        "dim_gap": dim_gap,
        "selectivity_pass": sel_pass,
        "vocab_pass": vocab_pass,
        "dim_pass": dim_pass,
        "ci_excludes_majority": ci_excludes_majority,
        "passes_gate1": passes_gate1,
    }


# ---------------- main ----------------

def print_summary_table(summaries):
    print("\n=== Per-task Gate 1 summary (best layer by wrong-subset acc) ===")
    print(f"{'task':<28s} {'L':>3} {'acc_w':>6} {'CI':>14} {'maj':>5} "
          f"{'sel_w':>6} {'vocab_w':>7} {'dim_w':>6} {'p':>7} {'pass':>5}")
    for s in summaries:
        ci_str = f"[{s['real_acc_wrong_ci'][0]:.2f},{s['real_acc_wrong_ci'][1]:.2f}]"
        flag = "✓" if s["passes_gate1"] else " "
        flags = "".join([
            "S" if s["selectivity_pass"] else ".",
            "V" if s["vocab_pass"] else ".",
            "D" if s["dim_pass"] else ".",
            "C" if s["ci_excludes_majority"] else ".",
        ])
        print(f"{s['task']:<28s} {s['best_layer']:>3d} "
              f"{s['real_acc_wrong_mean']:>6.3f} {ci_str:>14s} "
              f"{s['majority_baseline']:>5.3f} {s['selectivity_wrong']:>+6.3f} "
              f"{s['vocab_acc_wrong_mean']:>7.3f} {s['dim_acc_wrong_mean']:>6.3f} "
              f"{s['mcnemar_p']:>7.4f} {flag}{flags}")

    print()
    n_pass = sum(s["passes_gate1"] for s in summaries)
    print(f"Tasks passing Gate 1: {n_pass} / {len(summaries)}")
    print("Flag legend: S=selectivity V=vocab D=DiM C=CI-excludes-majority "
          f"(thresholds: sel>{SELECTIVITY_THRESHOLD}, dim_gap>{DIM_GAP_THRESHOLD})")


def main():
    print("Loading dataset...")
    data = load_dataset()
    print(f"  X.shape = {data['X'].shape}")
    tasks = sorted(set(data["task"]))
    print(f"  tasks   = {len(tasks)}")
    print()

    all_rows = []
    summaries = []
    for task in tasks:
        rows, summary = process_task(data, task)
        all_rows.extend(rows)
        if summary is not None:
            summaries.append(summary)

    # Save full results
    detail_path = config.OUTPUTS_DIR / "baselines_v1.json"
    with open(detail_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved {len(all_rows)} detail rows -> {detail_path}")

    summary_path = config.OUTPUTS_DIR / "baselines_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {len(summaries)} task summaries -> {summary_path}")

    print_summary_table(summaries)


if __name__ == "__main__":
    main()
