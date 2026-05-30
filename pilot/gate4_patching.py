"""Gate 4: causal-usage test via difference-in-means activation steering.

G3 showed Jigsaw & Multi-view_Reasoning encode the GOLD answer in the residual
stream (visually) even when the model outputs the wrong letter. That is
correlational. G4 asks the causal question: does the model USE that direction?

Method (Marks & Tegmark 2023 style steering + control):
  - At the layer L where the DiM gold signal peaks, compute the class-mean
    difference d = mean(states|gold=class1) - mean(states|gold=class0).
  - For each WRONG-output sample, during the prefill forward pass we add, at the
    LAST prompt-token position (the position that predicts the answer letter),
    a vector that pushes the residual stream TOWARD the sample's gold class:
        steer = alpha * (d if gold==class1 else -d)
  - Generate greedily and check whether the output flips to gold.
  - CONTROL: a RANDOM direction of identical magnitude (alpha * ||d||).
  - Dose-response over alpha in {1, 2, 4}.

Causal claim: model USES the direction iff steer flip-to-gold rate >> random
flip-to-gold rate. If even strong steering barely moves the output, the direction
is encoded-but-not-used (knowledge-action gap, arXiv:2603.18353).

VALIDITY CHECK: with alpha=0 (no steering) the model must reproduce its original
wrong answer (greedy decoding is deterministic). If baseline reproduction is low,
the harness is broken and results are void.

Run:   ./env/bin/python gate4_patching.py            (full, ~30 min on Mac)
       ./env/bin/python gate4_patching.py --selftest (baseline reproduction on 6 samples)
       ./env/bin/python gate4_patching.py --limit 10 (quick partial run)
"""
import json
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch

import config
from data_loader import load_blink_samples
from model_setup import load_model_and_processor, _get_decoder_layers, _format_question
from answer_parser import parse_answer, is_correct

TASKS = ["Jigsaw", "Multi-view_Reasoning"]
ALPHAS = [1.0, 2.0, 4.0]
RANDOM_SEED = 2024


# ---------------- load per-task hidden states + labels from Phase-1 npz ----------------

def load_task_samples(task):
    """Return dict idx -> (hidden(36,2048) f32, gold letter, correct bool)."""
    d = config.INFERENCE_DIR / task
    out = {}
    for fp in d.glob("*.npz"):
        z = np.load(fp, allow_pickle=True)
        out[int(z["idx"])] = (z["hidden_states"].astype(np.float32), str(z["gold"]), bool(z["correct"]))
    return out


# The honest CV layer profile (layer_profile diagnostic, 2026-05-30) showed gold is
# decodable at ~1.0 from the EARLIEST layers (incl. L0) for these binary tasks — a
# shallow/early signal, NOT deep grounding (see docs/03 Gate 3 addendum). We therefore
# intervene at a principled MID-layer band, where a steering effect (if any) has many
# downstream layers to propagate through, and which is past the layer-0 artifact zone.
INTERVENTION_BAND = (8, 24)


def best_layer_and_direction(samples):
    """Pick the intervention layer by HONEST 5-fold CV DiM gold-recovery on the wrong
    subset, restricted to the mid-layer band; return (L, d, classes, cv_acc).

    d = mean(states|class1) - mean(states|class0) at L, computed on full data (a
    steering DIRECTION, not an accuracy estimate, so full-data use is appropriate).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    idxs = sorted(samples)
    X = np.stack([samples[i][0] for i in idxs])
    gold = np.array([samples[i][1] for i in idxs])
    correct = np.array([samples[i][2] for i in idxs], dtype=bool)
    classes = sorted(set(gold))
    enc = {c: j for j, c in enumerate(classes)}
    y = np.array([enc[g] for g in gold])
    wrong = ~correct

    lo, hi = INTERVENTION_BAND
    best_L, best_acc = lo, -1.0
    for L in range(lo, min(hi + 1, X.shape[1])):
        pred = np.full(len(y), -1)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X[:, L, :], y):
            sc = StandardScaler()
            xa = sc.fit_transform(X[tr, L, :]); xb = sc.transform(X[te, L, :])
            mu = np.stack([xa[y[tr] == c].mean(0) for c in range(len(classes))])
            pred[te] = np.linalg.norm(xb[:, None, :] - mu[None, :, :], axis=2).argmin(1)
        acc = (pred[wrong] == y[wrong]).mean() if wrong.any() else 0.0
        if acc > best_acc:
            best_acc, best_L = acc, L
    XL = X[:, best_L, :]
    d = (XL[y == 1].mean(0) - XL[y == 0].mean(0)).astype(np.float32)
    return best_L, d, classes, float(best_acc)


# ---------------- steered inference ----------------

@contextmanager
def steer_hook(model, layer_idx, vec):
    """Add `vec` to the last prompt-token hidden state at layer_idx during prefill."""
    layers = _get_decoder_layers(model)
    handle = None

    def fn(module, inputs, output):
        if vec is None:
            return None
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] > 1:  # prefill only (decode steps have seq_len 1)
            hs = hs.clone()
            hs[:, -1, :] = hs[:, -1, :] + vec.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return None

    try:
        handle = layers[layer_idx].register_forward_hook(fn)
        yield
    finally:
        if handle is not None:
            handle.remove()


def run_steered(model, processor, images, question, options, layer_idx, vec):
    prompt_text = _format_question(question, options, len(images))
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=list(images), padding=True, return_tensors="pt").to(model.device)
    with steer_hook(model, layer_idx, vec), torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


# ---------------- main experiment ----------------

def run_task(model, processor, task, limit=None, selftest=False):
    samples = load_task_samples(task)
    L, d, classes, dim_acc = best_layer_and_direction(samples)
    enc = {c: j for j, c in enumerate(classes)}
    dnorm = float(np.linalg.norm(d))
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"\n[{task}] intervention layer L={L} (DiM wrong-acc={dim_acc:.3f}), ||d||={dnorm:.2f}, classes={classes}")

    # reload images/question/options by idx
    blink = list(load_blink_samples(task=task, n=10_000, split=config.BLINK_SPLIT))

    wrong_idxs = [i for i in sorted(samples) if not samples[i][2]]
    if limit:
        wrong_idxs = wrong_idxs[:limit]

    d_t = torch.tensor(d, device=model.device)
    cond_flip = {"baseline": 0, **{f"steer_a{a}": 0 for a in ALPHAS}, **{f"rand_a{a}": 0 for a in ALPHAS}}
    baseline_reproduces = 0
    n = 0

    for idx in wrong_idxs:
        _, gold, _ = samples[idx]
        images, question, options, answer = blink[idx]
        gold_enc = enc.get(gold)
        if gold_enc is None:
            continue
        sign = 1.0 if gold_enc == 1 else -1.0

        # baseline (no steer) — must reproduce the wrong answer
        base_txt = run_steered(model, processor, images, question, options, L, None)
        base_parsed = parse_answer(base_txt, options)
        n += 1
        if base_parsed is not None and base_parsed != gold:
            baseline_reproduces += 1            # reproduced a (still-wrong) answer
        if is_correct(base_parsed, gold):
            cond_flip["baseline"] += 1

        if selftest:
            print(f"  idx={idx} gold={gold} baseline_out={base_parsed!r}")
            continue

        # steer toward gold + random control, per alpha
        rand_dir = rng.normal(size=d.shape).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir) * dnorm   # match ||d||
        rand_t = torch.tensor(rand_dir, device=model.device)
        for a in ALPHAS:
            steer_txt = run_steered(model, processor, images, question, options, L, sign * a * d_t)
            if is_correct(parse_answer(steer_txt, options), gold):
                cond_flip[f"steer_a{a}"] += 1
            rand_txt = run_steered(model, processor, images, question, options, L, a * rand_t)
            if is_correct(parse_answer(rand_txt, options), gold):
                cond_flip[f"rand_a{a}"] += 1

    res = {"task": task, "layer": L, "dim_wrong_acc": dim_acc, "n_wrong": n,
           "baseline_reproduces_wrong_rate": baseline_reproduces / max(n, 1),
           "flip_rates": {k: v / max(n, 1) for k, v in cond_flip.items()}}
    return res


def main():
    limit = None
    selftest = "--selftest" in sys.argv
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if selftest:
        limit = 6

    model, processor = load_model_and_processor()
    t0 = time.time()
    results = []
    for task in TASKS:
        results.append(run_task(model, processor, task, limit=limit, selftest=selftest))
        if not selftest:
            r = results[-1]
            print(f"  baseline reproduces wrong: {r['baseline_reproduces_wrong_rate']:.2f}  "
                  f"(validity check — should be high)")
            print(f"  flip-to-gold rates: {json.dumps(r['flip_rates'], indent=0)}")

    if not selftest:
        out = config.OUTPUTS_DIR / "gate4_patching_summary.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {out}")
    print(f"Elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
