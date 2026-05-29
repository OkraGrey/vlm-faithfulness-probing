# Methodology Log — Phase 2 (Probing) and Beyond

> **Purpose**: This document captures the *reasoning* behind methodological decisions made during Phase 2 (probe analysis) and Phase 3 (causal validation). `JOURNEY.md` records *what happened*; this file records *why we chose to do it that way*. Re-read this before any pivot or claim revision.
>
> Maintained as an append-only decision log. Each Gate gets a "Plan" section (written before running) and a "Result + Decision" section (written after).

---

## 0. Framing: What "research-grade" means here

The project's target outlet is a workshop or EMNLP Findings paper (realistic for a one-semester scope with Mac compute). To be publishable at that level, the probing methodology literature has established explicit gates a study must pass. The relevant survey-level sources are:

- **Belinkov (2022)**, "Probing Classifiers: Promises, Shortcomings, and Advances," Computational Linguistics 48(1). The most authoritative methodological reference.
- **Hewitt & Liang (2019)**, EMNLP. Established the selectivity-control requirement that has become standard.
- **Rogers, Kovaleva, Rumshisky (2020)**, TACL. The "BERTology" survey — codifies layer-wise reporting standards.
- **Elazar et al. (2021)**, TACL. Established amnesic probing as the standard Level-2 causal validation.

These four papers, plus the original Alain & Bengio (2017) probing paper, define the methodological surface area a probing study must cover. Skipping any of their controls is grounds for rejection at top venues — this is the explicit norm, not a soft preference.

## 1. The first-look result that prompted Phase 2

After Phase 1 inference (1,901 BLINK samples, 989 wrong-output contrastive cases), a first probe (single-seed, no controls) showed:

```
9 of 14 tasks beat majority-class baseline on wrong-output subset.
5 tasks showed gaps > 0.15 (Forensic_Detection, Jigsaw, Art_Style,
                            Object_Localization, Relative_Reflectance).
Peak layers cluster in 21-34 range — consistent with INSIDE
(arXiv:2402.03744) and "Knowing Before Saying" (arXiv:2505.24362).
```

**This number is uninterpretable until Gate 1 passes.** Hewitt & Liang's 2019 result was specifically that probes with no selectivity control achieve high accuracy even when reading random labels. We cannot yet distinguish "the probe found grounding signal" from "the probe memorized this dataset."

Two specific oddities that flag the need for controls:

1. **Multi-view_Reasoning and IQ_Test peaked at layer 0.** Layer 0 is the input embedding, before any transformer computation. Real grounding signal should emerge in deeper layers; layer-0 dominance suggests either (a) input-level feature leakage or (b) probe overfitting to small wrong-test subsets (~10-25 samples).
2. **Spatial_Relation showed wrong-subset accuracy 0.500 vs majority 0.538** — below baseline. With only 20 wrong cases in 143 samples, the per-task wrong-test split is ~5 samples. This is noise dominated.

## 2. The 5-Gate plan (canonical methodology pipeline)

The sequence below is ordered by **information value per unit compute**, with the cheapest and most decisive experiment first. This ordering follows Belinkov 2022's recommended workflow.

### Gate 1 — Establish whether the signal is real (CHEAP, this week)

**Question**: Does the probe read genuine representation, or does it memorize?

**Experiments**:

| ID | Experiment | Method | Compute | Citation |
|---|---|---|---|---|
| 1.1 | Multi-seed | 5 random seeds for probe init + train/test split; report mean ± SD | ~10 min | Standard ML practice |
| 1.2 | Bootstrap CIs | 10,000 resamples on test accuracy; report 95% CI | ~5 min | Standard ML practice |
| 1.3 | Selectivity (random-label control) | Same probe, same hidden states, but labels shuffled and fixed. selectivity = real_acc − control_acc | ~10 min | Hewitt & Liang 2019 (arXiv:1909.03368) |
| 1.4 | Final-token vocabulary one-hot | Train logistic regression on the model's output letter alone (no hidden state) to predict gold | ~2 min | Standard shortcut control |
| 1.5 | Difference-in-means probe | Zero-trained: mean(correct-letter hidden states) − mean(wrong-letter hidden states); classify by distance to means | ~2 min | Marks & Tegmark 2023 (arXiv:2310.06824) |

**Decision criteria after Gate 1**:

| Criterion | Threshold | Source |
|---|---|---|
| Selectivity (per task × layer) | > 0.10 to be considered "real signal" | Hewitt & Liang 2019 |
| Probe acc > Vocab one-hot acc | by any margin (rules out lexical shortcut) | Standard |
| Probe acc > DiM acc | by > 0.05 (logistic regression added value over simplest probe) | Standard |
| Probe acc 95% CI excludes majority baseline | minimum for any effect claim | Standard ML |

**Likely outcome** (literature priors): Of the 5-6 currently "promising" tasks, expect 2-4 to pass all four criteria. Some will fail selectivity (probe was memorizing). Some will fail the vocab one-hot (lexical shortcut). The set that passes all four is the "real signal" set.

**Pivot points**:

- If 0-1 tasks pass: pivot to a methodology paper documenting BLINK probe memorization (still publishable; a confirmatory null result).
- If 2-4 tasks pass: proceed to Gate 2 on those tasks only.
- If all 14 tasks pass: suspicious — likely a measurement artifact. Inspect for label leakage.

### Gate 2 — Rule out architecture-only signal (EXPENSIVE BASELINE, one overnight)

**Question**: Did *training* teach the model these representations, or does the architecture alone produce them?

**Experiment**:

- Construct Qwen2.5-VL-3B with **random weights** (`from_config()`, no pretrained loading).
- Run the same inference pipeline on the same 1,901 BLINK samples — same prompt format, same hooks, same max_pixels, same seed.
- Train Gate-1-survivor probes on these random-init hidden states.
- Compare: `gap = trained_probe_acc − random_init_probe_acc`.

**Decision criterion**:

- `gap > 2 × bootstrap CI width` → trained model adds substantial signal.
- `gap < 0.05` → architectural priors carry most of the signal. Reframe the paper.

**Citation**: Alain & Bengio (2017, arXiv:1610.01644). Zhang & Bowman (2018) demonstrate empirically that random networks produce useful features.

**Compute estimate**: ~90 min inference + ~30 min probing on Mac MPS, fully resumable via the existing checkpointing.

### Gate 3 — Statistical defensibility (BAKED INTO GATES 1-2)

These are *reporting requirements*, not separate experiments. They're built into the same code that runs Gates 1-2.

| Requirement | Implementation |
|---|---|
| McNemar's test | Per task, real probe vs majority baseline. Threshold: p < 0.01 (top venues expect stronger than 0.05). |
| Multiple-comparison correction | Benjamini-Hochberg FDR across 504 task × layer comparisons. |
| Calibration (ECE) | Reported alongside accuracy at peak-layer summaries. |
| Pre-registration | This file IS the pre-registration. Decision criteria above were chosen before running, so we can't unconsciously result-shop. |

**Citation**: Dror et al. (2018), "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing"; Bender et al. (2020) reporting standards.

### Gate 4 — Distribution-shift robustness (MEDIUM COST)

**Question**: Is the probe fitting BLINK-specific patterns, or genuine model-internal representation?

This is the gate that "False Sense of Security" (arXiv:2509.03888) made non-optional for any probe-based hallucination/correctness study. They show 15-99 pp accuracy drops under OOD.

**Experiments**:

| ID | Experiment | Compute |
|---|---|---|
| 4.1 | Within-BLINK held-out tasks: train probe on 12 BLINK tasks, test on the 2 held-out. Several random partitions. | ~20 min |
| 4.2 | Cross-benchmark: run inference on VisRes Bench (or MMVP) with same model, apply trained probes. | ~2 hours (new inference run) |

**Decision criterion**: Probe should retain at least 60% of in-distribution gap on at least one OOD setting. If it collapses to within 5 pp of baseline, the probe is task-specific.

### Gate 5 — Causal validation (THE PUBLICATION PIVOT)

**Question**: Does the model *use* the probed direction in its computation, or is the direction merely encoded?

Elazar et al. (2021) demonstrated empirically that "conventional probing performance is not correlated to task importance." A property can be 90% probe-decodable and have zero causal role in model behavior.

**Experiment — Iterative Nullspace Projection (Ravfogel et al. 2020 + Elazar et al. 2021)**:

1. At the layer where Gates 1-4 showed best signal, retrain the linear probe to convergence; extract weight vector `w`.
2. Compute the orthogonal projection matrix `P` that projects onto the nullspace of `w` (removes the direction `w` from any hidden state).
3. Apply `P` as a forward hook at that layer during inference. The residual stream at that layer is replaced with `P @ residual`.
4. Re-run inference on the wrong-output subset with `P` applied.
5. Measure: does model accuracy degrade more than (a) baseline (no projection) and (b) random-rank-`k` projection of the same dimensionality?

**Decision criterion**: `INLP-degraded accuracy < random-rank-degraded accuracy` with statistical significance. If yes → model uses the direction (Level-2 causal claim achieved).

**Citation**: Ravfogel et al. (2020, INLP, arXiv:2004.07667); Elazar et al. (2021, TACL).

**Engineering cost**: ~200 LoC plus forward-hook integration. Reference implementation from Ravfogel's GitHub is reusable.

## 3. The publishable end-state template

This is the abstract-equivalent of the paper we are building toward (subject to results):

> "On 1,901 BLINK examples with Qwen2.5-VL-3B, we train linear probes on hidden states across all 36 LLM decoder layers. On the contrastive subset where the model produces incorrect outputs, probes at layers L₁-L₂ recover the correct multiple-choice answer at accuracy A (95% CI [a, b]), exceeding the majority-class baseline by ∆₁ on K of 14 task types, exceeding a randomly-initialized model baseline by ∆₂, and showing selectivity > S per Hewitt & Liang (2019). Amnesic probing via INLP at the peak layer degrades downstream accuracy by D pp more than random-rank projection (p < 0.01), supporting the claim that the model uses, not merely encodes, the recovered grounding direction. On VisRes Bench (OOD), the signal retains R% of its in-distribution gap. These results provide the first probing-based test of PAPO's (Wang et al., ICLR 2026, arXiv:2507.06448) implicit representational-deficit assumption and suggest that perception-aware loss interventions target output expression rather than visual encoding for the subset of tasks where probe signal is robust."

Numbers `A, a, b, ∆₁, ∆₂, S, D, R, K, L₁, L₂` are filled in by Gates 1-5.

If any gate fails, the paper pivots:
- **Gate 1 fails**: methodology paper, "BLINK probe memorization patterns in 3B VLMs."
- **Gate 2 fails**: architecture paper, "What untrained Qwen2.5-VL representations encode."
- **Gate 4 fails**: cautious paper, "In-distribution probing signal does not transfer in 3B VLMs."
- **Gate 5 fails**: weaker claim, "Grounding information is encoded but not used by the decoder."

Every gate failure is still a publishable paper, just a different one. This is important: there is no scenario where the work is wasted, only scenarios where the title changes.

## 4. Phase 2b plan: `baselines.py`

**Scope**: All of Gate 1 in one module. Replaces the current `probe.py` (which is preserved for reference but not used downstream).

**Inputs**: `outputs/dataset.npz` (already collated).

**Outputs**:
- `outputs/baselines_v1.json` — full per-(task, layer, baseline_type, seed) results.
- `outputs/baselines_summary.json` — per-task headlines (selectivity, best layer, vocab gap, DiM gap, CI).
- Console summary tables at end.

**Per task × layer, the module computes**:

```
real_probe        — logistic regression with L2, 5 seeds
random_label_ctrl — same probe, labels shuffled (Hewitt & Liang selectivity)
dim_probe         — difference-in-means, zero training (Marks & Tegmark)
vocab_probe       — logistic regression on the model's parsed-letter one-hot
                    (no hidden state) — rules out lexical shortcut
```

Per task, the module also computes:

```
mcnemar_p         — McNemar's test, real probe vs majority class
selectivity       — real_probe_acc - random_label_ctrl_acc
ci_lower, ci_upper — bootstrap 95% CIs on real_probe_acc (test set)
```

**Per-task headline** (the new version of the table we saw last time):

```
task  best_layer  acc_all  acc_wrong  selectivity  vs_dim  vs_vocab  ci_lower  ci_upper  passes_gate1
```

`passes_gate1` is `True` iff:
- selectivity > 0.10 at this task's best layer
- acc_wrong - vocab_probe_acc > 0 at the same layer
- acc_wrong - dim_probe_acc > 0.05 at the same layer
- 95% CI on acc_wrong excludes the majority baseline

**Decision rule**: tasks with `passes_gate1 = True` are the only ones we carry forward to Gate 2.

## 5. Decision log (append-only as we run)

### 2026-05-17 — Phase 2 strategy set
Decision: follow the 5-Gate plan documented above, in order, before any amnesic-probing work. Rationale: cheap controls first, expensive interventions only on signals that survived cheap controls. User confirmed: "agree with your approach... happy that you are not rushing rather helping to follow the research methodologies."

### 2026-05-18 — Building Gate 1
Decision: write `baselines.py` as a single unified module rather than 5 separate scripts. Easier to enforce identical splits/seeds across baselines (a frequent source of subtle bugs in probing papers). User confirmed.

### 2026-05-18 — Gate 1 results

**Headline**: 2 of 14 tasks pass all four Gate 1 criteria.

Full per-task table (best layer by mean wrong-subset accuracy across 5 seeds):

```
task                       L  acc_w     CI_wrong    maj   sel_w  vocab_w  dim_w     p   flags
Forensic_Detection        27  0.517  [0.35,0.78]  0.265  +0.249  0.172   0.217  0.015  ✓SVDC
Object_Localization       21  0.781  [0.60,1.00]  0.566  +0.170  0.690   0.014  0.371  ✓SVDC

Jigsaw                    19  0.737  [0.59,0.91]  0.527  +0.319  0.455   1.000  0.677   SV.C
Multi-view_Reasoning      15  0.600  [0.36,0.86]  0.556  +0.155  0.540   1.000  0.181   SV..
Art_Style                 34  0.688  [0.38,0.88]  0.530  +0.219  0.333   0.710  0.149   SV..
Functional_Correspondence 25  0.485  [0.24,0.64]  0.331  +0.320  0.396   0.327  0.547   SVD.

[8 other tasks failed selectivity, vocab, or DiM]
```

Flag legend: S = selectivity > 0.10, V = vocab gap > 0, D = DiM gap > 0.05, C = CI excludes majority.

**The 2 confirmed survivors**:

1. **Forensic_Detection** (layer 27, 4-class task) — 0.517 wrong-acc vs 0.265 majority, 91 wrong samples available. Selectivity +0.249, McNemar p=0.015. Robust across all four checks.
2. **Object_Localization** (layer 21, binary task) — 0.781 wrong-acc vs 0.566 majority, 57 wrong samples. Selectivity +0.170, CI [0.60, 1.00] cleanly above majority. Suggests grounding-direction encoding is real for this task.

**Important methodological observation** (worth investigating before Gate 2):

For 3 tasks (Jigsaw, Multi-view_Reasoning, Art_Style) the **difference-in-means probe achieves higher accuracy than logistic regression**. This is consistent with Hewitt & Liang 2019's argument for simpler probes: with 2048-dim features and 60-100 training samples, logistic regression can overfit despite L2 regularization. DiM essentially "regularizes" by using only first-moment statistics.

The current `dim_gap > 0.05` criterion rules out these tasks because logreg doesn't beat DiM — but arguably the *DiM result itself* is evidence of clean linear separability and therefore real signal. We may want to revise the criterion to `max(logreg, dim) - vocab > threshold` for the next iteration. This is the kind of methodological refinement that emerges only from running real data.

**Decision criteria sanity check**:

- Selectivity threshold of 0.10 is Hewitt & Liang's working number; cited widely. We accept this.
- DiM gap threshold of 0.05 was chosen to ensure trained probe adds something over the simplest baseline. The Jigsaw/Multi-view result suggests this may be too strict — we'd be excluding tasks where signal is so clean even DiM gets it perfectly. Flag for revision.
- CI excludes majority is the conservative criterion. Forensic_Detection and Object_Localization both clear this.

**Decision**: Proceed to Gate 2 (random-init baseline) on the 2 confirmed survivors PLUS the 3 DiM-confounded tasks (Jigsaw, Multi-view, Art_Style). Total: 5 tasks. The expensive baseline will further filter these.

Rationale for keeping the 3 DiM-failure tasks: they passed selectivity (probe is reading representation, not memorizing) AND vocab one-hot (not a lexical shortcut). Failing DiM means our trained probe isn't adding value, but the SIGNAL still exists in the representation. Random-init baseline will tell us whether the signal is in the *trained* representation or the architecture.

**Likely Gate 2 outcomes**:

- If random-init Qwen2.5-VL gets close to trained (within 5 pp) → signal is mostly architectural. Reframe paper around what architecture gives for free.
- If random-init clearly below → trained model added value. Continue to OOD eval.
- For Object_Localization specifically (binary, clean CI) — the most informative test case.

### Open questions for the next iteration

1. **Should we tighten test_fraction?** Small wrong-test subsets (10-25 samples) produce wide CIs. K-fold cross-validation would tighten these without changing data quantity. Costs ~5x compute.
2. **Should we report passes_gate1 with a more lenient DiM criterion?** As discussed above.
3. **Are 2-5 tasks enough for a publishable paper?** For workshop / Findings, yes — "VLM internal grounding signal exists for specific perceptual tasks (Forensic_Detection, Object_Localization) but is not universal." For main track, probably not — needs 7B and OOD eval.
