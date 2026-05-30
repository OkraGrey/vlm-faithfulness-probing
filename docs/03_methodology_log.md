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

---

## Gate 0 — Honest re-confirmation under nested CV (2026-05-30)

**Why this gate exists** (added during the 2026-05-30 literature refresh, see `06`): Gate 1 selected the best of 36 layers *by wrong-subset accuracy on a single 25% test split* (4–25 wrong-test samples), then reported that layer's CI uncorrected — a winner's-curse / selection-bias estimate (`05` Q2/Q3 flagged it). Gate 0 re-estimates honestly: **nested 5-fold CV** (layer chosen on inner folds of the outer-train only; pooled across outer folds every sample is predicted once, so the wrong-subset CI uses the FULL 20–111 wrong samples), with **difference-in-means as the primary probe** (D2 decision in `08`). Code: `pilot/gate0_cv.py` (self-test passes: recovers a planted signal layer, asserts no train/test leakage). Output: `pilot/outputs/gate0_cv_summary.json`.

### Result — the Gate 1 headline does NOT survive

| | Gate 1 (single split, logreg, best-by-wrong) | Gate 0 (nested CV, DiM primary) |
|---|---|---|
| Forensic_Detection | "PASS" L27, wrong-acc 0.517, CI [0.35,0.78] | **FAIL** — wrong-acc 0.182, CI [0.13,0.30] **below** majority 0.265, selectivity −0.046 |
| Object_Localization | "PASS" L21, wrong-acc 0.781, CI [0.60,1.00] | **FAIL** — wrong-acc 0.291, CI [0.16,0.39] **below** majority 0.566, selectivity −0.200 |

The inner-CV layer-selection histograms confirm the diagnosis: Object_Localization's "peak" scatters across L33/31/32/34/2/27/12/0/28/26/10 — there was no stable peak; L21 was a single-split artifact. **The "2/14 robust tasks" claim is retracted.**

### The decomposition that replaces it — the probe reads OUTPUT, not gold, on most tasks

Decisive diagnostic (`gate0_diag`, transient): on the wrong-output subset, what does the DiM prediction match — the gold answer (H1: "knows but says wrong") or the model's own parsed output (PAPO: internal state matches the wrong answer)?

```
task                        n_w  ->gold  ->output ->neither  reading
Jigsaw                       80    1.00      0.00      0.00   GOLD (H1)
Multi-view_Reasoning         70    0.91      0.09      0.00   GOLD (H1)
Art_Style                    57    0.58      0.42      0.00   GOLD (H1, weak)
Functional_Correspondence   101    0.27      0.16      0.57   mixed/noise
IQ_Test                     111    0.40      0.05      0.55   mixed/noise
Forensic_Detection           91    0.21      0.59      0.20   OUTPUT (PAPO)
Object_Localization          57    0.26      0.74      0.00   OUTPUT (PAPO)
Relative_Reflectance         77    0.25      0.53      0.22   OUTPUT (PAPO)
Semantic_Correspondence      90    0.07      0.78      0.16   OUTPUT (PAPO)
Visual_Correspondence       104    0.10      0.79      0.12   OUTPUT (PAPO)
Counting / Relative_Depth / Spatial_Relation / Visual_Similarity
                                   0.00      1.00      0.00   OUTPUT (PAPO)
```

**Interpretation:** the high "→output" fractions confirm the probe machinery is sound (the pre-generation hidden state strongly encodes the answer the model is about to emit). The scientific signal is in *which* tasks ALSO recover gold on wrong outputs:
- **GOLD-reading (candidate H1): Jigsaw, Multi-view_Reasoning, Art_Style(weak).** Hidden state encodes the correct answer despite a wrong output → "sees but can't say."
- **OUTPUT-reading (PAPO-consistent): 9 tasks.** Internal state matches the wrong output → no hidden correct grounding to recover; supports PAPO's premise for these tasks.

This is consistent with Gate 1's raw DiM numbers (Object_Localization DiM 0.014; Jigsaw/Multi-view DiM 1.00) — Gate 1 only "failed" Jigsaw/Multi-view because logreg couldn't beat DiM (the `dim_gap` criterion `05` Q1 flagged as too strict). Making DiM primary (D2) vindicates that flag.

### Decision

1. **Retract the Forensic/Object_Localization headline.** They read output, not gold.
2. **Carry Jigsaw, Multi-view_Reasoning, Art_Style into G3 (counterfactual image-swap construct validity).** These are the only GOLD-reading candidates.
3. **Treat Jigsaw/Multi-view's ~1.0 with strong skepticism** — both binary; a 1.00 DiM is exactly what a non-visual question/format artifact would also produce. G3 is now load-bearing, not confirmatory: it decides whether the GOLD signal is visual or artifactual. If G3 shows the signal does not track the swapped image, the honest conclusion is "the apparent H1 signal on these tasks is a construct artifact" — a cautionary methods finding (still publishable).
4. **The dominant story so far is PAPO-consistent** (9/14 tasks). Be honest that the project, run rigorously, currently *supports* PAPO more than it refutes it — pending G3 on the 3 candidates.

---

## Gate 3 — Construct validity (noise-image ablation), 2026-05-30

**Question**: is the GOLD-reading signal on Jigsaw / Multi-view / Art_Style genuinely *visual*, or a question/format artifact (the worry raised by their suspiciously clean binary ~1.0)?

**Method**: re-ran inference on the SAME samples with each image replaced by uniform RGB noise of the same size/count — text prompt and vision-token structure held identical, only image *content* changed (`pilot/run_inference_noise.py`, 400 samples, 0 errors, 833 s on Mac MPS). Then compared DiM nested-CV gold-recovery on the wrong-output subset: real-image vs noise-image (`pilot/gate3_construct.py`, output `gate3_construct_summary.json`). If the signal is visual, noise collapses gold-recovery to chance; if it's a text/format artifact, noise retains it. (This is the doc-04 vision-ablation idea, re-targeted to the correct tasks per Gate 0, and used as the load-bearing construct test because these tasks embed options in the image so a clean question-fixed image-swap isn't available — see `08` D1.)

**Result** (wrong-output subset, binary tasks, 5-seed mean):

```
task                  majority  real gold-acc(wrong)  noise gold-acc(wrong)  visual_attr  verdict
Jigsaw                  0.527    0.990 CI[1.00,1.00]   0.588 CI[0.41,0.64]      +0.402     VISUAL (H1 holds)
Multi-view_Reasoning    0.556    0.934 CI[0.84,0.97]   0.511 CI[0.40,0.63]      +0.423     VISUAL (H1 holds)
Art_Style               0.530    0.519 CI[0.46,0.70]   0.396 CI[0.32,0.56]      +0.123     inconclusive
```

**Interpretation**:
- **Jigsaw and Multi-view_Reasoning: H1 CONFIRMED, construct-valid.** Real-image gold-recovery is ~0.93–0.99; replacing the image with noise collapses it to chance (≈ majority). The binary-artifact worry is *ruled out* — a format/text artifact would have survived noise. So the residual stream linearly encodes the *correct* answer, derived from the *image*, on examples where the model outputs the wrong answer. For these two tasks, Qwen2.5-VL-3B "sees but cannot say."
- **Art_Style: dropped.** Real gold-recovery (0.519) is not clearly above majority (CI includes 0.530); the signal was weak to begin with. Not an H1 task.
- Sanity check: under noise the model collapsed to a degenerate prior (e.g., always outputting "B" on Art_Style) — expected when no visual information is present.

**Scope/limits to carry into the writeup**: (1) only 2/14 tasks — the effect is *task-specific*, not a general property; (2) both binary (the dissociation is easiest to detect there); (3) this is decodability + construct validity — **correlational**; it does NOT yet show the model *uses* the encoded direction (G4); (4) 3B only; generality (7B, other architectures) untested.

**Decision**: H1 holds for Jigsaw and Multi-view_Reasoning. Carry exactly these two into **G4 (activation patching)** — the causal-usage test. Pre-register the knowledge–action-gap risk (2603.18353): the likely outcome is that the direction is encoded but only partially used.

### Gate 3 ADDENDUM — layer profile forces a weaker claim (2026-05-30, same day)

Before running G4 I computed the honest per-layer CV DiM gold-recovery profile (`layer_profile`, transient) for the two tasks. **This tempers the Gate 3 conclusion and must be reported honestly:**

```
Jigsaw       : gold-acc = 1.00 at EVERY layer L0..L34 (0.99 at L35)
Multi-view   : gold-acc = 1.00 at L0-L1, ~0.91-0.99 across L2-L35
```

**Why this matters.** A genuine "the model visually *reasons* to the correct answer and then suppresses it" signal should **emerge with depth** (low in early layers, rising mid/late) — this is the pattern §1 of this log pre-registered as the expectation, and layer-0 dominance as the warning sign. A signal that is already perfect at **layer 0** (essentially the input/embedding representation) is the layer-0 artifact signature.

**Reconciliation with Gate 3.** The noise-image ablation still holds — the signal *is* image-dependent (noise collapses it to chance), so it is not text/format leakage. But combined with the flat-from-L0 profile, the correct interpretation is: **the gold answer is linearly decodable from the image-conditioned representation from the earliest layer — a shallow/early visual feature, NOT evidence of deep internal reasoning that the decoder discards.**

**Revised claim (honest):**
- *Literal H1* ("the correct answer is linearly encoded from the image on wrong-output examples") — **holds** for these 2 binary tasks.
- *Interpretive H1* ("the model internally computes the right answer through visual reasoning, then fails to express it") — **NOT supported**; the layer-0 dominance is more consistent with a shallow image cue (or a BLINK binary-item regularity) than with suppressed reasoning.
- The dissociation (gold decodable at 100% from L0, yet the model is wrong ~50% of the time) is *suspiciously strong* and should be treated as a candidate artifact until G4 + an image-permutation control adjudicate.

**Consequence for G4.** Intervene at a principled MID-layer (band L8–L24, CV-best within it; `gate4_patching.py` `INTERVENTION_BAND`), not at L0 — so any steering effect has downstream depth to propagate, and we avoid steering on the artifact-prone earliest layers. G4 is now doing double duty: (a) does amplifying the gold direction flip the output (causal usage)? and (b) implicitly, is the mid-layer gold direction real enough to be causal at all?

**Honest headline as of now:** run rigorously, the project so far **supports PAPO for 9/14 tasks**, and for the 2 binary tasks where gold is decodable, the signal looks **shallow/early (possible artifact)** rather than deep grounding. H1's strong form is not yet supported. (An image-permutation control — re-extract with each question paired to a mismatched image — is the clean next artifact test if G4 is ambiguous.)

## Gate 4 — Causal usage via DiM activation steering (2026-05-30)

**Question**: does the model *use* the decodable gold direction, or is it encoded-but-inert? Code `pilot/gate4_patching.py` (validity check: baseline reproduces the wrong answer 100% of the time for both tasks — harness sound). Intervention at the CV-best mid-band layer (Jigsaw L8, Multi-view L13); steer the last prompt token's residual toward the gold class by α·(class-mean-difference), vs a random direction of identical magnitude; α ∈ {1,2,4}. Output `gate4_patching_summary.json`.

**Result — flip-to-gold rate on the wrong subset:**

```
                       baseline  steer a1  steer a2  steer a4 | rand a1  rand a2  rand a4   n_wrong
Jigsaw  (L8)             0.000     0.013     0.000     0.025  |  0.000    0.000    0.000      80
Multi-view (L13)         0.000     0.014     0.014     0.129  |  0.014    0.014    0.014      70
```

**Interpretation — the direction is largely NOT causally used:**
- **Jigsaw: causally inert.** Gold is decodable at 1.00, yet steering along that exact direction — even at 4× the class separation — flips only 2/80 outputs (2.5%), barely above the random control (0%). The model's output is almost completely insensitive to the direction it supposedly "encodes."
- **Multi-view: weak, high-magnitude-only effect.** At α=4, steering flips 9/70 (12.9%) vs 1/70 (1.4%) for random — a real but small effect that appears only under a large, likely off-manifold push; at realistic magnitudes (α=1,2) it is indistinguishable from random. That random doesn't flip to gold confirms the model is generally perturbation-robust and the gold direction has at most weak causal pull.

**This is the "encoded but not used" / knowledge–action gap outcome** pre-registered as likely (arXiv:2603.18353). Combined with the flat-from-L0 profile (Gate 3 addendum), the three gates together give a coherent, honest conclusion for Qwen2.5-VL-**3B** on BLINK:

> The apparent internal "knowledge" of the correct answer on the wrong-output subset is (a) restricted to a small set of (binary) tasks, (b) shallow/early rather than emergent-with-depth, and (c) **causally inert or near-inert** — the model does not "see but fail to say." H1's strong form is **rejected at 3B**. The result is broadly **consistent with PAPO's premise** that the fix must target the model's use of perception, not merely its internal representation.

**Limitations of G4 (state honestly):** single mid-layer per task; DiM-steering at the last prompt token only (a stronger test — multi-site patching or correct-donor interchange — could reveal more, and the Multi-view α=4 effect shows the method *can* flip some outputs, so the near-null is a genuine small effect, not a broken intervention).

**Decision → scale to 7B.** The 3B study is now complete and gives a clean (H1-negative) result. The natural and well-motivated next experiment is **Qwen2.5-VL-7B** (runs locally on the user's 48GB M5 Pro). This is NOT to rescue the hypothesis — it is the necessary scale test, because the most plausible reason a 3B shows no usable grounding is that a 3B may genuinely lack it. The decisive 7B diagnostics: (1) does the gold-signal layer profile **emerge with depth** (genuine grounding) instead of being flat-from-L0? (2) is the steered causal effect **larger** at 7B? A 7B that also shows shallow + inert signal would make the H1-negative robust across scale; a 7B that shows emergent-with-depth + causally-usable grounding would revive H1 and become the headline. Either is publishable. (Optional, cheap, on 3B: the image-permutation control to characterize *what* the inert decodable signal is — now lower priority since G4 shows it's inert regardless of source.)
