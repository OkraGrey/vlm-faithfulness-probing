# Probing VLM Faithfulness: Do Internal Representations Encode Correct Visual Grounding When Outputs Fail?

## Abstract

Vision-Language Models (VLMs) frequently produce wrong answers on visual tasks, and a recent line of work — typified by PAPO (Wang et al., ICLR 2026, arXiv:2507.06448) — attributes the majority of such errors to "perception failure" and proposes corrective changes to the training loss. PAPO's diagnosis is established entirely from output-level error categorization on 200 manually annotated examples drawn from math/STEM benchmarks; it never probes the model's internal representations to verify that the visual encoding is in fact deficient. This omission leaves a basic alternative untested: VLMs may already encode correct visual grounding signals in their hidden states even when the generated output is wrong — a translation failure, not a representation failure. Two recent papers, "Seeing But Not Believing" (Liu et al., arXiv:2510.17771) and "The Hidden Life of Tokens" (Lin et al., arXiv:2502.03628), provide preliminary attention-based and vocabulary-rank-based evidence that this phenomenon exists; neither connects it to PAPO's training-loss framing nor establishes a causal account.

This project provides the first probing-based test of PAPO's implicit assumption that perception failures correspond to deficient internal representations. We evaluate Qwen2.5-VL-7B on BLINK (Fu et al., ECCV 2024, arXiv:2404.12390), a benchmark whose 14 task types are explicitly constructed so that caption + LLM achieves random performance — guaranteeing that any model failure is perception-grounded rather than reasoning-grounded. We extract hidden states across the model's three computational stages (ViT encoder, MLP merger, LLM decoder) on the contrastive subset where the model produces a wrong answer, then train layer-wise linear probes (Alain & Bengio 2017, arXiv:1610.01644) to detect whether correct grounding is encoded internally. Selectivity controls (Hewitt & Liang 2019, arXiv:1909.03368), randomly-initialized-model baselines, layer-0 baselines, and final-token vocabulary controls are all included by design. Causal validation uses amnesic probing via Iterative Nullspace Projection (Elazar et al. 2021, TACL; Ravfogel et al. 2020) to determine whether erasure of the probed direction degrades downstream task performance — moving the claim from "the property is encoded" to "the model uses it." Out-of-distribution evaluation is performed on VisRes Bench, addressing the OOD-collapse concerns documented in "False Sense of Security" (arXiv:2509.03888) and Dubanowska et al. (arXiv:2509.19372).

The central empirical claim is contrastive: probe accuracy on the *wrong-output* subset, when significantly above all baselines and stable OOD, provides evidence that PAPO's training-loss interventions may be masking output-level symptoms while the underlying representation is already adequate. A null result is also informative: it would confirm PAPO's representational-deficit assumption and refute the alternative.

---

## 1. Problem Statement and Motivation

### 1.1 The Specific Gap

PAPO (Perception-Aware Policy Optimization, ICLR 2026 Poster) proposes an Implicit Perception Loss — a KL divergence between the model's output distributions under the original image and a patch-masked version — and a Double Entropy stabilizer. Headline improvement: 4.4%–17.5% across eight benchmarks; 30.5% reduction in "perception errors" as the authors categorize them. The 67% perception-failure figure is derived from manual annotation of 200 error cases on Geometry3K, MMK12, LogicVista, and MathVerse, with no inter-annotator agreement reported and no rubric published. HallusionBench is not used.

The unaddressed assumption is mechanistic: PAPO infers from output errors that the model's visual encoding is deficient, and applies a loss that increases output sensitivity to image content. The alternative — that the visual encoding is correct but the language decoding does not preserve it — would predict the same output errors but require a fundamentally different fix (decoding-time intervention, residual-stream steering) rather than a training-loss change.

Direct preliminary evidence that this alternative is plausible:
- "Seeing But Not Believing" (arXiv:2510.17771): Relative Attention Per Token analysis shows VLM deep-layer attention correctly localizes visual evidence at comparable rates for correct AND incorrect outputs.
- "The Hidden Life of Tokens" (arXiv:2502.03628): Visually-grounded tokens not expressed in output maintain high vocabulary rankings (~rank 5,000 of 32K) throughout generation; accumulated language priors in residual streams progressively dilute the visual signal.
- HALP (arXiv:2603.05465, EACL 2026): Internal-representation probes achieve up to 0.93 AUROC for VLM hallucination detection. For Qwen2.5-VL specifically, visual-only mid-layer features outperform late-fusion features.

None of these works runs a probing-based test against PAPO's training-loss premise. This project does so.

### 1.2 Hypothesis

**Primary hypothesis (H1)**: On BLINK examples where Qwen2.5-VL-7B produces an incorrect output, a linear probe trained on hidden states at one or more layers will recover the correct visual grounding with accuracy significantly above baselines, with selectivity > 10 percentage points (Hewitt & Liang 2019), and OOD performance not collapsing to chance.

**Secondary hypothesis (H2 — causal usage)**: Amnesic probing via Iterative Nullspace Projection (Ravfogel et al. 2020; Elazar et al. 2021) of the probed direction at the most informative layer will reduce downstream BLINK performance significantly more than random nullspace projection of the same rank.

**Null result of interest**: If H1 fails — probe accuracy is near chance or collapses OOD — this constitutes evidence supporting PAPO's representational-deficit assumption and against the translation-failure alternative.

---

## 2. Related Work

### 2.1 Foundations of Probing Methodology

Linear probes on frozen intermediate representations were established as an interpretability tool by Alain & Bengio (arXiv:1610.01644). The selectivity control task — comparing probe accuracy on real labels vs. randomly assigned labels held fixed across train/test — was introduced by Hewitt & Liang (arXiv:1909.03368, EMNLP 2019) and is the central guardrail against probe memorization. Belinkov's survey (Computational Linguistics 48(1), arXiv:2102.12452) catalogs the dataset-artifact, label-leakage, and complexity confounds. Pimentel et al. (arXiv:2004.03061, ACL 2020) frame probe accuracy as a mutual-information estimate, addressing the probe-complexity debate.

### 2.2 LLM Internal-State Probing for Truthfulness

SAPLMA (Azaria & Mitchell, arXiv:2304.13734, EMNLP Findings 2023) demonstrated that hidden-state classifiers detect truth/falsity of LLM-generated statements at 71–83% accuracy. INSIDE (Chen et al., arXiv:2402.03744, ICLR 2024) shows middle-layer last-token embeddings dominate for hallucination detection. Marks & Tegmark's *Geometry of Truth* (arXiv:2310.06824, COLM 2024) establishes via difference-in-means probes and causal interventions that LLM truthfulness representations are linear and causally implicated. Semantic Entropy Probes (Farquhar et al., arXiv:2406.15927) show that probes targeting uncertainty generalize OOD better than probes targeting direct correctness. "Knowing Before Saying" (arXiv:2505.24362) and "Reasoning Models Know When They're Right" (arXiv:2504.05419) both probe last-token-of-CoT-chunk hidden states for downstream success prediction — the design template closest to this project.

### 2.3 VLM Internal-State Probing

HALP (arXiv:2603.05465, EACL 2026) is the most directly relevant prior work: probes three families of internal VLM representations (visual-only features, vision-token decoder states, query-token decoder states) and finds for Qwen2.5-VL that visual-only mid-layer features carry the strongest hallucination signal (≈0.79 AUROC), differing from most architectures where late fusion dominates. "Towards Interpreting Visual Information Processing in VLMs" (arXiv:2410.07149) uses logit-lens analysis on LLaVA, finding that middle-to-late layers (around 25/33) show peak visual grounding signal and that removing object-specific patch tokens drops accuracy by 70–96%. VIB-Probe (arXiv:2601.05547, 2026) and FaithSCAN (arXiv:2601.00269, 2026) propose alternative probing-style architectures for VLM hallucination detection; neither addresses the PAPO premise.

### 2.4 PAPO and the Training-Loss Family of Fixes

PAPO (Wang et al., arXiv:2507.06448, ICLR 2026) introduces the Implicit Perception Loss for GRPO/DAPO training. Concurrent and follow-up work (CPPO, arXiv:2601.00501; VPPO-RL, ICLR 2026 Spotlight) proposes alternative perception-aware policy losses. All members of this family operate at the output/loss level. None probes internal representations to validate the diagnosis.

### 2.5 The OOD-Failure Risk for Probing

"False Sense of Security" (arXiv:2509.03888, NeurIPS 2025 Mech Interp Workshop) systematically demonstrates that LLM/VLM probes trained on safety/correctness tasks drop 15–99 percentage points OOD, with some configurations reaching near-zero accuracy on semantically equivalent paraphrased data. Dubanowska et al. (arXiv:2509.19372, EMNLP 2025 Findings) provide complementary evidence specifically for hallucination probes. Orgad et al. (referenced in the 2025 ACL probing literature) show probes do not transfer across task types requiring different skills. These results make OOD evaluation a non-optional component of any new probing study.

### 2.6 Why Not Detector-Based Verification

CoRGI (arXiv:2508.00378), an essentially equivalent design — Grounding DINO + step-wise CoT verification on HallusionBench — was withdrawn from arXiv on 2025-10-14 after reporting only +1.3 to +2.3 point improvements. Grounding DINO (Liu et al., arXiv:2303.05499, ECCV 2024) is structurally unsuited for charts, OCR, and synthetic/illusion content because (a) its training data contains no such content, (b) it produces high-confidence false positives for absent entities (GitHub issue #84), and (c) its bounding-box output formalism cannot represent quantitative, relational, or textual claims. The Devil-is-in-the-Fine-Grained-Details work (CVPR 2024) further documents Grounding DINO's weakness on attribute discrimination.

This study therefore avoids detector-based verification entirely. The choice of BLINK (Section 4) eliminates the need for it: BLINK provides ground-truth answers and is designed so that any failure is perception-grounded by construction, requiring no post-hoc attribution mechanism.

---

## 3. Proposed Method

### 3.1 Why BLINK as Primary Benchmark

BLINK (Fu et al., arXiv:2404.12390, ECCV 2024) comprises 3,807 multiple-choice questions across 14 classic computer-vision tasks (depth estimation, visual correspondence, multi-view reasoning, forensics detection, relative reflectance, art style, semantic correspondence, etc.). The benchmark was constructed so that "Caption + GPT-4" achieves near-random performance on every task — caption-mediated reasoning cannot solve BLINK, confirming all tasks genuinely require visual perception. GPT-4V scores ~51%; smaller open-source VLMs score substantially lower.

This structural guarantee eliminates the perception-vs-reasoning attribution problem that motivated the original Grounding DINO pipeline. When Qwen2.5-VL-7B answers a BLINK question incorrectly, the failure is necessarily perception-related; we do not need a separate annotator to attribute it. The contrastive subset for probing is therefore directly the set of incorrectly-answered questions.

### 3.2 Model: Qwen2.5-VL-7B-Instruct

Architecture (verified against the model card):
- **Vision Transformer**: 32 layers, hidden dimension 1280. Window attention in most layers; full attention at layers {7, 15, 23, 31}.
- **MLP merger**: 2-layer MLP that compresses 4 adjacent patches into one token; projects into the LLM's representation space (3584-dim).
- **LLM decoder (Qwen2.5-7B)**: 28 transformer layers, hidden dimension 3584.

This is a correction from the project's earlier draft, which incorrectly assumed "32 layers at 4096-dim." Hidden-state storage estimate: 28 × 3584 × 2 bytes (fp16) ≈ 200 KB per LLM token per sample. For a probe training set of ~5,000 samples extracting one token position, total ≈ 1 GB. The ViT adds 32 × 1280 × 2 = 80 KB per sample. Practical on commodity workstation memory.

The 7B model (vs. 3B) is chosen for three reasons: (a) it appears on the HallusionBench leaderboard (0.529 vs unmeasured 3B), making cross-benchmark comparison feasible; (b) CoT and multi-step reasoning quality is materially higher than at 3B scale, per the compositional-ability-gap analysis (arXiv:2505.19406); (c) HALP's reported probe accuracies are specifically for Qwen2.5-VL-7B, providing a direct reference point.

### 3.3 Hidden-State Extraction Locations

Probe at every computational stage:

| Stage | Locations | Aggregation |
|---|---|---|
| ViT | layers {7, 15, 23, 31} (full-attention layers) | mean over patch tokens AND object-region tokens AND CLS-equivalent |
| MLP merger output | one location | per-token-position dump |
| LLM decoder | layers {1, 4, 8, 14, 20, 24, 28} | last token of answer AND mean over visual tokens AND first generated token |

All extractions via PyTorch forward hooks with explicit `.detach().cpu()` in the hook body and `try/finally` removal — standard practice; failure to remove hooks is a known memory-leak footgun.

### 3.4 Probe Architectures

- **Primary**: Logistic regression with L2 regularization, regularization strength tuned on a held-out validation set.
- **Secondary**: One-hidden-layer MLP (256 units, ReLU), to detect non-linearly encoded structure. Difference in accuracy between linear and MLP is itself a reported finding.
- **Difference-in-means probe** (zero-trained): One vector per layer, computed as the mean of correct-grounding hidden states minus mean of incorrect-grounding states (Marks & Tegmark 2023). Serves as a complexity-floor baseline.

### 3.5 Labels

For each BLINK question, two labels:
- `output_correct`: standard task accuracy from BLINK's multiple-choice scoring.
- `grounding_signal`: a binary label derived from the BLINK ground-truth (correct option). Concretely: the probe is trained to predict the correct multiple-choice option from hidden states; "grounded" hidden states are those from which the correct option is recoverable.

The contrastive subset is the set of samples where `output_correct = False`. Probe accuracy on this subset is the primary outcome.

### 3.6 Required Baseline Suite

Every baseline below is non-optional given the standards documented in the probing methodology literature (Belinkov 2022; Hewitt & Liang 2019; Alain & Bengio 2017; "False Sense of Security" 2025):

| # | Baseline | Function | Citation |
|---|---|---|---|
| 1 | Majority-class predictor | Absolute floor | Standard |
| 2 | Layer-0 / input-embedding probe | Pre-transformer signal | Alain & Bengio 2017 |
| 3 | Randomly-initialized-model probe | Architectural-only signal | Alain & Bengio 2017 |
| 4 | Random-labels (Hewitt & Liang) selectivity | Probe-memorization check | H&L 2019 |
| 5 | Final-token vocabulary one-hot | Lexical-surface confound | Standard control |
| 6 | Logit-based uncertainty (max softmax) | Output-layer probe | SAPLMA 2023 |
| 7 | Difference-in-means zero-trained probe | Simplest-possible probe | Marks & Tegmark 2023 |
| 8 | BLINK-task-type one-hot | Position/category confound | This project |

### 3.7 Causal Validation — Level 2 via Amnesic Probing

The probe alone establishes correlation. To make the stronger claim that the model uses the encoded grounding information, this project uses Iterative Nullspace Projection (Ravfogel et al. 2020) as in Elazar et al. (TACL 2021, arXiv:2006.00995):

1. Train a linear probe to recover the grounding label from hidden states at the layer of peak probe accuracy.
2. Compute the iterative nullspace projection P that maximally removes the probe's classification ability.
3. Apply P to the hidden states at that layer during model inference (intervene on the residual stream forward pass).
4. Measure downstream BLINK accuracy on a held-out set vs. an untouched model AND vs. random nullspace projection of the same rank.

If P significantly degrades BLINK accuracy while random projection of the same rank does not, the model uses the probed direction. This is the standard Level-2 causal evidence in current probing work and is sufficient for top-venue publication of a usage claim. Activation-patching (Meng et al. 2022, ROME) would be Level 3 but is out of scope for this semester.

### 3.8 Out-of-Distribution Evaluation

Probe trained on BLINK; tested on VisRes Bench (arXiv:2512.21194), a three-level perception-to-reasoning hierarchy benchmark. VisRes Bench is structurally different from BLINK (procedurally generated vs. naturalistic images; multi-attribute compositional reasoning vs. classic CV tasks), making this a genuine distribution shift. A probe that holds at non-trivial accuracy on VisRes Bench Level 1 (pure perception) has crossed the bar set by "False Sense of Security."

Secondary OOD: held-out BLINK task types (e.g., train on 12 task types, test on the other 2). Documents whether the probe is task-type-specific or genuinely model-general.

### 3.9 Statistical Validation

- 5 random seeds for probe training; mean ± SD reported.
- 95% bootstrap confidence intervals (10,000 resamples) on all accuracy/AUROC metrics.
- McNemar's test for paired comparisons against baselines; threshold p < 0.01.
- Benjamini-Hochberg FDR correction for layer-wise multiple comparisons.
- Calibration (ECE) reported alongside discrimination (AUROC).

---

## 4. Experimental Setup

### 4.1 Data

**Primary**: BLINK validation + test splits, all 14 task types, 3,807 questions, with multiple-choice ground-truth answers. Split discipline: probe train/val/test split is enforced at the image level — no image's questions appear in more than one split.

**OOD**: VisRes Bench Level 1, 2, 3 splits.

**Pilot validation**: Random sample of 200 BLINK examples manually inspected to verify model error patterns and the correctness of the contrastive labeling.

### 4.2 Compute Requirements

- Inference on BLINK with Qwen2.5-VL-7B: ~14 GB GPU memory in fp16, single A100/H100 or equivalent.
- Probe training: CPU-feasible, but GPU-accelerated logistic regression (e.g., via sklearn + cuML or via PyTorch) is faster.
- Total wall time estimate (inference + extraction): ~12 hours on one A100. Probe training: ~30 minutes per layer per seed.

### 4.3 Code & Reproducibility

- All probing code released on GitHub at submission.
- Random seeds documented; bootstrap resamples saved.
- Hidden states optionally released for downstream community use.

---

## 5. Risks and Mitigations

| Risk | Source | Mitigation |
|---|---|---|
| Probe collapse OOD | "False Sense of Security" arXiv:2509.03888 documents 15–99 pp drops | Design VisRes Bench OOD eval from day one; consider uncertainty-targeted probes (Farquhar 2024) as a secondary target if correctness probes fail OOD |
| Probe memorization rather than encoding | Hewitt & Liang 2019 | Selectivity control mandatory; report at every layer |
| Final-token lexical confound | Token-identity may explain probe success | Final-token one-hot baseline (#5 in baseline suite) |
| Self-consistency in label generation | If hidden states themselves drive labels | Labels are BLINK ground-truth (external), not model-generated |
| Small contrastive subset | If model is too accurate or too inaccurate, the wrong-output subset is small or unbalanced | Qwen2.5-VL-7B accuracy on BLINK is moderate (~45-55%), keeping both subsets sizable |
| Amnesic probing collateral damage | INLP may remove other useful information | Compare to random-rank projection (same nullspace rank); report difference |
| Negative result interpretation | If H1 fails | Frame the paper so a null result is publishable (it supports PAPO's premise and is a contribution in itself) |

---

## 6. Semester Timeline (14 weeks)

| Week | Milestone | Deliverable to advisor |
|---|---|---|
| 1–2 | Locked hypothesis; literature review (the 15 papers above) | Hypothesis document approved |
| 3 | Inference pipeline on BLINK with Qwen2.5-VL-7B; hidden-state extraction hooks verified | Extracted hidden states for a 200-sample pilot |
| 4 | Pilot validation against manual inspection; sanity check the contrastive labeling | Pilot report |
| 5–6 | Full baseline suite (1–8) implemented and run; numbers frozen BEFORE main experiments | Baseline table |
| 7–8 | Main probing experiments across all ViT/merger/LLM positions; selectivity at every layer | Layer-wise results table + figures |
| 9 | Amnesic probing (INLP) at the layer of peak accuracy | Causal-validation results |
| 10 | OOD evaluation on VisRes Bench (Level 1 first, then 2 and 3) | OOD results |
| 11 | Ablations: linear vs. MLP probe; aggregation strategies; held-out task types | Ablation table |
| 12 | First draft following Hewitt & Liang (2019) and Elazar et al. (2021) as structural templates | Complete first draft |
| 13 | Revisions per advisor feedback; tighten claims to match evidence | Final draft |
| 14 | Code release prep; submission to workshop or EMNLP Findings | Submitted |

---

## 7. Expected Outcomes and Their Meaning

| Outcome | Interpretation |
|---|---|
| H1 confirmed, H2 confirmed, holds OOD | Strong evidence that PAPO's training-loss family addresses output expression rather than visual representation. Direct contribution to the VLM interp literature. |
| H1 confirmed, H2 not, holds OOD | The grounding information is *encoded* but not *used*. Still a publishable finding about VLM representational structure; suggests decoding-time interventions over training-loss changes. |
| H1 confirmed, OOD collapse | Confirms the "False Sense of Security" pattern in a new domain. Cautionary result for the VLM probing literature. |
| H1 not confirmed | Supports PAPO's representational-deficit assumption. Null but informative — and the absence of evidence is itself a contribution given the recent attention-based and rank-based claims. |

---

## 8. Differences from Earlier Draft

For transparency, the substantive corrections from the prior version of this document:

1. **Benchmark changed from HallusionBench to BLINK** (primary) + VisRes Bench (OOD). BLINK's structural guarantee that all failures are perception-grounded eliminates the need for the Grounding DINO + atomic CoT decomposition pipeline that prior work (CoRGI, withdrawn from arXiv 2025-10-14) confirmed to be unworkable.
2. **Model changed from Qwen2.5-VL-3B to Qwen2.5-VL-7B**. The 3B variant is not on the relevant leaderboards and was not trained with structured CoT; the 7B variant matches HALP's reported probing setup.
3. **Architecture description corrected**: 28 LLM layers at 3584-dim (7B) plus 32 ViT layers at 1280-dim and an MLP merger — not "32 layers at 4096-dim" as the earlier draft stated.
4. **Citation corrections**: PAPO is real but did not evaluate on HallusionBench; the 67% figure comes from math/STEM benchmarks. "Causal probing (2026)" replaced with Marks & Tegmark *Geometry of Truth* (arXiv:2310.06824). VADE replaced with Dubanowska et al. (arXiv:2509.19372) for the OOD-warning citation.
5. **Causal evidence upgraded from absent to Level 2** via amnesic probing (INLP). The earlier draft treated probe accuracy alone as evidence of internal encoding; the literature is explicit (Elazar et al. 2021; Ravichander et al. 2021) that probe accuracy without causal validation is insufficient for usage claims.
6. **Baseline suite expanded from implicit to explicit**. All 8 baselines listed in Section 3.6 are mandatory per Hewitt & Liang 2019, Belinkov 2022, and recent probing work; their absence is grounds for rejection at top venues.
