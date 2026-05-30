# Literature Refresh + Methodology Lock — 2026-05-30

> **Purpose**: Triggered by a paper the user dropped (arXiv:2604.02486) plus a fresh research-scientist audit. This document (a) re-positions the project's novelty against the 2025–2026 literature that has appeared since the original audit (`01_research_validation.md`, 2026-05-17), (b) locks the methodology for the remaining experiments, revising two pre-registered choices, and (c) defines the single decisive next experiment. Every citation here was fetched and verified by research agents during this session; unverified items are explicitly flagged.
>
> Read this together with `01_research_validation.md` (still valid on foundations and the PAPO/CoRGI/Grounding-DINO diagnosis) and `03_methodology_log.md` (Gate 1 results stand; Gate 5 is revised below).

---

## 0. TL;DR

1. **The paper the user linked (arXiv:2604.02486, "VLMs Need Words") is not PAPO.** It argues a *similar* hypothesis (info present internally, output wrong) but does **not** scoop our method.
2. **The phenomenon is no longer novel; the method and framing still are.** The "encoded-but-not-expressed" claim was originated by **Fu et al. 2025 (arXiv:2506.08008)** and related 2025–2026 work. Our novelty must be the *rigorous causal probing methodology on the wrong-output subset, on BLINK with Qwen2.5-VL, framed as a direct PAPO rebuttal* — not the discovery of the phenomenon.
3. **The crux is construct validity, and our planned next step (blank-image ablation) is necessary but not sufficient.** Upgrade the primary construct-validity test to a **counterfactual image-swap probe**. An independent 2026 paper (arXiv:2603.06054) validates this exact design.
4. **Revise the causal plan: demote INLP.** Use **activation patching** as the primary causal test; if doing concept erasure, use **LEACE**, not INLP (the INLP critique, arXiv:2207.04153, makes an INLP accuracy-drop ambiguous).
5. **The most likely *and* most valuable outcome is the middle case** — "encoded but not used" (the knowledge–action gap). A 2026 paper (arXiv:2603.18353) found probes at 98% AUROC but interventions fixing only ~45% of errors. Design the study to land this distinction cleanly, not to chase a binary YES.
6. **Statistics need fixing before any claim**: the current best-layer-by-wrong-subset selection is a winner's-curse on n=4–25 test samples. Adopt nested k-fold CV.

---

## 1. The catalyzing paper, precisely characterized

**arXiv:2604.02486 — "VLMs Need Words: Vision Language Models Ignore Visual Detail In Favor of Semantic Anchors"** (Shahgir, Chen, Fu, Shayegani, Abu-Ghazaleh, Kementchedjhieva, Dong; UC Riverside + MBZUAI; submitted 2 Apr 2026). Verified against the arXiv HTML.

- **Thesis**: the "hidden-in-plain-sight" gap is explained by *nameability*. VLMs solve fine-grained visual tasks by mapping entities to **semantic labels ("semantic anchors")** and reasoning verbally; nameable → success via language shortcut, unnameable → brittle hallucinated descriptions. Framed as a *learned shortcut*, not an architectural limit.
- **Internal-rep method**: **Logit Lens** on LLM-decoder visual tokens (Mean Jaccard Distance between decoded token sets) + a **training-free MaxSim matcher** (ColBERT operator). **No trained probes. No selectivity/random-init controls. No causal internal intervention.**
- **Data / models**: SPair-71k, synthetic 2D shapes, face correspondence. **Qwen3VL / Gemma3 / InternVL3.5.** **Not BLINK. Not Qwen2.5-VL.**
- **Evidence shape**: *aggregate, correlational* (dataset-averaged representation-vs-output gap, contrasted Named vs No-Name). **No wrong-output subset analysis.** Causal evidence is *behavioral* (teaching arbitrary names raises 29% → 86%; task-specific finetuning generalizes), **not** an internal intervention.
- **Status**: preprint, no venue, no code/data release found.

**What this means for us:** it is the single closest competitor on the *idea*, and a strong related-work anchor — but it leaves our entire methodological contribution open (trained probes, controls, wrong-output contrastive subset, BLINK, Qwen2.5-VL, causal *internal* manipulation, PAPO framing).

---

## 2. Revised novelty positioning

The original audit (2026-05-17) treated the phenomenon as "real but unfilled." That is no longer accurate. The 2025–2026 literature has filled the *phenomenon*; it has not filled our *method*.

### 2.1 The phenomenon is now established (do not claim to discover it)

The true priors — read and differentiate from these, more than from 2604.02486:

| Paper | arXiv | What it established |
|---|---|---|
| Fu, Bonnen, Guillory, Darrell — "Hidden in plain sight: VLMs overlook their visual representations" | **2506.08008** (Jun 2025) | Closest published statement of our thesis. VLMs underperform a *direct readout of their own visual encoder*; the bottleneck is the LM's *use* of vision, not the encoder. **But: reads out the visual encoder, not LLM hidden states; vision-centric tasks, not BLINK; no wrong-output contrastive probe.** |
| Takishita et al. — "LLMs Can Compensate for Deficiencies in Visual Representations" | 2506.05439 (EMNLP'25 Findings) | Decoder can access/compensate for visual semantics — opposite of PAPO's "deficient representation" premise. |
| Yao et al. — "Reading Between the Lines" (OCR abstention via latent probes) | 2511.19806 | Hidden-state probes detect when output is wrong; best signal in *intermediate* layers. OCR-only. |
| Ashok et al. — "Can VLMs Recall Factual Associations From Visual References?" | 2508.18297 (EMNLP'25 Findings) | Internal-state probes flag unreliable responses >92%. Factual recall, not perception. |

### 2.2 The defensible, currently-unclaimed contribution

Per the frontier survey, **no published work** does the following combination, and each clause is individually unclaimed:

> *We present the first contrastive hidden-state probing study of the **wrong-output subset**, on **BLINK** with **Qwen2.5-VL**, with **construct-validity controls** (counterfactual image-swap) and a **causal internal intervention** (activation patching), framed as a direct empirical rebuttal of **PAPO's** premise that perception errors reflect deficient internal visual representations.*

Note: this clause-set **supersedes** the "amnesic/INLP causal validation" phrasing the original spec used — see §4.

### 2.3 Two papers we must explicitly differentiate from (closest method neighbors)

- **arXiv:2603.06054 — "Probing Visual Concepts in Lightweight VLMs for Automated Driving"** (Mar 2026). Uses **linear probes on counterfactual image sets** to separate **"perceptual failure" (not linearly encoded)** from **"cognitive failure" (encoded but not aligned to language)**. This is *almost our exact taxonomy and our recommended construct-validity method.* Differentiator: driving domain, lightweight VLMs (Qwen2.5-VL not confirmed), no PAPO framing, no causal removal. **We adopt their counterfactual-probe design and differentiate on domain + model + causal stage + PAPO framing.**
- **arXiv:2604.12119 — "Beyond Perception Errors: Semantic Fixation in Large VLMs"** (Apr 2026). Conceptually our strongest ally: argues many "perception" errors are rule-mapping failures, not encoding failures, and **late-layer activation steering partially recovers** performance. Custom benchmark, no probing study, no PAPO framing.

---

## 3. The crux: construct validity (revising the "vision ablation" proposal)

The probe predicts the **gold answer letter** from hidden states. Above-chance accuracy on the wrong-output subset could mean (a) visual grounding [our hypothesis], (b) question/task structure or option-co-occurrence priors with no vision, or (c) label artifacts. **No gate in the current plan separates these** — and this is exactly the failure mode the probing-methodology literature warns is fatal (Ravichander et al. "Probing the Probing Paradigm", arXiv:2005.00719; Belinkov, arXiv:2102.12452).

The pending `04_next_step_vision_ablation.md` proposal (blank-image / text-only) is **necessary but not sufficient**: a blank image leaves the question + options intact, so a probe can still read question-conditioned priors that correlate with the gold letter. We would then either under- or over-claim.

### Decision: counterfactual image-swap is the primary construct-validity test

For each surviving wrong sample with gold `g`, replace the image with one whose correct answer is a *different* option `g'`, holding the question template and option set fixed (BLINK's within-task structure makes these pairs cheap to mine from the existing 1,901 runs). **The probe's prediction must track the image (`g`→`g'`) when only the image changes.** Invariant to the swap ⇒ the signal is linguistic/label structure (hypothesis dies, cleanly). Flips with the image ⇒ the signal is genuinely visual.

This is interchange-intervention logic (Geiger et al., arXiv:2106.02997 / 2112.00826) and is independently validated as a published design by arXiv:2603.06054. **Keep blank-image as a cheap secondary null.** Keep selectivity (Hewitt & Liang, arXiv:1909.03368) on the swapped set.

**Probe choice for this stage:** use the **difference-in-means** direction (Marks & Tegmark, arXiv:2310.06824) as the *primary* probe, not high-capacity logistic regression. DiM directions are more causally implicated and carry directly into the causal stage. (Pimentel et al., arXiv:2004.03061, argue for highest-capacity probes — correct for measuring *how much info* is present, wrong for our *causal/behavioral* question; cite to acknowledge the deliberate trade-off.)

---

## 4. Revised causal plan: activation patching over INLP

`reqs.md` §3.7 and `03_methodology_log.md` Gate 5 commit to **amnesic probing via INLP** (Ravfogel et al. 2020; Elazar et al. 2021). **Revise this.**

- **Why demote INLP:** Kumar, Tan & Sharma (arXiv:2207.04153, NeurIPS 2022) prove post-hoc removal methods cannot fully remove a concept and can destroy collateral task features — so an accuracy drop after INLP is *ambiguous* (removed the answer direction, or collateral info?). Elazar et al. themselves caution against strong causal reads. This ambiguity is fatal for a load-bearing causal claim.
- **Primary causal test → activation patching** (Meng et al., arXiv:2202.05262; interchange-intervention logic, Geiger). Reuse §3's swapped pairs: patch a *correct-run* hidden state (or just the DiM component) at layer ℓ into the *wrong-run* forward pass and measure whether the output flips toward gold, **vs. a random-direction patch control** (the patching analogue of selectivity). Cheap, no retraining, runs on Mac/modest GPU for a 3B/7B forward pass.
- **If concept erasure is still wanted → LEACE** (Belrose et al., arXiv:2306.03819), the closed-form, provably minimal-edit successor to INLP. (arXiv:2506.11673 shows LEACE/mean-projection pass information-control tests INLP fails — *verify this ID before citing*.)
- **Steering** (add/subtract the DiM direction) is useful *corroboration*, not the load-bearing test (global perturbation, more confound-prone).

**Causal claim:** the model *uses* the direction iff patching the correct-run state into the wrong run flips the output to gold well above the random-direction control.

---

## 5. The outcome framing that maximizes value (and is most likely)

The binary "YES, the model sees but can't say" is *not* the most probable or most valuable result. The **knowledge–action gap** is:

- **arXiv:2603.18353 — "Interpretability without actionability"** (Mar 2026): on Qwen2.5-7B, linear probes hit **98.2% AUROC** while output sensitivity to interventions was only **45.1%** — a 53-point gap; four mechanistic methods barely fixed errors and damaged correct cases. (Text/clinical domain — *we would be the first to test whether this gap holds for visual grounding on BLINK/Qwen2.5-VL.*)

This makes the **"encoded but not used"** middle outcome both likely and the sharpest contribution: it *partially vindicates us* (representation not deficient — against PAPO) **and** *partially vindicates PAPO* (model still fails to use it), reframing the debate as **access/usage failure, not representation failure**. Design and write the study to land this trichotomy cleanly:

| Outcome | Reading | PAPO implication |
|---|---|---|
| Construct-valid + causally used | "Sees but can't say" — translation failure | PAPO's loss targets the wrong stage |
| Construct-valid + **not** causally used (likely) | "Encoded but dormant" — knowledge–action gap | Reframe: usage/access failure, not representation failure |
| Not construct-valid | Probe read linguistic/label artifact | Cautionary methods finding; original signal was a construct artifact |

All three are publishable. Pre-register all three so we cannot result-shop.

---

## 6. Locked minimal pipeline (supersedes the 5-gate order for what comes next)

Run only on the Gate-1 survivors. Order = cheapest hypothesis-killer first.

| Step | Test | Method (verified cite) | Decision threshold | Failure reading |
|---|---|---|---|---|
| **G0 (NEW, do first)** | Re-confirm survivors under honest layer selection | nested 5-fold CV; bootstrap CIs; drop best-layer-by-wrong-subset cherry-pick | survivor CI still excludes majority under CV | the "2/14" was a selection artifact |
| **G2** | Random-init baseline | Alain & Bengio (1610.01644) | trained probe ≫ random-init (non-overlapping CIs) | signal is probe-fitting, not learned reps → NO |
| **G3 (CRUX)** | Counterfactual image-swap (+ blank-image null) | Geiger (2106.02997/2112.00826); Ravichander (2005.00719); design echoes 2603.06054 | probe tracks image not question; swap-flip ≫ blank baseline | linguistic/label artifact → **clean NO** |
| **G4** | Activation patching (correct→wrong) + random-direction control | Meng (2202.05262); Geiger | patch flips to gold ≫ random-direction patch | encoded but not used → **knowledge–action gap** (publishable) |
| **G5** | OOD / cross-benchmark transfer | "False Sense of Security" (2509.03888) | probe + causal effect hold on a 2nd perception benchmark | in-distribution artifact → NO for generality |

Gate 1 stays as recorded in `03_methodology_log.md`. G0 is a re-run of it with corrected statistics, not a new experiment.

---

## 7. Statistics & scale (mandatory fixes)

- **Winner's curse**: `baselines.py` selects the best of 36 layers by wrong-subset accuracy on **n=4–25 test samples**, then reports that layer's CI uncorrected. This biases peaks upward. **Fix:** nested k-fold CV with layer selection inside the inner fold; report per-fold CIs.
- **Fragile survivors**: Object_Localization "passes" on **15 test samples** (binary, majority 0.566, CI [0.60,1.00]); Forensic_Detection on 23. Real but thin. G0 must confirm them honestly before anything is built on them.
- **Causal power**: activation-patching flip-rates at n≈57–91 will have wide CIs; a null would be *underpowered, not negative*. Raise n by (a) temperature/paraphrase sampling on the same BLINK items for the two surviving tasks, and/or (b) moving to **Qwen2.5-VL-7B** — which doubles as a generality check. **Do not** add tasks (2/14 is a real result, not a power problem). **Do not** fabricate a power number; rely on k-fold CV + bootstrap CIs + random-direction control.

---

## 8. Decision pending the user's return: 3B vs 7B

`reqs.md` v2 commits to **7B** (28 layers, 3584-dim); all current data is **3B** (36 layers, 2048-dim). The headline currently rests on a model the spec says is not the study model.

**Recommendation (reversible):** run **G0 + G3 (construct validity) on the existing 3B data first** — it is free and either kills or confirms the hypothesis cheaply. Commit to **7B for G4 (causal) + G5 (OOD) + final claims**, because (a) the spec already says 7B, (b) HALP's Qwen2.5-VL probe accuracies are 7B, (c) 3B survivors are statistically thin, (d) 7B doubles as the generality check. This needs ~1 modest GPU (forward passes only; no VLM training) — confirm cluster/cloud budget with the advisor (`05_open_questions.md` Q7).

---

## 9. Verified citations added this session

All fetched and confirmed (arXiv ID + title + authors) unless flagged.

**Closest priors / phenomenon (differentiate from these):**
- 2506.08008 — Fu et al., "Hidden in plain sight: VLMs overlook their visual representations" (Jun 2025). **The key prior.**
- 2506.05439 — Takishita et al., "LLMs Can Compensate for Deficiencies in Visual Representations" (EMNLP'25 Findings).
- 2511.19806 — Yao et al., "Reading Between the Lines" (latent probes for OCR abstention).
- 2508.18297 — Ashok et al., "Can VLMs Recall Factual Associations From Visual References?" (EMNLP'25 Findings).
- 2604.02486 — Shahgir et al., "VLMs Need Words" (the dropped paper).

**Method neighbors (adopt / differentiate):**
- 2603.06054 — "Probing Visual Concepts in Lightweight VLMs for Automated Driving" (counterfactual-probe perceptual-vs-cognitive taxonomy).
- 2604.12119 — "Beyond Perception Errors: Semantic Fixation in Large VLMs" (late-layer steering recovers).

**Causal interventions (precedent):**
- 2509.22674 — "Pathological Truth Bias in VLMs" (activation patching flips ~23% of wrong outputs).
- 2602.07025 — "The Geometry of Representational Failures in VLMs" (concept-vector steering, Qwen family).
- 2510.26769 — "SteerVLM" (lightweight activation steering).

**Threats to validity (must pre-register against):**
- 2603.18353 — "Interpretability without actionability" — the knowledge–action gap (98% AUROC vs 45% actionability).
- 2602.06652 — "Same Answer, Different Representations" (representation instability under benign perturbation).
- 2604.14888 — "Reasoning Dynamics and the Limits of Monitoring Modality Reliance" (CoT looks grounded but follows text).
- 2601.22150 — "Do VLMs Perceive or Recall? (VI-Probe)" (wrong outputs are not one mechanism).

**Methodology (causal-plan revision):**
- 2207.04153 — Kumar, Tan & Sharma, "Probing Classifiers are Unreliable for Concept Removal" (NeurIPS 2022) — the INLP critique.
- 2306.03819 — Belrose et al., "LEACE" (NeurIPS 2023) — INLP replacement.
- 2106.02997 / 2112.00826 — Geiger et al., causal abstraction / interchange interventions.
- 2202.05262 — Meng et al., ROME / activation patching.
- 2005.00719 — Ravichander, Belinkov, Hovy, "Probing the Probing Paradigm" (EACL 2021).

### Verification caveats (resolve before formal citation)
- **2506.11673** (LEACE/mean-projection > INLP on info-control) — surfaced, not directly fetched. Verify ID.
- **PAPO (2507.06448)**: confirmed it identifies perception as the major error source, but the fetched page did **not** state the strong phrasing "errors = deficient internal visual representation." Quote PAPO precisely; rebut its *premise that perception is the bottleneck*, do not attribute the stronger wording verbatim until confirmed in the PDF.
- **VisRes Bench (2512.21194)** and **Dubanowska et al. (2509.19372)** — carried forward from prior docs, not re-fetched this session. Verify before relying on them in G5.
- **2601.00269 (FaithSCAN), 2601.05547 (VIB-Probe)** — not re-fetched this session.
- Single-author fetches (2509.22674, 2604.12119) and the newest 2604.x preprints — confirm full author lists and peer-review status before formal citation.

---

## 10. Immediate next experiment (concrete)

1. **G0** — rewrite the layer-selection in `baselines.py` to nested 5-fold CV; re-confirm Forensic_Detection + Object_Localization survive honestly. (~1–2 h compute, existing data.)
2. **G3** — implement the counterfactual image-swap experiment on the two survivors using the existing inference infra (mine same-question/different-answer pairs from the 1,901 runs; re-extract states on swapped pairs; DiM probe must track the image). Keep blank-image as the secondary null. (~half day.)
3. Append results + decisions to `03_methodology_log.md`; update `00_start_here.md`.

If G3 passes, proceed to G4 (activation patching) — and at that point make the 3B→7B call (§8).
