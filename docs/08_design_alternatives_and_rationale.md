# Design Alternatives & Rationale — The Roads Not Taken

> **Purpose**: A record of the methodological decisions we made on 2026-05-30, each with the alternatives considered, the choice, the reason, and the condition under which we'd revisit it. This is the source material for the paper's *Methodology justification*, *Limitations*, and *Future Work* sections — and a guard against re-arguing settled decisions. Decisions are locked in `06_…methodology_lock.md`; the literature backing them is in `07_literature_survey_full.md`.
>
> Format per decision: **Options → Chosen → Why → Revisit if.**

---

## D1. How do we establish the probe signal is *visual* (construct validity)?

This is the crux of the whole project. A probe that predicts the gold answer from hidden states might read visual grounding, or question/option structure, or label artifacts.

**Options considered:**
1. **Blank-image / text-only ablation** (the original `04_next_step_vision_ablation.md` proposal): re-run with no image, train identical probes; if text-only matches image+text, signal isn't visual.
2. **Counterfactual image-swap**: hold the question + option set fixed, swap in an image whose correct answer is a *different* option; require the probe's prediction to track the image.
3. **Image-shuffle / noise control**: replace image with a shuffled or noise version.
4. **Attention-knockout**: zero attention to visual tokens and measure probe degradation.

**Chosen:** #2 (counterfactual image-swap) as **primary**, #1 (blank-image) as a cheap **secondary null**.

**Why:** Blank-image leaves the question and options intact, so a probe can still score off question-conditioned priors that correlate with the gold letter — meaning a "blank-image probe collapses" result could over-claim grounding, and a "blank-image probe survives" could be misread as "no vision." The counterfactual swap *varies the visual content while freezing every linguistic confound*, isolating the construct "visual grounding" by construction (interchange-intervention logic, Geiger et al. 2106.02997 / 2112.00826; the warning it addresses is Ravichander et al. 2005.00719). An independent 2026 paper (2603.06054) uses exactly this counterfactual-image-set design to cleanly separate "perceptual failure" from "cognitive failure," confirming it is a published, sound method. Blank-image is retained because it is nearly free and rules out the strongest pure-language null.

**Revisit if:** counterfactual pairs can't be mined with matched questions for a task (then fall back to noise-image control + attention-knockout as convergent evidence).

---

## D2. Which probe — high-capacity or low-capacity?

**Options:** (a) L2 logistic regression (current primary); (b) difference-in-means (DiM, zero-trained); (c) 1-hidden-layer MLP.

**Chosen:** **DiM as primary**, logreg as secondary/complementary, MLP only as a linear-vs-nonlinear ablation.

**Why:** Two reasons. (1) **Causal alignment:** Marks & Tegmark (2310.06824) show DiM directions are *more causally implicated* in model outputs than fitted probes — and our endgame is a causal claim (G4), so the probe object should carry into the intervention. (2) **Overfitting:** with 2048-dim features and ~50–90 training samples, logreg overfits (our own Gate 1 saw DiM *beat* logreg on Jigsaw/Multi-view/Art_Style — a tell-tale of logreg overfitting, not of weak signal). Pimentel et al. (2004.03061) argue for the highest-capacity probe — but that is the right advice for "how much information is present," not for our causal/behavioral question. We cite Pimentel to justify deliberately not maximizing capacity.

**Revisit if:** we reframe toward an information-quantity claim, in which case capacity should be maximized and MI-style reporting added.

---

## D3. Causal test — how do we show the model *uses* the encoded direction?

**Options:** (a) Amnesic probing via INLP (the pre-registered Gate 5); (b) LEACE concept erasure; (c) activation patching / interchange intervention; (d) activation steering.

**Chosen:** **activation patching as primary** (patch a correct-run hidden state into a wrong-run forward pass, measure flip-to-gold vs. a random-direction patch control); **LEACE if we erase at all**; **steering as corroboration only**; **INLP demoted.**

**Why:** INLP makes a load-bearing causal claim unsafe — Kumar, Tan & Sharma (2207.04153, NeurIPS 2022) prove it cannot fully remove a concept and can destroy collateral task features, so an accuracy drop after INLP is *ambiguous* (did we erase the answer direction, or unrelated info?). Elazar et al. (2006.00995) themselves caution against strong causal reads. Activation patching answers "does the model use this state?" directly with a forward-pass intervention, no retraining, cheap on a 3B/7B forward pass, and has a clean control (random-direction patch = the patching analogue of selectivity). It also has direct VLM precedent (2509.22674 flips ~23% of wrong outputs by patching). If concept erasure is still wanted for a complementary result, LEACE (2306.03819) is the closed-form, minimal-edit successor that passes info-control tests INLP fails. Steering (add/subtract the DiM direction) is a useful third signal but is a global perturbation and more confound-prone, so it corroborates rather than carries the claim.

**Revisit if:** a reviewer specifically wants amnesic-probing comparability with Elazar et al. — then run LEACE (not INLP) as the erasure variant and report alongside patching.

---

## D4. Layer selection — how do we avoid the winner's curse?

**Options:** (a) best layer by wrong-subset accuracy (current `baselines.py`); (b) best layer by all-test accuracy, report wrong-subset there; (c) nested k-fold CV with layer selection inside the inner fold; (d) report the full layer-wise curve / a layer profile, no single peak.

**Chosen:** **nested k-fold CV (c)** as the honest estimator, **plus** reporting (b)-style fixed-layer CV at the originally-named layer for comparability, **plus** the full curve (d) in the paper.

**Why:** The current code maximizes over 36 layers on **4–25 wrong-test samples**, then reports that layer's CI uncorrected — a textbook selection bias that inflates the peak. Nested CV moves layer selection inside the training partition so the reported accuracy is unbiased by selection. We additionally pin the originally-reported layers (Forensic L27, Object_Localization L21) and re-estimate under CV so the new and old analyses are directly comparable. The full curve guards against a reviewer suspecting a cherry-picked layer.

**Revisit if:** per-task wrong-subset N is too small even for CV (then pool tasks or report all-subset accuracy only, per `05_open_questions.md` Q2/Q4).

---

## D5. Sample size & whether to scale to 7B

**Options:** (a) stand on the current 3B wrong-output N (≈57–91 per surviving task); (b) augment N via temperature/paraphrase sampling on the same BLINK items; (c) move to Qwen2.5-VL-7B.

**Chosen:** **phased** — (a)+CV for G0/G3 on existing 3B data (free); **(c) 7B for the causal (G4) + OOD (G5) + final claims**, with (b) as a supplementary lever if 7B wrong-N is still thin.

**Why:** G0 and G3 either kill or confirm the hypothesis cheaply on data we already have. But activation-patching flip-rates at N≈57 have wide CIs — a null there would be *underpowered, not negative*. The 7B run is justified independently (spec already targets it; HALP's Qwen2.5-VL probe numbers are 7B; 3B survivors are statistically thin) and *doubles as the generality check* (does the phenomenon survive scale?). We explicitly do **not** add more *tasks* (2/14 surviving is a real empirical result, not a power problem) and do **not** fabricate an a-priori power number (no honest published effect-size anchor for patching flip-rates exists); rigor comes from k-fold CV + bootstrap CIs + the random-direction control.

**Revisit if:** advisor confirms no GPU budget — then stay on 3B, augment via (b), and scope every claim to "3B, suggestive, generality untested."

---

## D6. Outcome framing — binary or trichotomy?

**Options:** (a) binary "the model sees but can't say" (YES) vs. "representation is deficient" (NO); (b) a trichotomy adding "encoded but not used."

**Chosen:** **trichotomy (b)**, pre-registered.

**Why:** The knowledge–action gap paper (2603.18353) found 98% probe AUROC alongside only ~45% intervention success — so "encoded but dormant" is likely the *most probable* outcome, and it is the *sharpest* scientific result: it partially refutes PAPO (representation not deficient) *and* partially vindicates it (model still fails to use the info), reframing the debate as access/usage failure rather than representation failure. Pre-registering all three outcomes prevents result-shopping (we commit to the interpretation of each before running).

**Revisit:** not a revisit item — this framing should hold regardless of result; it only changes which branch we report as the headline.

---

## D7. Benchmark & OOD target

**Options:** primary BLINK (locked) vs. HallusionBench (abandoned, see `01`/`02`). OOD: VisRes Bench vs. MMVP vs. held-out BLINK task types.

**Chosen:** **BLINK primary** (construct guarantee: caption+LLM ≈ random); OOD = held-out BLINK task types (secondary, cheap) + **one external perception benchmark** for G5 (VisRes Bench if its existence/access verifies; MMVP as fallback).

**Why:** BLINK's construction removes the perception/reasoning attribution problem that sank the original Grounding-DINO pipeline. For OOD, "False Sense of Security" (2509.03888) makes a genuine distribution shift mandatory; held-out task types are the cheapest shift, an external benchmark the strongest. **VisRes Bench (2512.21194) is UNVERIFIED** — confirm it exists and is accessible before committing; otherwise MMVP.

**Revisit if:** VisRes Bench is inaccessible or doesn't fit (swap to MMVP or POPE-style perception split).

---

## Parking lot — options we have NOT pursued but might (Future Work seeds)

- **Logit Lens on the visual tokens** (as in 2604.02486 / 2410.07149) as a *training-free* complement to our trained probes — cheap convergent evidence.
- **Cross-architecture replication** (LLaVA-1.6, InternVL3.5) to claim a model-general phenomenon (`05_open_questions.md` Q8).
- **Decoding-time intervention as the "so what"**: if G4 shows the direction is usable, steer at inference to fix wrong-output cases (the practical-impact experiment that lifts the paper from Findings to main-track; `05` Q10).
- **Probe-instability stress test** (per 2602.06652): perturb images benignly, measure whether the grounding direction is stable — a robustness section.
- **Nameability axis** (per 2604.02486): split BLINK errors by whether the target is nameable, to test if our "encoded" signal concentrates on unnameable entities — would directly engage the semantic-anchor mechanism.
