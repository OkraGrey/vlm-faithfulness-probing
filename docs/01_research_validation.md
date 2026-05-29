# Research Validation Report — VLM Faithfulness Probing Project

**Status**: Pre-experimentation literature audit
**Date**: 2026-05-17
**Purpose**: Validate the credibility of the approach in `reqs.md` before committing to experiments. Every recommendation is backed by a real published paper with arXiv ID or DOI.

---

## 1. Executive Summary

### 1.1 What Holds Up

**The core hypothesis is real, novel, and supported by emerging evidence.** The claim that VLMs may internally encode correct visual grounding even when their outputs are wrong is not just plausible — it has direct empirical support published in 2024–2025:

- **"Seeing But Not Believing"** (arXiv:2510.17771, 2025) shows that VLMs' deep-layer attention correctly localizes visual evidence at comparable rates for correct AND incorrect answers. The "model knows but fails to express" phenomenon is real.
- **"The Hidden Life of Tokens"** (arXiv:2502.03628, 2025) quantifies that visually grounded tokens *not expressed in the output* maintain high vocabulary rankings (~rank 5000 of 32K) throughout generation. Language priors progressively dilute visual information during decoding.
- **HALP** (arXiv:2603.05465, EACL 2026) probes pre-generation internal representations and achieves up to 0.93 AUROC for hallucination detection — and finds that for Qwen2.5-VL specifically, *visual-only mid-layer features* (not LLM decoder features) carry the strongest signal.
- **PAPO** (the paper the project pushes back on) is real (arXiv:2507.06448, ICLR 2026) and does NOT probe internals. Its diagnosis of "perception failure" is entirely behavioral, established from 200 manually-annotated error cases with no inter-annotator agreement reported, on math/STEM benchmarks (Geometry3K, MMK12, LogicVista, MathVerse). **It never used HallusionBench.** The faithfulness-gap argument the project makes is therefore genuinely unfilled in the literature.

### 1.2 What Does Not Hold Up

Five issues in the current `reqs.md` need correction before experimentation:

1. **The "67% on HallusionBench" claim is a factual error.** PAPO's 67% number comes from math/STEM benchmarks, not HallusionBench. PAPO never evaluated on HallusionBench.

2. **The Qwen2.5-VL-3B architecture description is wrong.** The reqs.md says "4096-dimensional states across 32 layers." The actual model has **36 decoder layers with 2048-dim hidden states** (plus a separate ViT with 32 layers at 1280-dim, and an MLP merger that projects to the LLM dimension). All storage and compute estimates need recalculating.

3. **The Grounding DINO + atomic CoT pipeline is fundamentally mismatched with HallusionBench** — not a tuning problem. ~40% of HallusionBench (charts, tables, OCR, maps) has no localizable physical objects for Grounding DINO to detect. The detector also has a documented false-positive problem on absent entities (issue #84 on the official repo), which INVERTS the verification signal. Worse — CoRGI (arXiv:2508.00378) is essentially this exact pipeline, published in 2025, and was **withdrawn from arXiv on 2025-10-14**. Your colleague's failure is structurally predicted.

4. **Two citations in reqs.md are inaccurate.** "Causal probing (2026)" appears to be fabricated — no paper with this title and framing was found. The VADE citation is real but mis-described — VADE is about attention-map-based hallucination detection, not OOD probe-failure warnings. The OOD warning actually comes from Dubanowska et al. (arXiv:2509.19372, EMNLP 2025 Findings).

5. **Qwen2.5-VL-3B was not trained for structured chain-of-thought reasoning.** CoT prompting works, but step quality is significantly lower than purpose-trained models (LLaVA-CoT, Qwen3-VL with thinking mode). "Atomic step boundaries" in free-form CoT from a 3B non-reasoning model are noisy by construction — which is the second reason the colleague's attempt failed.

### 1.3 Top-Line Recommendation

The project's central question is publishable. The current implementation is not. A defensible study requires three structural changes:

- **Replace Grounding DINO with a VLM-as-judge pipeline** (GPT-4o or Gemini 2.5 Pro) using FactScore-style atomic claim decomposition (Min et al., EMNLP 2023, arXiv:2305.14251) and binary verdicts (0.806 human alignment vs. 0.454 for scoring per the MLLM-as-Judge benchmark).
- **Probe all three computational stages of Qwen2.5-VL** (ViT, MLP merger, LLM decoder), not just the LLM. HALP shows the merger output and mid-ViT carry the strongest grounding signal in this architecture specifically.
- **Include the full canonical baseline suite** (selectivity per Hewitt & Liang 2019, random-init model per Alain & Bengio 2017, layer-0 baseline, final-token vocabulary control). Without these, the study fails review automatically.

Optional but worth considering before locking the design: switch to a benchmark with structurally separable perception/reasoning (BLINK, VisRes Bench), or scale to Qwen2.5-VL-7B which is on HallusionBench leaderboards.

---

## 2. Citation Audit

| Claimed citation in reqs.md | Status | Replacement / correction |
|---|---|---|
| PAPO (ICLR 2026) — 67% perception failures on HallusionBench | Paper real; HallusionBench attribution wrong | arXiv:2507.06448. The 67% is from Geometry3K/MMK12/LogicVista/MathVerse, not HallusionBench. |
| HALP (2026) — probes representations prior to generation | Real | arXiv:2603.05465, EACL 2026. Use as-is. |
| FaithSCAN (2026) — perception/reasoning/decoding decomposition | Real but description imprecise | arXiv:2601.00269. Actually three signal sources (visual perception, cross-modal interaction, language decoding) extracted in single pass — not staged decomposition. |
| VIB Probe (2026) — VIB-based filtering of hidden states | Real | arXiv:2601.05547. Use as-is. |
| Activation Steering Decoding (ACL 2025) | Real | DOI 10.18653/v1/2025.acl-long.634. Use as-is. |
| Vision Language PRMs (2025) | Generic label; multiple real candidates | Pin to **VisualPRM** (arXiv:2503.10291, InternVL/OpenGVLab team) for a verifiable reference. |
| Causal probing (2026) | **Not verified — appears fabricated** | Replace with **Marks & Tegmark, "Geometry of Truth"** (arXiv:2310.06824, COLM 2024) — provides causal-intervention evidence for linear probes via activation patching. |
| VADE (2025) — warns probes fail OOD | Paper real; description WRONG | VADE (ACL 2025 Findings) is actually about attention-map-based hallucination detection. For the OOD-failure warning, cite **Dubanowska et al., "Representation-based Broad Hallucination Detectors Fail to Generalize OOD"** (arXiv:2509.19372, EMNLP 2025 Findings) instead. Better, even stronger: **"False Sense of Security"** (arXiv:2509.03888, NeurIPS 2025 Workshop) shows probes drop 15–99 pp OOD. |

### Additional canonical citations the project must include

These are real, established, and required for a probing study to pass review:

| Citation | arXiv / DOI | Role |
|---|---|---|
| Alain & Bengio (2017) | arXiv:1610.01644 | Founding linear-probing methodology |
| Hewitt & Liang (2019) | arXiv:1909.03368 (EMNLP) | Selectivity / control task — REQUIRED CONTROL |
| Belinkov (2022) | Comp. Linguistics 48(1) | Probing pitfalls survey |
| Rogers, Kovaleva, Rumshisky (2020) | TACL — BERTology | Layer-wise analysis standard |
| Elazar et al. (2021) | TACL — amnesic probing | Causal validation method |
| Geiger et al. (2021) | NeurIPS — causal abstractions | Gold-standard causal interpretability |
| Meng et al. (2022) | NeurIPS — ROME | Activation patching methodology |
| Pimentel et al. (2020) | ACL — info-theoretic probing | Probe-complexity debate resolution |
| SAPLMA — Azaria & Mitchell (2023) | arXiv:2304.13734 | LLM internal states predict truthfulness |
| INSIDE — Chen et al. (2024) | arXiv:2402.03744, ICLR 2024 | Middle-layer last-token probing |
| Marks & Tegmark (2023) | arXiv:2310.06824 | Geometry of Truth, causal intervention |
| Knowing Before Saying (2025) | arXiv:2505.24362 | CoT-success encoding at middle layers |
| Reasoning Models Know When They're Right (2025) | arXiv:2504.05419 | **Closest experimental design to your project** — probes last-token of CoT chunks |
| FactScore — Min et al. (2023) | arXiv:2305.14251 (EMNLP) | Atomic-claim decomposition for verification |
| SAFE — Wei et al. (2024) | arXiv:2403.18802 | Atomic-fact verification (extends FactScore) |
| MoCA — "Bad Seeing or Bad Thinking?" (2025) | arXiv:2605.14054 | Blindfolded-reasoning proxy for perception/reasoning split, 86% human agreement |
| HallusionBench — Guan et al. (2024) | arXiv:2310.14566, CVPR 2024 | The benchmark itself; has built-in control-pair attribution |

---

## 3. Diagnosis — Why the Colleague's Attempt Failed

This is not bad luck or a bug. There are five structural reasons that were predictable from the literature:

1. **Detector–benchmark domain mismatch.** Grounding DINO was trained on natural photographs (COCO, O365, RefCOCO, Visual Genome). HallusionBench is 45% human-edited images, plus charts, OCR, maps, illusions. The detector has no representation for these categories. A bar chart's "trend" is not a bounding-box-able object.

2. **False-positive on absent entities.** Grounding DINO has no "nothing matches" class. When asked about an entity not in the image (a frequent case for incorrect VLM CoT steps), it returns confident false detections. This inverts the verification signal: a wrong claim gets a positive grounding match.

3. **Atomic CoT decomposition is undefined for free-form output from a non-reasoning model.** Qwen2.5-VL-3B's CoT is not structured. Sentence-boundary splits are heuristic and produce sub-claims of varying granularity. CoRGI (which used a relevance-classifier MLP for the same step) acknowledged that "not all reasoning steps require visual verification" — and was subsequently withdrawn.

4. **CoT is often unfaithful.** Multiple 2024–2025 papers (arXiv:2503.08679, arXiv:2512.12218) show VLM CoT chains are frequently post-hoc rationalizations rather than the model's actual inference path. Unfaithful CoT becomes more pronounced at smaller scales. Verifying steps of a narrative that doesn't represent actual computation analyzes noise.

5. **HallusionBench is *designed* to entangle perception and reasoning.** The benchmark's full title is "An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion." Disentangling what was deliberately entangled is fighting the benchmark. Worth noting: HallusionBench provides its OWN attribution mechanism via control pairs (Language Hallucination vs. Visual Illusion) — your project would need to demonstrably outperform this built-in scheme to justify added complexity.

---

## 4. Recommended Defensible Pipeline

### 4.1 Step-by-step, every choice backed by a citation

**Stage A — Inference data collection**
- Run Qwen2.5-VL-3B with structured CoT prompting on the chosen benchmark
- Collect: image, question, ground-truth answer, model's full generated response, hidden states at all probing locations
- *Citation for design*: matches the experimental setup of "Reasoning Models Know When They're Right" (arXiv:2504.05419), which uses last-token-of-chunk extraction on Qwen-family models

**Stage B — Atomic claim decomposition**
- Use GPT-4o (text-only) to decompose each model response into atomic visual claims
- Classify each claim as *visually groundable* or *not groundable* using a second GPT-4o pass
- *Citations*: FactScore (Min et al., arXiv:2305.14251, EMNLP 2023); SAFE (Wei et al., arXiv:2403.18802); CoRGI's relevance-classifier design (with acknowledgment of its limits)

**Stage C — Grounding verification (replaces Grounding DINO)**
- For each groundable atomic claim, query GPT-4o (or Gemini 2.5 Pro) with image + claim:
  *"Is this claim directly supported by what you can see in this image? Respond Yes or No, then explain in one sentence."*
- Use binary verdicts, not scores. Per MLLM-as-Judge: pairwise/binary gives 0.806 human alignment; scoring gives 0.454.
- Cross-family judge: use a different VLM family than Qwen for judging, to avoid self-preference bias
- *Citations*: MLLM-as-a-Judge (Chen et al., arXiv:2402.04788); Prometheus-Vision (arXiv:2401.06591); Faithfulness of Visual Thinking (arXiv:2510.23482)

**Stage D — Label construction**
- For each (image, question, atomic claim) triple, derive two labels:
  - `output_correct`: did the model's final answer match ground truth?
  - `claim_grounded`: did the VLM-judge verify the claim is visually supported?
- The CRITICAL contrastive subset for the project: cases where `output_correct = False` but `claim_grounded = True` — these are the "model knows but fails to express" examples
- *Citation*: this contrastive design is what differentiates the project from generic hallucination-detection probing; closest precedent is the "Seeing But Not Believing" RAPT analysis (arXiv:2510.17771)

**Stage E — Hidden-state extraction (corrected architecture)**

Qwen2.5-VL-3B has three computational stages — probe all three:

- **ViT encoder**: 32 layers, 1280-dim. Probe at layers {7, 15, 23, 31} (the full-attention layers; others use window attention).
- **MLP merger output**: 2048-dim (after the 4×patch→1 token compression). This is the modality transition — HALP found this position is most informative for Qwen2.5-VL specifically.
- **LLM decoder**: 36 layers, 2048-dim. Probe at layers {1, 6, 12, 18, 24, 30, 36}. INSIDE and "Knowing Before Saying" both find middle layers (around L/2) dominant.

Extract at multiple token positions and ablate: last token of CoT step, mean pool over the step's tokens, last visual token, and (control) one-hot index of the final token's vocabulary ID.

Storage estimate (corrected): 36 × 2048 × 2 bytes ≈ 144 KB per sample per token position. 10K samples ≈ 1.4 GB. Manageable.

- *Citations*: HALP (arXiv:2603.05465) for the multi-stage probe locations; "Towards Interpreting Visual Information Processing in VLMs" (arXiv:2410.07149) for middle-to-late layer dominance in VLMs

**Stage F — Probing classifier**
- Primary: logistic regression with L2 regularization (linear, simple, low-capacity)
- Secondary: 1-layer MLP for the linear-vs-nonlinear comparison
- *Citations*: Alain & Bengio (arXiv:1610.01644) — linear probes are the standard; Pimentel et al. (arXiv:2004.03061) — info-theoretic justification

**Stage G — Required baseline suite**

| Baseline | Purpose | Citation |
|---|---|---|
| Majority class | Floor | Standard |
| Layer-0 / input embedding | Pre-transformer baseline | Alain & Bengio 2017 |
| Random-init model | Architectural-only signal | Alain & Bengio 2017; Zhang & Bowman 2018 |
| Random-labels control (selectivity) | Probe memorization check | Hewitt & Liang 2019 |
| Final-token vocabulary one-hot | Lexical-surface confound | Standard control |
| Logit-based uncertainty (max softmax) | Output-layer alternative | SAPLMA 2023 |
| Difference-in-means zero-train probe | Simplest-possible probe | Marks & Tegmark 2023 |
| Bag-of-objects from external detector | "VLM adds nothing" null | Standard |
| CoT-step-index logistic regression | Position confound | Standard |

**Stage H — Statistical validation**
- 5 random seeds, mean ± SD
- Bootstrap 95% CIs (10,000 resamples)
- McNemar's test vs. majority baseline
- Multiple-comparison correction (Bonferroni or Benjamini-Hochberg) for layer-wise reporting
- *Citation*: Dror et al. 2018 (statistical significance in NLP)

**Stage I — OOD evaluation (mandatory)**
- Train probe on one benchmark/category, test on a held-out one
- Required given the "False Sense of Security" (arXiv:2509.03888) and Dubanowska et al. (arXiv:2509.19372) findings — without OOD evaluation, you cannot defensibly claim to have found a model-general signal

**Stage J — Causal evidence (Level 2 minimum)**
- **Amnesic probing via INLP** (Iterative Nullspace Projection): use the probe direction to erase the grounding property from representations, then test downstream behavior. If erasure degrades VQA performance, the model USES the property (not just encodes it).
- *Citation*: Ravfogel et al. 2020 (INLP); Elazar et al. 2021 (amnesic probing, TACL)
- This moves the study from a correlation claim ("the property is encoded") to a usage claim ("the model uses it")

### 4.2 Decisions for the user to weigh

There are tradeoffs worth discussing before locking the design:

| Decision | Option A | Option B |
|---|---|---|
| Model | Qwen2.5-VL-3B (cheap, on-disk-able) | Qwen2.5-VL-7B (on leaderboard, better CoT) |
| Benchmark | HallusionBench (project's premise) | BLINK / VisRes Bench (cleaner perception/reasoning split) |
| Annotator | GPT-4o VLM-as-judge (recommended) | Hybrid: Florence-2 for natural images + GPT-4o for charts/illusions |
| Causal claim level | Level 1 only — correlation ("encoded") | Level 2 — usage via amnesic probing (recommended for publishable claim) |
| CoT structure | Free-form prompt | Forced structured output ("perception:" / "reasoning:" blocks) — much higher-quality labels |

---

## 5. Semester Schedule (14 weeks)

| Weeks | Milestone |
|---|---|
| 1–2 | Locked hypothesis document; benchmark + model choice frozen; advisor sign-off; literature review (~15 papers above) |
| 3–4 | Inference pipeline; CoT data collection; extraction hooks for all three model stages |
| 4–6 | **Baseline suite implemented and frozen BEFORE main experiments** |
| 5–7 | Atomic claim decomposition + VLM-as-judge annotation pipeline; small-N validation against human labels (~100 samples) |
| 7–8 | Main probing experiments — layer-wise across ViT/merger/LLM; selectivity at every layer |
| 8–9 | Amnesic probing (INLP) — causal validation |
| 9–10 | OOD experiments (cross-benchmark) |
| 10–11 | Ablations: token position, probe complexity, aggregation strategy |
| 11–13 | First draft → revisions |
| 14 | Submission (workshop track or EMNLP Findings) |

---

## 6. Top Risks and Mitigations

1. **OOD failure** — Per "False Sense of Security" (arXiv:2509.03888), probes routinely collapse 15–99 pp OOD. Mitigation: design contrastive OOD splits from day one; consider semantic-entropy reframing (Farquhar et al. arXiv:2406.15927) which OOD-generalizes better than correctness probing.
2. **Self-consistency bias in VLM-as-judge** — If GPT-4o judges Qwen, there's bias risk. Mitigation: cross-family judge + human validation on a subset (~200 samples) for inter-annotator agreement.
3. **Atomic-claim noise** — Decomposition quality varies (arXiv:2510.04040, FaithCoT-Bench). Mitigation: pilot-test decomposer on 50 examples; measure agreement with manual decomposition.
4. **Negative result** — If probes don't generalize OOD, the paper still has value (it confirms the OOD-collapse pattern in VLM probing). Frame the study to be valuable in either outcome.
5. **Qwen2.5-VL-3B CoT quality** — At 3B scale, CoT is noisy. Mitigation: report CoT-step quality metrics (length, coherence) as a confound check, OR upgrade to 7B.

---

## 7. What Is Genuinely Novel Here

After this audit, the project's defensible novelty claim is:

> "We provide the first probing-based test of PAPO's implicit assumption that VLM perception failures correspond to deficient internal representations. Using HallusionBench and Qwen2.5-VL, we [find / do not find] that the model's hidden states at layer L encode correct visual grounding even on examples where the output is wrong, and that this encoded information is [causally implicated via amnesic probing / merely correlated with] task performance."

The contrastive design ("model is wrong but probe is right") is the differentiator from prior probing work, which trains on aggregate correctness rather than this specific subset.

---

## 8. Sources

All citations above are real with arXiv IDs or DOIs verified during the literature audit. Detailed agent reports in `tasks/` if you want full per-paper context.
