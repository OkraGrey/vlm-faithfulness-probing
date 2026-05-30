# Full Literature Survey — VLM "Encoded-but-Not-Expressed" Probing

> **Purpose**: Report-ready related-work archive. Every paper here was fetched and verified (arXiv ID + title + authors confirmed) by research agents on 2026-05-30, except items explicitly marked UNVERIFIED. This is the raw material for the paper's Related Work section. Decisions made *from* this survey live in `06_literature_refresh_and_methodology_lock.md`; the design alternatives we weighed live in `08_design_alternatives_and_rationale.md`.
>
> Organized into the clusters a reviewer thinks in: (A) the phenomenon, (B) causal interventions, (C) threats to validity, (D) probing methodology foundations, (E) benchmarks & the model, (F) what we push back on. Each entry: **ID — title (year/venue)** · finding · relation to us.

---

## A. The phenomenon: information is internally present despite wrong/weak output

**This cluster is the reason our novelty is now the *method*, not the *phenomenon*.** These papers (especially A1) already establish that VLMs internally retain visual information their outputs discard. We must cite them as prior, and position our contribution as the rigorous causal-probing test on the wrong-output subset, not as the discovery of the effect.

- **2506.08008 — Fu, Bonnen, Guillory, Darrell, "Hidden in plain sight: VLMs overlook their visual representations" (Jun 2025).** *The key prior.* VLMs underperform a direct readout of their *own visual encoder* on vision-centric tasks (depth, correspondence), dropping toward chance; the bottleneck is the language component's *use* of vision, not the encoder. **Differentiator:** reads out the visual encoder, not LLM hidden states; vision-centric tasks, not BLINK; no wrong-output contrastive probe; no causal internal intervention.
- **2604.02486 — Shahgir et al., "VLMs Need Words: VLMs Ignore Visual Detail In Favor of Semantic Anchors" (Apr 2026).** *The paper the user dropped.* "Hidden-in-plain-sight" gap explained by *nameability*: VLMs map entities to semantic labels and reason verbally; nameable → success, unnameable → hallucinated descriptions. Method: Logit Lens + training-free MaxSim matcher on Qwen3VL/Gemma3/InternVL3.5 over SPair-71k / synthetic shapes / faces. Evidence is aggregate + correlational; causal evidence is *behavioral* (finetuning), not internal intervention. **Differentiator:** no trained probes, no controls, no wrong-output subset, not BLINK, not Qwen2.5-VL, no PAPO framing. Preprint, no code found.
- **2506.05439 — Takishita et al., "LLMs Can Compensate for Deficiencies in Visual Representations" (EMNLP 2025 Findings).** Self-attention ablations on 3 CLIP-based VLMs show the decoder can access/compensate for visual semantics — the opposite of PAPO's "deficient representation" premise. About compensation, not the wrong-output subset.
- **2511.19806 — Yao et al., "Reading Between the Lines: Abstaining from VLM-Generated OCR Errors via Latent Representation Probes" (Nov 2025).** Hidden-state/attention probes detect when OCR output is wrong; best signal in *intermediate* layers (a useful pointer for where to probe). OCR-only, abstention framing, no causal test.
- **2508.18297 — Ashok et al., "Can VLMs Recall Factual Associations From Visual References?" (EMNLP 2025 Findings).** VLMs link facts worse from visual than textual references; internal-state probes flag unreliable responses >92%. Factual recall, not perception/BLINK.
- **2510.17771 — Liu et al., "Seeing But Not Believing" (2025).** [team-known] Relative-Attention-Per-Token shows deep-layer attention localizes correctly even for incorrect outputs; uses inference-time attention masking on LLaVA/Qwen/Gemma/InternVL.
- **2502.03628 — Li et al., "The Hidden Life of Tokens: Reducing Hallucination of Large VLMs via Visual Information Steering" (2025).** [verified 2026-05-30 — note: first author is **Li**, NOT "Lin" as earlier docs stated] Visually-grounded tokens not expressed keep high vocabulary rankings (~rank 5,000/32K); language priors progressively dilute the visual signal.

## B. Causal interventions on visual-grounding signals (our G4 precedents)

- **2509.22674 — "Pathological Truth Bias in Vision-Language Models" (Sep 2025).** *Closest existing causal test.* MATS audit uses **activation patching** to localize failures to mid-to-late cross-attention layers; patching clean→corrupted flips erroneous outputs in ~23% of cases. On LLaVA-1.5/QwenVL-chat, spatial-contradiction audit — not Qwen2.5-VL, not BLINK, no PAPO framing. (Single-author listing — confirm authors before citing.)
- **2602.07025 — "The Geometry of Representational Failures in VLMs" (Feb 2026).** Distills "concept vectors" in Qwen/InternVL/Gemma; geometric overlap predicts errors; validated causally by **steering** (e.g., forcing "red flower" → perceived blue). Multi-object error geometry, not the wrong-output-grounding hypothesis.
- **2510.26769 — "SteerVLM: Robust Model Control through Lightweight Activation Steering for VLMs" (Oct 2025).** Inference-time steering module (0.14% of model size) controls outputs / mitigates hallucination. A steering *tool* we could cite/use; not a probing study.
- **2604.12119 — "Beyond Perception Errors: Semantic Fixation in Large VLMs" (Apr 2026).** *Conceptually our strongest ally.* Argues many "perception" errors are rule-mapping/"semantic fixation" failures, not encoding failures; **late-layer activation steering partially recovers** performance. Custom VLM-Fix benchmark, no probing study, no PAPO framing. (Single-author listing — confirm.)

## C. Threats to validity — pre-register against these

- **2603.18353 — "Interpretability without actionability: mechanistic methods cannot correct LM errors despite near-perfect internal representations" (Mar 2026).** *The central threat to our causal claim.* On Qwen2.5-7B, linear probes hit **98.2% AUROC** but output sensitivity is only **45.1%** — a 53-point **knowledge–action gap**; four mechanistic methods barely fix errors and damage correct cases. Text/clinical domain — we'd be first to test whether the gap holds for visual grounding on BLINK/Qwen2.5-VL. **This is why we design for the "encoded but not used" outcome.**
- **2603.06054 — "Probing Visual Concepts in Lightweight VLMs for Automated Driving" (Mar 2026).** *Our closest method neighbor.* Linear probes on **counterfactual image sets** separate **"perceptual failure" (not linearly encoded)** from **"cognitive failure" (encoded but not aligned to language)** — almost our exact taxonomy and our chosen construct-validity method. Driving domain, lightweight VLMs (Qwen2.5-VL not confirmed), no PAPO framing, no causal removal. **We adopt the design, differentiate on domain/model/causal stage/framing.**
- **2602.06652 — "Same Answer, Different Representations: Hidden instability in VLMs" (Feb 2026).** Outputs stay fixed while internal representations drift by near inter-image magnitude under benign perturbations (SEEDBench/MMMU/POPE). A probe-robustness caveat: "the representation encodes X" can be unstable.
- **2604.14888 — "Reasoning Dynamics and the Limits of Monitoring Modality Reliance in VLMs" (Apr 2026).** Across 18 VLMs, CoT can look visually grounded while actually following textual cues. Unfaithful-CoT pitfall — motivates probing hidden states over trusting verbalized reasoning.
- **2601.22150 — "Do VLMs Perceive or Recall? (VI-Probe)" (Jan 2026).** Probes perception-vs-memory with visual illusions; persistence of wrong answers has heterogeneous causes. Caution: don't assume a single "encoded-but-unexpressed" mechanism.
- **2509.03888 — "False Sense of Security" (NeurIPS 2025 Mech Interp Workshop).** [team-known] LLM/VLM probes drop 15–99 pp OOD; some reach near-zero on paraphrases. Makes OOD evaluation (G5) non-optional.
- **2509.19372 — Dubanowska et al., "Representation-based Broad Hallucination Detectors Fail to Generalize OOD" (EMNLP 2025 Findings).** [carried forward, UNVERIFIED this session] Complementary OOD-failure evidence for hallucination probes.

## D. Probing methodology foundations (our controls & analysis)

- **1610.01644 — Alain & Bengio, "Understanding intermediate layers using linear classifier probes" (2017).** Founding linear-probe method; random-init baseline (G2).
- **1909.03368 — Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" (EMNLP 2019).** Selectivity / control-task — the probe-memorization guardrail (used in G1, G3).
- **2102.12452 — Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances" (Comp. Linguistics 48(1)).** Survey of dataset-artifact, label-leakage, construct-validity confounds. (Abstract-level verification; standard reference.)
- **2005.00719 — Ravichander, Belinkov, Hovy, "Probing the Probing Paradigm" (EACL 2021).** Models encode properties they don't use; pretrained embeddings (not the task) can drive probe accuracy. The construct-validity warning behind G3.
- **2004.03061 — Pimentel et al., "Information-Theoretic Probing for Linguistic Structure" (ACL 2020).** Probe accuracy as mutual-information estimate; argues for highest-capacity probes. We deliberately do *not* follow this (our question is causal, not info-quantity) — cite to justify the trade-off.
- **2310.06824 — Marks & Tegmark, "The Geometry of Truth" (COLM 2024).** Difference-in-means probes + causal interventions; truth directions are linear and *more causally implicated* than fitted probes. Basis for our primary (DiM) probe choice.
- **2106.02997 / 2112.00826 — Geiger et al., "Causal Abstractions of Neural Networks" / "Inducing Causal Structure" (interchange interventions).** The causal-abstraction logic behind the counterfactual image-swap (G3) and patching (G4). (Attribution corroborated across sources; abstract pages not each individually opened.)
- **2202.05262 — Meng et al., "Locating and Editing Factual Associations in GPT" (ROME, causal tracing/activation patching).** The patching machinery for G4.
- **2207.04153 — Kumar, Tan & Sharma, "Probing Classifiers are Unreliable for Concept Removal and Detection" (NeurIPS 2022).** Proves post-hoc removal (INLP) can't fully remove a concept and may destroy collateral features — **why we demote INLP.**
- **2306.03819 — Belrose et al., "LEACE: Perfect linear concept erasure in closed form" (NeurIPS 2023).** Provably minimal-edit linear erasure — INLP's replacement if we erase.
- **2506.11673 — "Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE" (Findings ACL 2025).** [UNVERIFIED — confirm ID] LEACE/mean-projection pass info-control tests INLP fails.
- **2006.00995 — Elazar et al., "Amnesic Probing" (TACL 2021).** Behavioral-explanation-via-removal; headline caution "probing performance is not correlated to task importance." Cite as motivation + caveat, not as our primary tool.

## E. Benchmarks & model

- **2404.12390 — Fu et al., "BLINK: Multimodal LLMs Can See but Not Perceive" (ECCV 2024).** Our primary benchmark. 3,807 MCQs across 14 perception tasks; constructed so caption+LLM ≈ random, so any failure is perception-grounded by construction (eliminates the need for detector-based attribution).
- **2512.21194 — VisRes Bench.** [carried forward, UNVERIFIED this session] Three-level perception→reasoning hierarchy; the planned OOD target for G5. Verify before relying on it.
- **Model: Qwen2.5-VL.** 3B = 36 LLM-decoder layers @ 2048-dim (current data); 7B = 28 layers @ 3584-dim (spec target). ViT 32 layers @ 1280-dim + 2-layer MLP merger. HALP's Qwen2.5-VL probe accuracies are 7B.
- **2603.05465 — HALP, "Detecting Hallucinations in VLMs without Generating a Single Token" (EACL 2026).** Single-forward-pass probes over three representation families, up to 0.93 AUROC; **for Qwen2.5-VL specifically, the signal sits in *visual* features (~0.79 AUROC)** unlike most models where late query-token states win — a methodological pointer for where to probe in the 7B run.

## F. What we push back on

- **2507.06448 — Wang et al., "PAPO: Perception-Aware Policy Optimization" (ICLR 2026).** Implicit Perception Loss + Double Entropy stabilizer for GRPO/DAPO. Identifies perception as the major error source from 200 manually-annotated math/STEM errors (Geometry3K/MMK12/LogicVista/MathVerse — **not** HallusionBench); never probes internals. **Caveat:** the fetched page confirms it treats perception as the bottleneck but did *not* state the strong phrasing "errors = deficient internal visual representation." Quote PAPO precisely; rebut its *premise that perception is the bottleneck*.
- Follow-up loss-family work to acknowledge: CPPO (2601.00501), VPPO-RL — all operate at output/loss level, none probes internals. [UNVERIFIED this session]

## G. Withdrawn / eliminated (kept so they aren't re-cited)

- **2508.00378 — CoRGI** (Grounding DINO + step-wise CoT on HallusionBench) — **withdrawn from arXiv 2025-10-14** after +1.3–2.3 pt gains. The colleague's original pipeline was a parallel instance; structurally predicted to fail.
- **2303.05499 — Grounding DINO** — real but unsuited to charts/OCR/illusions; false positives on absent entities. Eliminated by the BLINK switch.

---

## Verification ledger (read before formal citation)

- **Re-fetched & confirmed (first pass):** 2507.06448, 2510.17771, 2506.08008, 2506.05439, 2511.19806, 2508.18297, 2603.05465, 2509.22674, 2602.07025, 2510.26769, 2604.12119, 2603.18353, 2602.06652, 2604.14888, 2603.06054, 2601.22150, 2604.02486, 1909.03368, 2310.06824, 2202.05262, 2207.04153, 2306.03819, 2004.03061, 2005.00719, 2006.00995, 1610.01644, 2404.12390, 2509.03888, 2102.12452 (abstract-level).
- **Geiger 2106.02997 / 2112.00826:** attribution corroborated via search; individual abstract pages not each opened.

### Verified on 2026-05-30 (second pass) — abstract pages fetched individually

All 10 previously-flagged IDs resolved. None failed.

1. **2506.11673** — VERIFIED, title exact. Dobrzeniecka, Fokkens, Sommerauer. ACL 2025 Findings.
2. **2509.19372** — VERIFIED. Exact title "…Fail to Generalize Out of Distribution" (docs abbreviate "OOD"). Dubanowska, Żelaszczyk, Brzozowski, Mandica, Karpowicz. EMNLP 2025 Findings.
3. **2512.21194** — VERIFIED. "VisRes Bench: On Evaluating the Visual Reasoning Capabilities of VLMs." Malagurski Törtei et al. (8 authors). Preprint Dec 2025.
4. **2601.00269** — VERIFIED. "FaithSCAN: Model-Driven Single-Pass Hallucination Detection for Faithful Visual Question Answering." Tong, Zhang, Li, Jiang, Liu.
5. **2601.05547** — VERIFIED. "VIB-Probe: Detecting and Mitigating Hallucinations in VLMs via Variational Information Bottleneck." Zhang, Wu, Wang, Wang, Lv, Huang, Zheng.
6. **2410.07149** — VERIFIED. "Towards Interpreting Visual Information Processing in Vision-Language Models." Neo, Ong, Torr, Geva, Krueger, Barez. ICLR 2025.
7. **2502.03628** — VERIFIED **with author correction**: "The Hidden Life of Tokens…", first author **Zhuowei Li** (10 authors). Docs' "Lin et al." was WRONG → use **Li et al.** (fixed in §A).
8. **2601.00501** — VERIFIED. "CPPO: Contrastive Perception Policy Optimization for VLM Agents." Rezaei et al. (8 authors).
9. **2509.22674** — VERIFIED, **single author** (Yash Thube) confirmed — §B hedge resolved.
10. **2604.12119** — VERIFIED, **single author** (Md Tanvirul Alam) confirmed — §B hedge resolved.

### DID NOT RESOLVE — do not cite
*(None — all 10 verified on the 2026-05-30 second pass.)*

- **Newest preprints (2604.x):** real and resolving, but treat as non-peer-reviewed until publication is confirmed.
