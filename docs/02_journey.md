# Project Journey — Probing VLM Faithfulness

A narrative record of how this semester's research project moved from idea to first usable dataset. Written so a reader returning weeks later (or a future collaborator reading cold) can reconstruct what was done, why, what state the code and data are in, and what is still open.

Last updated: 2026-05-17.

---

## 1. Project Overview

This project tests a single scientific claim: **on visual perception tasks where a Vision-Language Model (VLM) produces a wrong answer, does the model's internal residual stream already encode the correct grounding signal, or is the visual representation genuinely deficient?** The question matters because PAPO (Wang et al., ICLR 2026, arXiv:2507.06448) and the growing family of "perception-aware" training-loss interventions all rest on the unverified assumption that perception failures reflect bad internal representations. If a linear probe on Qwen2.5-VL's hidden states can recover the correct answer on the *wrong-output* subset — significantly above baselines, with proper selectivity controls, holding under amnesic ablation, and not collapsing out-of-distribution — that would be evidence the training-loss family of fixes is masking a translation failure rather than repairing a representation failure. A null result is informative too: it would support PAPO's premise. The full spec lives in `/Users/hasnainsohail/Documents/Computer/LUMS/PROJECT/reqs.md` (v2).

---

## 2. Stage 1: The Original Idea and Audit

### 2.1 Where the idea came from

The user (LUMS, semester research project) read PAPO and noticed a gap: PAPO categorises 200 manually annotated math/STEM errors at the *output* level, concludes that ~67% are "perception failures," and proposes a training-loss change (Implicit Perception Loss + Double Entropy stabilizer). It never probes hidden states. That meant the diagnosis — "visual encoding is deficient" — was being inferred from output behaviour alone, and an obvious alternative ("encoding is fine, decoding fails") was untested.

A colleague had already attempted a probing-style investigation using Qwen2.5-VL-3B on HallusionBench, with **Grounding DINO + atomic chain-of-thought** as the verification pipeline. The colleague's word for the outcome was that results were "not very promising." The user wanted to redo the work rigorously, with every methodological choice grounded in published literature.

### 2.2 The six-agent literature audit

Six parallel research agents were dispatched to audit the colleague's original `reqs.md` line by line. Their consolidated findings live in `/Users/hasnainsohail/Documents/Computer/LUMS/PROJECT/research_validation.md`. The audit produced both negative findings (things that did not hold up) and positive findings (real and stronger evidence for the project's core claim).

**Citations that held up (real, used correctly):**
- PAPO itself (arXiv:2507.06448, ICLR 2026) — real, but does not evaluate on HallusionBench.
- HALP (arXiv:2603.05465, EACL 2026) — real, and directly relevant: probes pre-generation internal VLM representations and finds for Qwen2.5-VL specifically that *visual-only mid-layer features* (not LLM decoder features) carry the strongest hallucination signal (~0.79 AUROC).
- FaithSCAN (arXiv:2601.00269) — real, with a more precise description (three concurrently extracted signal sources, not staged decomposition).
- VIB-Probe (arXiv:2601.05547) — real.
- Activation Steering Decoding (DOI 10.18653/v1/2025.acl-long.634) — real.

**Citations that were factually wrong:**
- The "67% failure on HallusionBench" attribution to PAPO was a factual error. The 67% figure comes from manual annotation on Geometry3K, MMK12, LogicVista and MathVerse — PAPO never used HallusionBench at all.
- The architecture description for Qwen2.5-VL-3B in the original spec said "4096-dim across 32 layers." The actual model has **36 decoder layers at 2048-dim hidden size**, plus a 32-layer ViT at 1280-dim, plus a 2-layer MLP merger. All storage and compute estimates were re-derived.

**Citations that appear to be fabricated or seriously misattributed:**
- "Causal probing (2026)" — no paper with this title and framing was found. Replaced with **Marks & Tegmark, *Geometry of Truth*** (arXiv:2310.06824, COLM 2024), which actually provides causal-intervention evidence for linear probes via activation patching.
- "VADE (2025)" — the paper is real but is about attention-map-based hallucination detection, not OOD probe-failure warnings. The OOD-failure claim was redirected to its actual sources: **Dubanowska et al., "Representation-based Broad Hallucination Detectors Fail to Generalize OOD"** (arXiv:2509.19372, EMNLP 2025 Findings) and the stronger **"False Sense of Security"** (arXiv:2509.03888, NeurIPS 2025 Workshop), which documents 15–99 pp OOD drops in probing studies.

**Structural diagnosis of why the colleague's attempt failed (five reasons, none of which were tuning issues):**
1. Grounding DINO (arXiv:2303.05499) was trained on natural photographs (COCO, O365, RefCOCO, Visual Genome). About 40% of HallusionBench is charts, OCR, maps, and human-edited illusions — content categories Grounding DINO has no representation for.
2. Grounding DINO has no "nothing matches" class. Asked about an absent entity, it returns confident false detections (GitHub issue #84). For VLM hallucinations — where the model claims something not in the image — this *inverts* the verification signal.
3. Atomic CoT decomposition is ill-defined for free-form output from a 3B non-reasoning model. Sentence boundaries are heuristic; sub-claim granularity is noisy.
4. CoT chains from small VLMs are frequently post-hoc rationalizations rather than the actual computation path (arXiv:2503.08679, arXiv:2512.12218). Verifying steps of an unfaithful narrative analyzes noise.
5. HallusionBench is *by design* "an advanced diagnostic suite for entangled language hallucination and visual illusion." Disentangling perception from reasoning on it is fighting the benchmark's stated purpose. It also has its own built-in attribution mechanism via control pairs; any new method has to beat that to justify the complexity.
6. The most damning external fact: **CoRGI (arXiv:2508.00378)** — published in 2025 with essentially the same Grounding DINO + step-wise CoT verification pipeline on HallusionBench — was **withdrawn from arXiv on 2025-10-14** after reporting only +1.3 to +2.3 point improvements. The colleague's failure was a structurally predicted outcome.

**Preliminary evidence the project's hypothesis is real (not just plausible):**
- **"Seeing But Not Believing"** (arXiv:2510.17771): Relative Attention Per Token analysis shows VLM deep-layer attention correctly localizes visual evidence at comparable rates for correct AND incorrect outputs.
- **"The Hidden Life of Tokens"** (arXiv:2502.03628): Visually-grounded tokens not expressed in the output maintain high vocabulary rankings (~rank 5,000 of 32K) throughout generation, while language priors progressively dilute the visual signal in the residual stream.

These two papers provide the most direct preliminary evidence that the "model knows but fails to express" phenomenon is empirically real. Neither connects it to PAPO's training-loss framing or attempts a probing-based, causal test. The gap is genuinely open.

---

## 3. Stage 2: Spec Revision (`reqs.md` v2)

Following the audit, the user rewrote `reqs.md` end-to-end. The revisions are catalogued explicitly in Section 8 of that file ("Differences from Earlier Draft"); the most consequential decisions:

1. **Benchmark: HallusionBench → BLINK (primary) + VisRes Bench (OOD).** BLINK (Fu et al., arXiv:2404.12390, ECCV 2024) is 3,807 multiple-choice questions across 14 classic computer-vision tasks (depth, correspondence, multi-view, forensics, reflectance, art style, etc.), explicitly constructed so that "Caption + GPT-4" achieves random performance on every task. The structural guarantee — caption-mediated reasoning cannot solve BLINK — means any model failure is necessarily perception-grounded. This *eliminates* the need for Grounding DINO, atomic CoT decomposition, or any post-hoc attribution mechanism. The contrastive subset for probing is directly the set of incorrectly-answered questions.
2. **Model: Qwen2.5-VL-3B → Qwen2.5-VL-7B (with 3B retained for Mac development).** The 7B variant appears on the HallusionBench leaderboard, has materially better CoT and multi-step reasoning quality, and matches HALP's reported probing setup.
3. **Architecture description corrected.** The v2 spec records the actual model: 28 LLM decoder layers at 3584-dim hidden size for 7B (or 36 layers at 2048-dim for 3B), 32 ViT layers at 1280-dim, 2-layer MLP merger.
4. **Causal evidence upgraded from "absent" to "Level 2."** The original spec treated probe accuracy alone as evidence of internal encoding. The v2 spec commits to **amnesic probing via Iterative Nullspace Projection** (Ravfogel et al. 2020; Elazar et al. 2021, TACL — arXiv:2006.00995): erase the probed direction from the residual stream during a forward pass, then measure downstream BLINK accuracy versus an untouched model AND versus random nullspace projection of the same rank. If targeted erasure degrades accuracy and random-rank erasure does not, the model *uses* the encoded direction. Activation-patching (Level 3) is acknowledged but out of scope for one semester.
5. **Baseline suite made explicit (eight non-optional baselines).** Listed in `reqs.md` §3.6: majority class, layer-0 probe, randomly-initialized-model probe, random-labels selectivity (Hewitt & Liang 2019, arXiv:1909.03368), final-token vocabulary one-hot, logit-based uncertainty (SAPLMA, arXiv:2304.13734), difference-in-means probe (Marks & Tegmark), BLINK-task-type one-hot. Their absence is grounds for top-venue rejection per the probing methodology literature.
6. **OOD evaluation made mandatory.** Probe trained on BLINK; tested on VisRes Bench Level 1/2/3 (arXiv:2512.21194). Held-out BLINK task types are a secondary OOD split.

The v2 abstract is now contrastive and honest about what each outcome would mean — including framing the null result as publishable (it would *support* PAPO's premise, which is itself a contribution given the recent attention-based and rank-based claims to the contrary).

---

## 4. Stage 3: Pilot Implementation

### 4.1 Discipline

The user explicitly insisted on a **venv at `pilot/env/`**, not a global install. All work lives in `/Users/hasnainsohail/Documents/Computer/LUMS/PROJECT/pilot/`. The folder structure (every file a single responsibility, every knob centralised in `config.py`) is recorded in `/Users/hasnainsohail/Documents/Computer/LUMS/PROJECT/pilot/plan.md`.

Files as they stand:

| File | Role |
|---|---|
| `pilot/config.py` | Single source of truth: model name, device auto-detection (CUDA > MPS > CPU), dtype, BLINK task list, sample count, output paths. |
| `pilot/data_loader.py` | BLINK loader from HuggingFace. Single-task generator (`load_blink_samples`) and multi-task generator (`load_blink_multi`). Gathers `image_1` through `image_4` per example. |
| `pilot/model_setup.py` | Loads Qwen2.5-VL + processor; registers per-decoder-layer forward hooks; runs a single inference. Hook removal is in a `try/finally` to avoid the standard "leaked-hook memory accumulator" footgun. |
| `pilot/answer_parser.py` | Robust free-text → multiple-choice letter parser. Three strategies (letter-at-start, option-text-at-start, option-text-in-head). Includes self-tests at bottom. |
| `pilot/pilot.py` | The original 5-sample smoke pilot. Saves one combined `.npz`. |
| `pilot/run_inference.py` | Phase 1 multi-task runner with per-sample `.npz` checkpointing and full resumability. |
| `pilot/smoke_test.py` | No-download fast check (imports, config, BLINK loader, model_setup importability). |
| `pilot/requirements.txt` | torch>=2.2, torchvision, transformers>=4.49, accelerate, datasets, Pillow, numpy, scikit-learn, qwen-vl-utils. |

### 4.2 The 5-sample smoke pilot

`pilot.py` was implemented per `plan.md` and run on Qwen2.5-VL-3B-Instruct (Mac, MPS, fp16) on five `Spatial_Relation` samples. It worked end-to-end: model loaded, hooks attached to all 36 decoder layers, inference generated text, hidden states extracted at shape `(36, 2048)` per sample, combined output saved to `outputs/pilot_hidden_states.npz`.

### 4.3 Real bug discovered: BLINK schema mismatch

The `plan.md` template assumed BLINK's `choices` field contained letter-prefixed strings like `["A. left", "B. right"]`. The actual schema (verified against the live HuggingFace dataset) is:

- `choices` is a list of bare option texts: `["yes", "no"]` — no letter prefixes.
- `answer` is formatted `"(B)"` (parenthesised letter).

`data_loader.py` was corrected: strip the parens from `answer`, and construct the letter→text mapping by enumerating `choices` against `string.ascii_uppercase`. This bug only surfaced when checking the parsed output letters against gold, because the original prefix-splitting code silently produced wrong dictionary keys on the real schema. Captured in `data_loader.py` lines 5–8 as a schema note.

---

## 5. Stage 4: Phase 1 Inference

### 5.1 Scope expansion

After the smoke pilot, the goal was a full single-pass run over BLINK's validation split with all 14 task types, with hidden-state extraction and per-sample checkpointing. The implementation lives in `run_inference.py`. Three substantive changes from the pilot:

1. **Multi-image support.** Several BLINK tasks present 2–4 images per question (e.g. `Multi-view_Reasoning`, `Jigsaw`, `Visual_Similarity`, `Visual_Correspondence`). The pilot's single-image path was insufficient. `data_loader.py::_gather_images` collects every non-None `image_N` field; `model_setup.py::run_single` accepts a list of PIL images and builds the chat content with one image block per image followed by the text block.
2. **Constrained letter-only prompt.** `model_setup.py::_format_question` builds prompts of the form:
   ```
   You are shown N image(s).
   <question>

   Options:
   A. <text>
   B. <text>
   ...

   Answer with the letter only.
   ```
   `max_new_tokens=16` when options are supplied (vs 64 free-form). This keeps answer parsing trivial and reduces generation noise. Hidden-state extraction at the last token is unaffected by prompt format.
3. **Per-sample `.npz` checkpointing.** `run_inference.py::save_sample` writes each sample to `outputs/inference/{task}/{idx:04d}.npz` with fields: `hidden_states (num_layers, hidden_dim) float32`, `task`, `idx`, `question`, `options` (JSON string), `gold`, `generated`, `parsed`, `correct`. `already_done()` checks for existing files, so restarts skip completed samples. Per-sample exceptions are logged and counted but do not abort the run.

### 5.2 The MPS OOM and the fix

First multi-image attempt crashed with an MPS allocation request near 13 GB on a 16 GB Mac. Root cause: by default `AutoProcessor` for Qwen2.5-VL uses `max_pixels = 1280*28*28` (~1M visual pixels) *per image*; a 4-image example asks the model to process ~4M visual pixels, each becoming a vision token after the merger. Fix in `model_setup.py::load_model_and_processor`:

```python
processor = AutoProcessor.from_pretrained(
    config.MODEL_NAME,
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28,
)
```

This caps any single image at ~200K pixels, so even a 4-image example stays under ~800K total. With this in place the run completed with zero OOMs.

### 5.3 Full run results

Full BLINK validation split (1901 samples across all 14 tasks) was run on Mac with Qwen2.5-VL-3B in fp16 on MPS. Wall time approximately 57 minutes (3428 s elapsed across the largest segment per `outputs/full_run.log`, ~1.8 s/sample average), **zero runtime errors**.

Top-line numbers, computed from the saved `.npz` files (not the log — these are authoritative):

- Total samples: **1901**
- Parsed: **1900 / 1901** (one unparseable output)
- Correct (of parsed): **911 / 1900 = 47.9%**
- **Wrong-output contrastive subset: 989 samples**

Per-task breakdown (N = saved samples; parsed = answer parser succeeded; acc% = correct / parsed):

| Task                       |   N | parsed | correct |  acc% | wrong |
|---                         |---:|---:|---:|---:|---:|
| Art_Style                  | 117 | 117 |  60 | 51.3 |  57 |
| Counting                   | 120 | 120 |  79 | 65.8 |  41 |
| Forensic_Detection         | 132 | 132 |  41 | 31.1 |  91 |
| Functional_Correspondence  | 130 | 130 |  29 | 22.3 | 101 |
| IQ_Test                    | 150 | 150 |  39 | 26.0 | 111 |
| Jigsaw                     | 150 | 150 |  70 | 46.7 |  80 |
| Multi-view_Reasoning       | 133 | 133 |  63 | 47.4 |  70 |
| Object_Localization        | 122 | 122 |  65 | 53.3 |  57 |
| Relative_Depth             | 124 | 124 |  88 | 71.0 |  36 |
| Relative_Reflectance       | 134 | 134 |  57 | 42.5 |  77 |
| Semantic_Correspondence    | 139 | 139 |  49 | 35.3 |  90 |
| Spatial_Relation           | 143 | 143 | 123 | 86.0 |  20 |
| Visual_Correspondence      | 172 | 171 |  67 | 39.2 | 104 |
| Visual_Similarity          | 135 | 135 |  81 | 60.0 |  54 |

Hidden-state shape per sample: `(36, 2048)` — exactly the Qwen2.5-VL-3B LLM-decoder geometry the corrected `reqs.md` describes. This is the "last LLM-decoder-token hidden state at each of 36 layers," not yet the ViT or merger states (those will be added when we move to the 7B model on the cluster).

The 47.9% overall accuracy yields ~989 wrong vs ~911 correct — close to a 1:1 class balance for the contrastive probe — which is the ideal regime. The risk note in `reqs.md` Section 5 ("Small contrastive subset") is mitigated by these numbers.

---

## 6. Current State

### 6.1 What exists on disk

```
/Users/hasnainsohail/Documents/Computer/LUMS/PROJECT/
├── reqs.md                 v2 research spec
├── research_validation.md  six-agent audit report
├── JOURNEY.md              this document
└── pilot/
    ├── env/                venv (gitignored)
    ├── plan.md             pilot implementation plan
    ├── README.md
    ├── requirements.txt
    ├── config.py           single source of truth
    ├── data_loader.py      BLINK loader, multi-image
    ├── model_setup.py      Qwen2.5-VL + hooks (with max_pixels cap)
    ├── answer_parser.py    free-text → letter, with self-tests
    ├── pilot.py            5-sample smoke entry point
    ├── run_inference.py    Phase 1 multi-task runner
    ├── smoke_test.py
    └── outputs/
        ├── pilot_hidden_states.npz       5-sample combined npz from pilot.py
        ├── full_run.log                  Phase 1 run log (3767 lines)
        └── inference/
            ├── Art_Style/                117 .npz files
            ├── Counting/                 120
            ├── Forensic_Detection/       132
            ├── Functional_Correspondence/130
            ├── IQ_Test/                  150
            ├── Jigsaw/                   150
            ├── Multi-view_Reasoning/     133
            ├── Object_Localization/      122
            ├── Relative_Depth/           124
            ├── Relative_Reflectance/     134
            ├── Semantic_Correspondence/  139
            ├── Spatial_Relation/         143
            ├── Visual_Correspondence/    172
            └── Visual_Similarity/        135
```

Total per-sample shards: **1901**. Each holds a `(36, 2048)` hidden-state tensor plus the question, options, gold letter, generated text, parsed letter, and correctness label.

### 6.2 The contrastive subset

The 989 wrong-output samples are the headline dataset for the first probe. Class balance against the 911 correct samples is ~52/48 — close enough to ignore class-weight adjustments at the first pass. Per-task wrong counts range from 20 (Spatial_Relation) to 111 (IQ_Test); training a single per-layer probe over all tasks pooled is the right first move, with per-task held-out probes as the secondary OOD analysis (one of the analyses already named in `reqs.md` §3.8).

### 6.3 What is running / what comes next

The user has just authorised starting the next phase **in parallel with the writing of this document**. The immediate next steps are:

1. Collate per-sample `.npz` files into one consolidated dataset (one matrix per layer, plus a metadata frame).
2. Implement the first **linear probe**: L2-regularized logistic regression, per-layer, predicting `gold_letter` (the correct multiple-choice option) from hidden states on the wrong-output subset.
3. Get a first signal: does any of the 36 layers exhibit above-chance probe accuracy on the wrong-output subset? Specifically, above the majority-class floor and above the layer-0 baseline.

The full baseline suite (8 baselines per `reqs.md` §3.6), selectivity controls, ViT/merger extraction, amnesic probing, and the move to the 7B model on a cluster — these are all explicitly downstream of getting first signal. The pilot's stated deliverable was "a saved `.npz` of `(N, num_layers, hidden_dim)` hidden states plus generated answers and gold labels"; that deliverable now exists at 1901-sample scale.

---

## 7. Decisions Pending

These are choices the user might want to revisit when next picking the project up. None blocks the first probe; all matter for the final paper.

1. **Qwen2.5-VL-3B for Mac development vs. 7B for the actual study.** All inference so far is on 3B at 36 layers × 2048-dim. The v2 `reqs.md` commits to 7B (28 layers × 3584-dim) for the published study; this is the model HALP's probing accuracies are reported against, and it is the model on the relevant leaderboards. We will need cluster access (single A100/H100, ~14 GB GPU memory in fp16, ~12 h wall time for full BLINK + extraction) to do the 7B run. The 3B run remains useful: as a development baseline, as a smaller cross-scale comparison, and as a sanity check that the pipeline is correct before consuming cluster budget.
2. **Spatial_Relation has only 20 wrong-output samples.** That task is too easy for Qwen2.5-VL-3B (86% accuracy). If any per-task analysis is reported, Spatial_Relation will be unreliable; consider pooling it with a neighboring task or excluding it from per-task held-out OOD splits. (For pooled probing over all tasks this is not an issue — the 20 samples just contribute few wrong examples.)
3. **The single unparseable sample (1 of 1901).** Worth a one-line inspection to confirm it is a degenerate output and not a bug in `answer_parser.py`. Sample is in `outputs/inference/Visual_Correspondence/`.
4. **ViT and merger hidden states are not yet extracted.** Current hooks only capture the LLM decoder. HALP's finding — that visual-only mid-layer ViT features carry the strongest grounding signal for Qwen2.5-VL — is part of the v2 spec but requires adding hooks on the ViT layers `{7, 15, 23, 31}` and on the merger output. To do once first LLM-decoder probe results are in hand; not before (we want to know what we are looking for before adding storage cost).
5. **Prompt format.** The current constrained letter-only prompt is good for parsing accuracy, but the hidden state captured is "the last token of the input + chat-template suffix." That is *before* generation begins — i.e. it captures the model's representation right at the answer-prediction point. This is consistent with INSIDE / "Knowing Before Saying" / "Reasoning Models Know When They're Right" — all of which probe pre-generation or last-token-of-chunk states. Worth being explicit about this in the paper so reviewers do not assume mid-generation extraction.
6. **Run reproducibility.** No explicit random seed is set anywhere (decoding is greedy `do_sample=False`, so the model output is deterministic, but worth pinning torch seeds before probe training for the eventual 5-seed mean ± SD reporting in `reqs.md` §3.9).
7. **The user's compute budget for the full 7B + VisRes Bench + amnesic probing pipeline** is not yet quantified. The 14-week timeline in `reqs.md` §6 assumes one A100 is available from week 3 onwards. This should be confirmed with the advisor before week 7 (when probing experiments begin in that schedule).

---

## 8. Glossary of Real Citations

Every paper actually used or planned for use, with arXiv ID / DOI. Status reflects the literature audit; "real" means existence and content verified.

**The paper this project pushes back on**

- **PAPO** — Wang et al., "Perception-Aware Policy Optimization for Multimodal Reasoning," arXiv:2507.06448, ICLR 2026 Poster. Implicit Perception Loss + Double Entropy stabilizer for GRPO/DAPO training. Real. 67% perception-failure figure comes from Geometry3K / MMK12 / LogicVista / MathVerse — not HallusionBench.

**Direct preliminary evidence the project's hypothesis is real**

- **"Seeing But Not Believing"** — arXiv:2510.17771, 2025. Relative Attention Per Token shows deep-layer attention localizes correctly even for incorrect outputs.
- **"The Hidden Life of Tokens"** — Lin et al., arXiv:2502.03628, 2025. Visually-grounded tokens not expressed maintain high vocabulary rankings; language priors progressively dilute visual signal in the residual stream.

**Methodology foundations**

- **Alain & Bengio**, arXiv:1610.01644, 2017. Linear probes on frozen intermediate representations — the founding methodology.
- **Hewitt & Liang**, "Designing and Interpreting Probes with Control Tasks," arXiv:1909.03368, EMNLP 2019. Selectivity / random-labels control — mandatory probe-memorization check.
- **Belinkov**, "Probing Classifiers: Promises, Shortcomings, and Advances," Computational Linguistics 48(1), arXiv:2102.12452, 2022. Survey of dataset-artifact, label-leakage, and complexity confounds.
- **Pimentel et al.**, "Information-Theoretic Probing for Linguistic Structure," arXiv:2004.03061, ACL 2020. Probe accuracy as mutual-information estimate; resolves probe-complexity debate.

**LLM internal-state probing for truthfulness / hallucination**

- **SAPLMA** — Azaria & Mitchell, arXiv:2304.13734, EMNLP Findings 2023. Hidden-state classifiers detect truth/falsity at 71–83%.
- **INSIDE** — Chen et al., arXiv:2402.03744, ICLR 2024. Middle-layer last-token embeddings dominate for hallucination detection.
- **Marks & Tegmark, "Geometry of Truth"** — arXiv:2310.06824, COLM 2024. Difference-in-means probes + causal interventions; LLM truthfulness representations are linear and causally implicated. Replaces the fabricated "Causal probing (2026)" citation.
- **Semantic Entropy Probes** — Farquhar et al., arXiv:2406.15927, 2024. Uncertainty-targeted probes generalize OOD better than direct-correctness probes.
- **"Knowing Before Saying"** — arXiv:2505.24362, 2025. Probes last-token of CoT chunks for downstream success prediction.
- **"Reasoning Models Know When They're Right"** — arXiv:2504.05419, 2025. Closest experimental-design precedent for this project; probes Qwen-family models at last-token-of-chunk hidden states.

**VLM internal-state probing**

- **HALP** — arXiv:2603.05465, EACL 2026. Probes three families of internal VLM representations; for Qwen2.5-VL specifically, visual-only mid-layer ViT features (≈0.79 AUROC) outperform late-fusion features.
- **"Towards Interpreting Visual Information Processing in VLMs"** — arXiv:2410.07149, 2024. Logit-lens analysis on LLaVA; middle-to-late layers (~25 of 33) show peak visual grounding; removing object-specific patch tokens drops accuracy 70–96%.
- **FaithSCAN** — arXiv:2601.00269, 2026. Three signal sources (visual perception, cross-modal interaction, language decoding) extracted in single pass.
- **VIB-Probe** — arXiv:2601.05547, 2026. VIB-based filtering of hidden states for VLM hallucination detection.

**Causal validation (amnesic probing)**

- **INLP — Ravfogel et al.**, "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection," ACL 2020. The nullspace projection algorithm itself.
- **Elazar et al.**, "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals," arXiv:2006.00995, TACL 2021. The Level-2 causal probing methodology this project uses.
- **Meng et al., ROME**, NeurIPS 2022. Activation-patching — referenced as the Level-3 target that is out of scope this semester.

**OOD failure warnings (mandatory for any new probing study)**

- **"False Sense of Security"** — arXiv:2509.03888, NeurIPS 2025 Mech Interp Workshop. LLM/VLM probes drop 15–99 pp OOD; some configurations reach near-zero on semantically equivalent paraphrases.
- **Dubanowska et al.**, "Representation-based Broad Hallucination Detectors Fail to Generalize OOD," arXiv:2509.19372, EMNLP 2025 Findings. Complementary evidence specifically for hallucination probes. Replaces the misattributed VADE citation.

**Benchmarks**

- **BLINK** — Fu et al., "BLINK: Multimodal Large Language Models Can See but Not Perceive," arXiv:2404.12390, ECCV 2024. 3,807 multiple-choice questions over 14 perception tasks; caption + LLM achieves random performance by construction.
- **VisRes Bench** — arXiv:2512.21194. Three-level perception-to-reasoning hierarchy; procedurally generated; the OOD evaluation target for this project.
- **HallusionBench** — Guan et al., arXiv:2310.14566, CVPR 2024. The original benchmark choice; replaced because of perception/reasoning entanglement and detector mismatch.

**Things that did not survive the audit (kept here so they aren't accidentally cited)**

- "Causal probing (2026)" — not verified; appears fabricated. Replaced with Marks & Tegmark 2023.
- "VADE warns probes fail OOD" — VADE is real but is about attention-map-based hallucination detection. The OOD-failure warning citation should be Dubanowska et al. or "False Sense of Security."
- "PAPO reports 67% perception failures on HallusionBench" — factual error. PAPO never used HallusionBench.
- **CoRGI** — arXiv:2508.00378. Real, but **withdrawn from arXiv 2025-10-14**. The structurally equivalent design (Grounding DINO + step-wise CoT on HallusionBench) was independently confirmed not to work. The colleague's attempt was a parallel instance of the same architecture.
- **Grounding DINO** — Liu et al., arXiv:2303.05499, ECCV 2024. Real and well-known, but structurally unsuited to HallusionBench. Eliminated by the switch to BLINK.

---

*End of document. The next entry in this journey will live in a separate `PROBING_RESULTS.md` once the first probe runs.*
