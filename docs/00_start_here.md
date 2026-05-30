# Briefing — 2026-05-30 (literature refresh + Gate 0)

> **Read this first.** It supersedes the 2026-05-18 briefing below for "where we are."

**Three things happened on 2026-05-30:**

1. **Literature refresh** (full detail in `06`, survey in `07`, alternatives in `08`). The user dropped arXiv:2604.02486 ("VLMs Need Words" — *not* PAPO). The "encoded-but-not-expressed" *phenomenon* is now established in the 2025–26 literature (esp. Fu et al. 2025, arXiv:2506.08008). **Our novelty is no longer the phenomenon — it's the rigorous causal-probing method on the wrong-output subset, BLINK, Qwen2.5-VL, framed as a PAPO rebuttal.** Two methodology changes locked: construct validity via **counterfactual image-swap** (not just blank-image); causal test via **activation patching** (INLP demoted — Kumar et al. 2022).

2. **Gate 0 (new): honest re-analysis under nested CV.** Code `pilot/gate0_cv.py`. **The Gate 1 headline is retracted** — Forensic_Detection and Object_Localization do NOT survive honest layer selection (their signal was a winner's curse on 15–23 test samples; they actually read the model's *output*, not gold). Full record in `03` §"Gate 0".

3. **The corrected picture:** on the wrong-output subset the probe reads the model's *own wrong output* for **9/14 tasks (PAPO-consistent)**, and reads **gold** only for **Jigsaw, Multi-view_Reasoning, Art_Style(weak)** — the candidate "sees but can't say" tasks. **But Jigsaw/Multi-view are binary and hit ~1.0, which is suspiciously clean** — could be a non-visual artifact.

**G3 DONE (2026-05-30): H1 confirmed and construct-valid for 2 tasks.** Noise-image ablation (`pilot/run_inference_noise.py` + `gate3_construct.py`, see `03` §Gate 3): on Jigsaw and Multi-view_Reasoning, real-image gold-recovery on the wrong-output subset is 0.99 / 0.93 and **collapses to chance (~majority) under noise** → the signal is genuinely visual, not a binary/format artifact. Art_Style dropped (real signal ≈ chance). So: **the "sees but can't say" effect is REAL but task-specific (2/14); the other 9 tasks remain PAPO-consistent.**

**G4 DONE (2026-05-30): the encoded direction is causally INERT — H1's strong form is REJECTED at 3B.** DiM steering toward gold (vs random control) at the mid-band layer flips almost no wrong outputs: Jigsaw 2.5% at α=4 (vs 0% random) — inert; Multi-view 12.9% at α=4 (vs 1.4% random) — weak, high-magnitude-only. See `03` §Gate 4. Combined with the flat-from-L0 profile: the apparent "knowledge" is shallow, task-specific, and not used → **the model does NOT "see but fail to say" at 3B; result is consistent with PAPO's premise.**

**THE 3B STUDY IS COMPLETE (and gives a clean H1-negative result).** What's next: **scale to Qwen2.5-VL-7B** — runs locally on the user's 48GB M5 Pro (no cluster needed). This is the principled scale test, not a rescue attempt: a 3B may simply lack usable grounding. Decisive 7B diagnostics: (1) does the gold-signal LAYER PROFILE emerge-with-depth (genuine grounding) vs flat-from-L0 (artifact, as at 3B)? (2) is the steered causal effect larger at 7B? 7B-negative → robust H1-negative across scale; 7B-positive (emergent + causally usable) → revives H1 as the headline. Either is publishable. Prep: pipeline is ~one-line portable (config.MODEL_NAME); add ViT/merger hooks (HALP: strongest Qwen2.5-VL signal is in visual features). Lower-priority optional: 3B image-permutation control to characterize the inert signal.

---

# Morning Briefing — 2026-05-18 (end-of-day)

> **Purpose**: Pick-up-cold summary. After reading this, you should know exactly where we are, what the open decision is, and what I recommend doing next. **(Superseded by the 2026-05-30 briefing above — kept for history.)**

## Where we are

We have:
- `reqs.md` v2 — research spec corrected against literature audit. Solid foundation.
- `pilot/` — working inference + probing pipeline on Mac MPS.
- **1,901 BLINK samples** with hidden states extracted across 36 LLM decoder layers for Qwen2.5-VL-3B. Stored at `pilot/outputs/inference/`.
- **989 wrong-output contrastive samples** — the subset where the model gave a wrong answer.
- **Gate 1 results** (probe controls + selectivity, run on all 1,901 samples):
  - **2 of 14 BLINK tasks pass all four Gate 1 criteria**: Forensic_Detection (layer 27, 4-option) and Object_Localization (layer 21, binary). Both show probe accuracy clearly above majority baseline on the wrong-output subset, selectivity > 0.10, beat vocab one-hot, beat difference-in-means.
  - 3 borderline tasks (Jigsaw, Multi-view_Reasoning, Art_Style) pass selectivity but the trained probe doesn't beat the simplest baseline (DiM). The signal exists; the trained probe just isn't adding value over a mean classifier.
  - 9 tasks fail. Most of the first-look "9 tasks beat baseline" headline was probe memorization or small-N noise.

This is **honest research progress.** The literature predicts exactly this kind of attrition when controls are added (Belinkov 2022; Hewitt & Liang 2019). 2 tasks with robust signal is a real first finding.

## The open decision

The canonical next gate in the probing-methodology pipeline is **Gate 2 — random-init Qwen2.5-VL baseline** (Alain & Bengio 2017). We discussed this. It's the standard move.

But I argued **last conversation that it's not the most informative experiment right now.** The most informative experiment is the one that asks: *is the probe signal we're seeing actually about visual grounding, or about something else?*

Full reasoning in [04_next_step_vision_ablation.md](04_next_step_vision_ablation.md). Short version:

Our probe predicts the gold answer letter from hidden states. We've been calling this "evidence the VLM internally encodes correct visual grounding." But strictly, it could be:
- Visual grounding (the project's hypothesis)
- Question semantics / task structure (the model knows the answer without using vision)
- Annotator/label correlation in the data

Gate 2 (random-init) and Gate 5 (amnesic probing) don't distinguish these. **Vision ablation does.**

The proposed experiment: same pipeline, same questions, image replaced with blank/none. Train probes on those hidden states. If text-only matches image+text probe accuracy, our signal isn't visual. If text-only collapses to chance while image+text retains the +0.25 gap, we have a clean visual-grounding finding.

This is the construct-validity question (Hewitt & Liang's selectivity tested one form; this tests another).

## What I recommend doing first thing

1. Read [04_next_step_vision_ablation.md](04_next_step_vision_ablation.md). Push back if the reasoning seems off.
2. If you agree: I implement vision ablation. ~60 min compute on Mac. Result by end of day.
3. If you disagree or want the canonical path: I implement Gate 2 (random-init baseline). ~90 min compute on Mac.

Either is defensible. The vision-ablation choice is mine because it tests the *interpretation* of the result rather than its *robustness* — and a wrong interpretation makes all downstream robustness work uninterpretable.

## What you don't need to decide right now

- Compute scaling (3B → 7B). Currently doable on Mac. Can defer until later phases.
- Cross-architecture (LLaVA). Same — useful but later.
- Which workshop/venue to target. Doesn't affect today's experiment.

## Where things live

```
/Users/hasnainsohail/Documents/Computer/LUMS/PROJECT/
├── reqs.md                                # Canonical research spec (v2)
├── docs/                                  # All documentation (this folder)
│   ├── README.md                          # Doc index
│   ├── 00_start_here.md                   # This file
│   ├── 01_research_validation.md          # Literature audit
│   ├── 02_journey.md                      # Chronological project history
│   ├── 03_methodology_log.md              # Methodology decisions + Gate 1 results
│   ├── 04_next_step_vision_ablation.md    # Proposed next step
│   └── 05_open_questions.md               # Parking lot
└── pilot/                                 # Implementation
    ├── env/                               # Python 3.11 venv with all deps
    ├── *.py                               # Pipeline code
    └── outputs/                           # All results
        ├── inference/{Task}/*.npz         # 1,901 per-sample hidden states
        ├── dataset.npz                    # Collated, fast to reload
        ├── baselines_v1.json              # Gate 1 detail rows (504 rows)
        └── baselines_summary.json         # Per-task headlines + passes_gate1 flag
```

## Quick commands to re-run anything

```bash
cd /Users/hasnainsohail/Documents/Computer/LUMS/PROJECT/pilot
source env/bin/activate

python smoke_test.py             # quick environment check, no model load
python run_inference.py          # resumable inference; skips done samples
python collate.py                # re-build dataset.npz from per-sample files
python baselines.py              # re-run Gate 1 (5 seeds, 5 baselines, ~15 min)
```

## State of the conversation

We agreed last conversation that the goal is the most-rigorous-possible research, irrespective of difficulty. You said you're not literally targeting NeurIPS but you want the principled path. The proposal in `04_next_step_vision_ablation.md` reflects that — it's not the canonical-pipeline path (Gate 2 next), it's the construct-validity path (vision ablation next), which I think is methodologically the right call.

Open to being wrong about that. If you'd rather follow the canonical pipeline strictly, say so in the morning and I'll do Gate 2 instead.
