# Open Questions / Parking Lot

> **Purpose**: Flagged decisions and methodological questions that emerged during the work but don't need to be answered today. Append as new ones surface; resolve and move to `03_methodology_log.md` once decided.

## Methodological refinements (flagged during Gate 1)

### Q1. DiM criterion: too strict?

**Current criterion**: trained logreg must beat difference-in-means probe by > 0.05 on wrong-output accuracy.

**The issue**: 3 tasks (Jigsaw, Multi-view_Reasoning, Art_Style) failed Gate 1 specifically because DiM matched or exceeded logreg. But DiM achieving high accuracy is itself evidence of clean linear separability in the representation — i.e., the *signal exists*, the trained probe just isn't adding value over a mean classifier.

Hewitt & Liang's argument cuts both ways: simpler probes are PREFERRED (so DiM excelling shouldn't disqualify a task). But if logreg can't add value, what's the gain over the simplest possible probe?

**Possible reformulations**:
- `max(logreg_acc, dim_acc) − vocab_acc > 0.05` — accepts whichever probe works, requires it to beat the lexical baseline
- Report DiM as the *primary* probe and logreg as the *complementary* probe (instead of treating logreg as primary)
- Tighten L2 regularization on logreg — current `C=1.0` may under-regularize with 2048-dim features and ~80 training samples

**Status**: Flagged. Worth experimenting with after vision-ablation results come in.

### Q2. Test fraction: too small for narrow tasks?

**Current setup**: 25% test split, stratified by gold label.

**The issue**: For tasks with 20-60 wrong-output samples, 25% test = 5-15 wrong-test samples. CIs are huge ([0.00, 0.80] for Relative_Depth). Single-split estimates are noise-dominated.

**Possible fixes**:
- **5-fold CV** with 5 seeds → 25 measurements per task. Cost: 5x compute on Gate 1.
- **Leave-one-out** for tasks with <30 wrong samples specifically (Spatial_Relation, Relative_Depth)
- Drop the wrong-output subset analysis for tasks where N_wrong < 30 and report all-test accuracy only

**Status**: Will likely adopt 5-fold CV before any "final" reporting. Not urgent for the next experiment.

### Q3. Best layer: best-overall or best-wrong?

**Current**: We select "best layer" by maximum mean wrong-subset accuracy across seeds.

**The issue**: A layer that's best for wrong-output samples specifically might not be the best for understanding the model in general. Some published probing work selects best layer by all-test accuracy and reports the same layer's wrong-subset performance. This avoids cherry-picking the best layer on the small wrong-test subset.

**Possible fixes**:
- Select best layer by all-test accuracy, report wrong-subset accuracy at that layer
- Report a layer profile (mean ± SD across 3-5 layers around the peak) rather than a single best layer
- Show the full layer-wise curve in the paper, not just the peak

**Status**: Defer to writing-up phase. Currently we report best-by-wrong which is what the project's hypothesis cares about.

## Project-direction questions

### Q4. Spatial_Relation's tiny wrong subset (n=20)

Spatial_Relation has 86% accuracy → only 20 wrong-output samples → ~5 wrong-test samples per seed. We can't draw meaningful conclusions about this task from the existing data. Options:

- **Drop the task** from analysis (cleanest)
- **Use full BLINK split (val + test)** to roughly double the data, accept the cost that test split is no longer held out
- **Augment with harder spatial benchmarks** (e.g., MMVP, ImageNet-Hard) to get more wrong-output samples on spatial tasks

**Status**: Lean toward "drop Spatial_Relation from per-task analysis but include in aggregate counts." Document the rationale.

### Q5. Counting at 100% in initial 5-sample test — why?

In the 15-sample multi-task verification, Counting scored 5/5 (100%). After full inference, Counting scored 65.8% (79/120). Initial 5 were probably the easy end of the distribution.

**Lesson learned**: Small-N initial signals can be very misleading. We saw this in Gate 1 too (9 tasks beat baseline first-look → 2 robust after controls). The pattern of attrition is consistent.

**Status**: Methodological note. Don't trust small-N pilot results to predict full-run results.

### Q6. The unparseable Visual_Correspondence sample

Visual_Correspondence has 171 parsed of 172 samples. One sample's output didn't parse to any letter. Worth a 5-minute investigation:

```bash
env/bin/python -c "
import numpy as np
from pathlib import Path
for fp in Path('outputs/inference/Visual_Correspondence').glob('*.npz'):
    d = np.load(fp, allow_pickle=True)
    if not str(d['parsed']):
        print(fp, repr(str(d['generated'])))
"
```

Knowing what the model output for that one sample tells us whether our parser has a bug or whether the model genuinely refused to answer.

**Status**: Low priority. Won't change Gate 1 conclusions.

## Compute / infrastructure

### Q7. Compute scaling: 3B forever, or upgrade?

Currently Qwen2.5-VL-3B on Mac MPS. The `reqs.md` v2 lists 7B as the target. The literature ([HALP], [Knowing Before Saying]) studies VLMs at 7B+ scale. 3B-only findings have lower external validity.

**Options**:
- LUMS cluster (if accessible) — talk to advisor
- Cloud GPU (Lambda Labs ~$1.50/hr, RunPod ~$2.50/hr for A100) — small budget ($20-50) gets all required experiments at 7B
- Stick with 3B — defensible if results hold, but limits the conclusions

**Status**: Genuine decision to make before any "final" experiments. Vision ablation, Gate 2, Gate 5 all benefit from being run at 7B too.

### Q8. Cross-architecture replication

For maximum external validity, replicate on LLaVA-1.6-7B or similar. Same pipeline, different model family. If the finding holds across architectures, it's a model-general phenomenon, not Qwen-specific.

**Status**: Important for the paper. Defer until amnesic probing + OOD eval are done on Qwen.

## Scientific / claim-shape questions

### Q9. What's the title of the paper actually going to be?

Three candidate framings, depending on results:

- **If hypothesis holds robustly**: "VLMs Encode Correct Visual Grounding That Their Decoders Fail To Express: A Probing Study on BLINK"
- **If hypothesis fails on vision-ablation**: "Probing-Based Evaluation of VLM Hallucination: The Vision Attribution Gap"
- **If hypothesis holds for some tasks only**: "Task-Specific Visual Grounding Encoding in Vision-Language Models"

Each is a different paper with different next experiments. Worth holding all three in mind as Gates 2-5 run.

**Status**: Pre-write the abstract for each scenario as Gate results come in. Helps with focused experiment design.

### Q10. Practical implication — what's the "so what"?

Top venues increasingly require a "so what" beyond the finding itself. For our project, candidates:

- **Decoding-time intervention**: use the probe direction to steer model outputs at inference, demonstrate improvement on wrong-output cases
- **Diagnostic tool**: probe accuracy as a real-time confidence signal for VLM outputs (closer to HALP's framing)
- **Critique of PAPO**: show that PAPO's training-loss intervention targets symptoms; suggest representation-aware alternatives

**Status**: Decide once we know which scenario (Q9) we're in. Adding a practical-implication experiment is what would push the work from Findings to main-track tier.
