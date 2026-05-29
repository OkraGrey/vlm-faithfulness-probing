# Proposal: Vision Ablation as the Next Experiment

> **Purpose**: My case for why vision ablation should come before Gate 2 (random-init baseline) in the canonical methodology sequence. Includes the reasoning, the experiment design, the failure modes, and what each outcome would mean.

## The core question this experiment answers

Construct validity. Specifically:

**When our probe predicts the gold answer letter from hidden states on the wrong-output subset, is it reading "the model encoded correct visual content" or "the model encoded answer-relevant information from the question text alone"?**

These are different findings with different implications. PAPO's claim is about *perception* failures — the model "doesn't see correctly." The project's hypothesis pushes back: the model "sees" correctly but doesn't express it. For that pushback to mean anything, the probe signal we find must be specifically about *seeing*.

If the probe works equally well without the image, we haven't shown anything about vision. We've shown that hidden states encode answer-relevant features, full stop — which is interesting but doesn't engage PAPO at all.

## Why this isn't covered by the canonical gates

| Gate | What it rules out | What it doesn't tell us |
|---|---|---|
| Gate 1 selectivity (Hewitt & Liang) | Probe memorization | Whether the signal is about vision |
| Gate 2 random-init baseline (Alain & Bengio) | Architecture-only signal | Whether the *trained* signal is about vision |
| Gate 4 OOD evaluation | Dataset-specific shortcut | Whether the signal in either domain is about vision |
| Gate 5 amnesic probing (Elazar et al.) | Property is encoded but not used | Whether the property is visual at all |

None of the canonical gates separate "model uses vision" from "model uses question structure." They all assume that question is settled. For a project whose entire framing is "the model encodes correct *visual* grounding," that assumption needs its own experiment.

The closest analog in the literature is BLINK's own design principle: the benchmark was constructed so that **caption + LLM = random** on every task — i.e., the questions are unsolvable without the image. But that's a property of the *benchmark*, established by the benchmark authors, not a property of *our model's hidden states*. We need to verify the property holds *inside our specific model* for our probing to mean what we claim.

## The experiment design

Same exact pipeline, two conditions per question:

**Condition A — image+text** (already done):
```
chat content = [{type: "image", image: img}, {type: "text", text: question}]
→ run model, extract hidden states
```

**Condition B — text-only** (new):
```
chat content = [{type: "text", text: question}]   # no image at all
→ run model, extract hidden states
```

Both conditions are run on the same 1,901 BLINK examples. For each, we train identical probes (logistic regression + Gate 1 baselines) to predict the gold answer letter. Comparison metric:

```
visual_attribution = probe_acc_image+text  -  probe_acc_text_only
```

A large `visual_attribution` (>0.15) means the probe's signal genuinely requires the image. A small one (<0.05) means our signal isn't visual.

## Two alternative formulations of text-only

There's a design choice here that matters:

**Option 1 — Question-only (no image at all)**. Cleanest baseline. The model sees just the question text. Hidden states reflect only what the question says.

**Option 2 — Question + blank/noise image**. Keeps the multi-modal pipeline structure intact (some VLMs behave differently when the image slot is missing entirely). Image is a black square or random noise. Tests whether the *content* of the image carries the signal, not its mere presence.

Both are valuable. I recommend running Option 2 (blank image) as the primary because:
- It preserves the prompt structure exactly (no chat-template differences)
- The model still sees an "image" — so any architectural artifact of image-token processing remains constant
- The only thing that varies is *what* the image content is

Option 1 (no image) is worth running as a secondary check for completeness.

## Compute and engineering cost

Almost identical to existing inference run:

- Same 1,901 questions × 1 inference each = 1,901 inferences
- 1-image input regardless of how many images BLINK normally provides for that task (multi-image tasks become 1-image-blank tasks)
- At ~1.8s per single-image sample (the speed we measured), total ≈ **57 min** on Mac MPS
- Code changes: minimal. `run_inference.py` already accepts an images list; we just substitute a blank PIL image.

```python
# pseudocode
blank = PIL.Image.new("RGB", (224, 224), color=(0, 0, 0))
for sample in dataset:
    generated, hidden = run_single(model, processor, [blank], sample.question, sample.options)
    save_to outputs/inference_blank/{task}/{idx:04d}.npz
```

Then we collate to `dataset_blank.npz` and re-run `baselines.py` pointing at the new dataset.

Total wall-clock: about 90 min including code changes.

## Decision criteria after this experiment

For each of the 2-5 tasks that passed Gate 1 (or were borderline):

| Outcome | What it means | Where the project goes |
|---|---|---|
| `image-text` probe ≫ `blank-image` probe (gap > 0.15) | Signal is genuinely visual. Hypothesis confirmed for that task. | Continue to Gates 2-5 with confidence. |
| `image-text` probe ≈ `blank-image` probe (gap < 0.05) | Signal is from question structure / task semantics, not vision. | Hypothesis dies for that task. Reframe: "VLM hidden states encode answer-relevant info without using vision." |
| Mixed across tasks | Some tasks visual, others not. | Cleanest story: focus paper on the tasks where attribution is clearly visual. |

Note: even the "hypothesis dies" outcome is publishable. It's a substantive negative result against PAPO's premise — PAPO assumed perception failure causes wrong outputs; if probes succeed without vision, the model isn't failing at perception, it's failing somewhere else entirely. Different finding, equally valid.

## Why I'm proposing this over Gate 2

Gate 2 (random-init baseline) answers: "does *training* matter?" If random-init gets the same probe accuracy, we learn nothing from training. If trained does better, we confirm training adds value.

But Gate 2 doesn't answer the harder question: "does *vision* matter?" Trained Qwen2.5-VL could outperform random Qwen2.5-VL purely because trained models learn to use question semantics, attention to question tokens, etc. — none of which require the model to actually look at the image.

Gate 2 strengthens an existing claim. Vision ablation tests whether the claim is even the right kind of claim.

**Methodologically, the ordering "are we measuring the right thing?" before "is the measurement robust?" is more fundamental.** If we run Gate 2 first and the result is positive (trained model adds value), we still don't know if it's visual value. If we run vision ablation first and the result is positive (image matters), Gate 2 then strengthens a claim we know is the right one.

## Risk: what if I'm wrong

If you push back and say Gate 2 should come first, the case for that is:

- Canonical-pipeline adherence: Belinkov 2022 lists random-init baseline before discussing construct validity. Some reviewers expect that order.
- Gate 2 result is binary and crisp; vision ablation has more interpretive nuance.
- BLINK's design (caption + LLM = random) is *some* evidence the signal is visual, even before we test internally.

I find these arguments weaker than the construct-validity argument, but they're not wrong. If you prefer to follow the canonical sequence strictly, that's a defensible call.

## Concrete plan if you greenlight this

1. Add a `blank_image_pilot.py` or extend `run_inference.py` with a `--blank-image` flag.
2. Run inference on 1,901 BLINK questions with blank images. Save to `outputs/inference_blank/`.
3. Add `collate.py --variant blank` to produce `dataset_blank.npz`.
4. Run `baselines.py --variant blank` on the new dataset. Get parallel Gate 1 results.
5. Compute per-task `visual_attribution = real_acc_wrong − blank_acc_wrong` for all tasks.
6. Append results to `03_methodology_log.md`.
7. Update `00_start_here.md` with the next decision.

End-to-end: ~3 hours including code and analysis.

## Citations that motivate this experiment

- **BLINK** (Fu et al., arXiv:2404.12390, ECCV 2024) — establishes that caption + LLM = random on every task; the benchmark-level construct-validity guarantee we're extending to model-level.
- **Belinkov 2022** ("Probing Classifiers: Promises, Shortcomings, and Advances") — §4 discusses construct validity in probing as a category beyond memorization controls.
- **Ravichander et al. 2021** ("Probing the Probing Paradigm") — empirical evidence that high probe accuracy doesn't entail what the researcher claims it entails.
- **Lipton 2018** ("The Mythos of Model Interpretability") — argues that interpretability claims must specify what cognitive/computational property is being identified, not just that *some* signal exists.

## Claude's note

This is the experiment I most want to run because its result genuinely could go either way. I would not be surprised if text-only probes recover the gold letter at non-trivial accuracy on tasks like Object_Localization — VLMs have been shown to over-rely on language priors, and the question "Which bounding box more accurately localizes X?" with options "Box A / Box B" carries some prior bias even without the image. If that's the case, our 0.781 wrong-acc result loses interpretation as a visual-grounding finding.

Conversely, on Forensic_Detection ("Which image is most likely a real photograph?"), I'd expect text-only to collapse hard. There's no language prior over which forensic image is the real one without seeing them.

Either result is informative. That's why I'd run this before going further.
