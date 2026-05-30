"""G3 construct-validity: noise-image inference for the GOLD-reading candidate tasks.

Re-runs Qwen2.5-VL on the SAME samples as Phase 1, but replaces each image with
uniform RGB noise of the SAME size and count. The text prompt and vision-token
structure are held identical; only the image *content* changes. We then ask
(in gate3_construct.py) whether the probe can still recover the gold answer from
these noise-image hidden states on the wrong-output subset:

  - gold-recovery collapses to chance  -> the signal required the real image -> VISUAL (H1)
  - gold-recovery persists              -> the signal was text/format artifact  -> H1 dies

Scope: only the three GOLD-reading candidates from Gate 0 (Jigsaw, Multi-view_Reasoning,
Art_Style). Resumable: skips samples already saved to outputs/inference_noise/.

Run: ./env/bin/python run_inference_noise.py
"""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import config
from data_loader import load_blink_multi
from model_setup import load_model_and_processor, run_single
from answer_parser import parse_answer, is_correct

# Default = the 3B GOLD-reading candidates. Override per model:
#   export CANDIDATE_TASKS="Jigsaw,Multi-view_Reasoning,Art_Style"
CANDIDATE_TASKS = os.environ.get(
    "CANDIDATE_TASKS", "Jigsaw,Multi-view_Reasoning,Art_Style").split(",")
NOISE_DIR = config.OUTPUTS_DIR / "inference_noise"
NOISE_SEED = 12345


def noise_like(img: Image.Image, rng) -> Image.Image:
    """Uniform RGB noise the same size as `img` (preserves vision-token count)."""
    w, h = img.size
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def sample_path(task: str, idx: int) -> Path:
    return NOISE_DIR / task / f"{idx:04d}.npz"


def main():
    print("=== G3: noise-image inference (construct validity) ===")
    print(f"Tasks: {CANDIDATE_TASKS}")
    print(f"Output: {NOISE_DIR}\n")
    rng = np.random.default_rng(NOISE_SEED)

    model, processor = load_model_and_processor()

    n_done = n_skip = n_err = 0
    t0 = time.time()
    for task, idx, images, question, options, gold in load_blink_multi(tasks=CANDIDATE_TASKS):
        out_path = sample_path(task, idx)
        if out_path.exists():
            n_skip += 1
            continue
        noise_imgs = [noise_like(im, rng) for im in images]
        try:
            generated, hidden = run_single(model, processor, noise_imgs, question, options=options)
        except Exception:
            n_err += 1
            print(f"  ERROR {task}[{idx}]: {traceback.format_exc().splitlines()[-1]}")
            continue

        parsed = parse_answer(generated, options)
        layer_idx = sorted(hidden.keys())
        import torch
        stacked = torch.stack([hidden[k] for k in layer_idx], dim=1).squeeze(0).numpy()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path, hidden_states=stacked, task=task, idx=idx,
            question=question, options=json.dumps(options), gold=gold,
            generated=generated, parsed=parsed if parsed is not None else "",
            correct=bool(is_correct(parsed, gold)),
        )
        n_done += 1
        if n_done % 25 == 0:
            print(f"  [{n_done}] {task}[{idx}] noise-out={parsed!r} gold={gold}")

    dt = time.time() - t0
    print(f"\nDone. new={n_done} skipped={n_skip} errors={n_err} elapsed={dt:.0f}s")


if __name__ == "__main__":
    main()
