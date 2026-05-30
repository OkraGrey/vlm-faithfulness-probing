"""Feasibility dry run: can Qwen2.5-VL-7B load + forward on THIS 24GB M4 Pro?

Not the real study (that belongs on the 48GB machine). This only answers:
  (1) does the 7B load without OOM on 24GB,
  (2) does a forward+generate produce hidden states of the expected (28, 3584) shape,
  (3) how much memory and time it costs.

Conservative: 2 single-image samples, MPS high-watermark disabled so the allocator
may spill to system memory instead of hard-failing early.
"""
import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # allow full system mem / swap
import time
import resource
import traceback

import torch
import config
config.MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"   # override (not persisted to config.py)

from model_setup import load_model_and_processor, run_single
from data_loader import load_blink_samples


def rss_gb():
    # macOS ru_maxrss is in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def mps_gb():
    try:
        return torch.mps.current_allocated_memory() / 1e9
    except Exception:
        return float("nan")


def main():
    print(f"=== 7B dry run on 24GB M4 Pro (watermark={os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']}) ===")
    t0 = time.time()
    try:
        model, processor = load_model_and_processor()
    except Exception:
        print("LOAD FAILED (likely OOM):\n" + traceback.format_exc())
        return
    print(f"LOADED in {time.time()-t0:.0f}s  RSS={rss_gb():.1f}GB  mps_alloc={mps_gb():.1f}GB")

    samples = list(load_blink_samples(task="Counting", n=2, split="val"))  # single-image task
    ok = 0
    for i, (images, q, opts, ans) in enumerate(samples):
        t1 = time.time()
        try:
            gen, hidden = run_single(model, processor, images, q, options=opts)
        except Exception:
            print(f"  sample {i} FAILED:\n" + traceback.format_exc())
            continue
        layers = sorted(hidden.keys())
        n_layers, dim = len(layers), hidden[layers[0]].shape[-1]
        ok += 1
        print(f"  sample {i}: imgs={len(images)} gen={gen!r:<12} "
              f"hidden=({n_layers} layers, {dim} dim) {time.time()-t1:.1f}s "
              f"mps_alloc={mps_gb():.1f}GB RSS={rss_gb():.1f}GB")

    print(f"\nDRY RUN {'OK' if ok else 'FAILED'}: {ok}/{len(samples)} samples ran. "
          f"total {time.time()-t0:.0f}s, peak RSS={rss_gb():.1f}GB")
    print("Expected 7B geometry: (28 layers, 3584 dim). 3B was (36, 2048).")


if __name__ == "__main__":
    main()
