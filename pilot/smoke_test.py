"""Lightweight checks that pass without downloading the model.

Verifies: imports, config, BLINK loader, device detection.
Fails fast if anything is broken before you wait for a model download.
"""
import sys


def main():
    print("--- Smoke test ---")

    # 1. Imports
    try:
        import torch
        import transformers
        import datasets
        import numpy
        from PIL import Image  # noqa: F401
        print(f"  torch:        {torch.__version__}")
        print(f"  transformers: {transformers.__version__}")
        print(f"  datasets:     {datasets.__version__}")
        print(f"  numpy:        {numpy.__version__}")
    except Exception as e:
        print(f"  IMPORT FAILED: {e}")
        return 1

    # 2. Config loads
    try:
        import config
        print(f"  device:       {config.DEVICE} ({config.DTYPE})")
        print(f"  model:        {config.MODEL_NAME}")
    except Exception as e:
        print(f"  CONFIG FAILED: {e}")
        return 1

    # 3. BLINK loader works (downloads task subset on first run)
    try:
        from data_loader import load_blink_samples
        samples = list(load_blink_samples(n=1))
        if not samples:
            print("  BLINK FAILED: no samples returned")
            return 1
        img, q, opts, ans = samples[0]
        print(f"  blink:        loaded 1 sample, image={img.size}, answer={ans}")
    except Exception as e:
        print(f"  BLINK FAILED: {e}")
        return 1

    # 4. model_setup imports (no model download)
    try:
        import model_setup  # noqa: F401
        print(f"  model_setup:  imports cleanly")
    except Exception as e:
        print(f"  model_setup FAILED: {e}")
        return 1

    print("--- All smoke checks passed. Run `python pilot.py` for full pilot. ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
