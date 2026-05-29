"""End-to-end pilot: load BLINK -> load model -> infer -> extract -> save.

This is the smallest run that proves the full pipeline works.
"""
import numpy as np
import torch

import config
from data_loader import load_blink_samples
from model_setup import load_model_and_processor, run_single


def main():
    print("=== Pilot run ===")
    print(f"Model:   {config.MODEL_NAME}")
    print(f"Device:  {config.DEVICE} ({config.DTYPE})")
    print(f"Task:    {config.BLINK_TASK}")
    print(f"Samples: {config.NUM_SAMPLES}")
    print()

    model, processor = load_model_and_processor()
    samples = list(load_blink_samples())

    all_layer_states: list[np.ndarray] = []     # final shape (N, num_layers, hidden_dim)
    records = []

    for i, (images, question, options, answer) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Q: {question[:70]}")
        try:
            generated, hidden = run_single(model, processor, images, question)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        layer_indices = sorted(hidden.keys())
        stacked = torch.stack([hidden[k] for k in layer_indices], dim=1).squeeze(0).numpy()
        all_layer_states.append(stacked)

        print(f"  generated: {generated[:80]!r}")
        print(f"  gold:      {answer}  options={options}")
        print(f"  hidden_states: {stacked.shape}  dtype={stacked.dtype}")

        records.append({
            "idx": i,
            "question": question,
            "options": options,
            "gold": answer,
            "generated": generated,
        })

    if not all_layer_states:
        print("\nNo samples succeeded. Aborting save.")
        return

    arr = np.stack(all_layer_states, axis=0)
    out_path = config.OUTPUTS_DIR / "pilot_hidden_states.npz"
    np.savez_compressed(
        out_path,
        hidden_states=arr,
        gold=np.array([r["gold"] for r in records]),
        generated=np.array([r["generated"] for r in records]),
        questions=np.array([r["question"] for r in records]),
    )
    print(f"\nSaved {arr.shape} -> {out_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
