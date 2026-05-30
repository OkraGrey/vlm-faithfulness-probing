# Exact runtime environment (captured 2026-05-30)

The results in `outputs/*.json` were produced under the following stack. `requirements.txt`
holds only loose lower bounds (`transformers>=4.49.0` etc.) and will **not** reproduce these
numbers on a fresh install; use `requirements.lock.txt` (exact `pip freeze`) instead.

| Component | Version used |
|---|---|
| Python | 3.13.7 |
| Platform | macOS-15.5-arm64 (Apple silicon, MPS backend) |
| torch | 2.12.0 |
| transformers | 5.8.1 |
| numpy | 2.4.5 |
| scikit-learn | 1.8.0 |
| datasets | 4.8.5 |
| Pillow | 12.2.0 |
| accelerate | 1.13.0 |

**Model snapshot (pin this — weights can change):**
`Qwen/Qwen2.5-VL-3B-Instruct` @ commit `66285546d2b821cf421d4f5eb2576359d3770cd3`
(HuggingFace Hub default cache). Pass `revision="66285546..."` to `from_pretrained` for an
exact-weights reproduction.

## Reproduce the environment

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.lock.txt   # exact pins (preferred)
# or, for a looser install that may drift:  pip install -r requirements.txt
```

## Determinism notes

- Inference uses greedy decoding (`do_sample=False`, `model_setup.run_single`) — deterministic
  given fixed weights + transformers version. Different transformers/torch versions can change
  tokenization, attention kernels, and therefore which samples land in the wrong-output subset.
- All probe/analysis steps fix seeds (`SEEDS = [0..4]`, `StratifiedKFold(random_state=...)`),
  so `gate0_cv`, `gate0_diag`, `layer_profile`, `gate3_construct`, `gate4_patching` are
  reproducible bit-for-bit on the same `dataset.npz`.
- MPS float16 matmuls are not guaranteed identical across torch versions; for an exact rerun of
  the *inference* stage, pin torch to the version above.
