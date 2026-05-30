# Probing pipeline — Qwen2.5-VL-3B on BLINK

End-to-end pipeline that extracts VLM hidden states on BLINK and runs the gated probing study
(Gates 1 → 0 → 0-diag → 3 → 3-addendum → 4). Research context: `../reqs.md`. Current state and
findings: `../docs/00_start_here.md`. Per-gate reasoning: `../docs/03_methodology_log.md`.
**Pipeline diagram + runbook + reproducibility status: `../docs/10_pipeline_and_reproducibility.md`.**

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.lock.txt    # exact pins (preferred). Versions/model commit: ENVIRONMENT.md
```

## Run

```bash
# Regenerate every gate result from the already-extracted dataset.npz (~2 min, no model):
./run_all.sh analysis

# Full pipeline incl. model inference + Gate 4 (hours on Mac MPS, resumable):
./run_all.sh full

# Individual stages (see docs/10 for the dependency order):
python smoke_test.py          # env check, no model load
python run_inference.py       # Phase 1 extraction (resumable)
python collate.py             # build outputs/dataset.npz
python gate0_cv.py            # Gate 0 (honest nested CV)
python gate0_diag.py          # Gate 0 diagnostic (gold vs own-output)
python gate3_construct.py     # Gate 3 (construct validity)
python layer_profile.py       # Gate 3 addendum (depth profile)
python gate4_patching.py      # Gate 4 (causal usage)
```

## What gets saved

`outputs/pilot_hidden_states.npz` with arrays:
- `hidden_states`  shape `(N, num_layers, hidden_dim)`
- `gold`           ground-truth answer letters
- `generated`      model's generated answers
- `questions`      original BLINK questions

## Hardware notes

- The pilot defaults to **Qwen2.5-VL-3B-Instruct** (~6 GB in fp16) for Mac compatibility.
- Switch to `Qwen2.5-VL-7B-Instruct` in `config.py` when running on a CUDA cluster.
- Mac users: device auto-selects to MPS; expect minutes-per-sample inference on a M-series chip.
- Cluster users: device auto-selects to CUDA; expect seconds-per-sample on an A100.
