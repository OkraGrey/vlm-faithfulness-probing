# Pilot

Minimal end-to-end pipeline to verify VLM hidden-state extraction on BLINK before any probing experiments.

See `plan.md` for the full implementation plan and `../reqs.md` for the research context.

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Verify imports + BLINK loader without downloading the model:
python smoke_test.py

# Full pilot (downloads Qwen2.5-VL-3B on first run, ~6 GB):
python pilot.py
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
