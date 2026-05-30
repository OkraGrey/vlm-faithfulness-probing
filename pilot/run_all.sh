#!/usr/bin/env bash
# Reproducible end-to-end pipeline for the VLM faithfulness probing study (Qwen2.5-VL-3B / BLINK).
#
# Two modes:
#   ./run_all.sh analysis   # FAST (~2 min, no GPU/model): regenerates every gate result
#                           # from the already-extracted outputs/dataset.npz + inference_noise/.
#   ./run_all.sh full       # SLOW (hours on Mac MPS): also re-runs model inference from scratch.
#
# Each stage is idempotent. Inference stages are resumable (skip already-saved samples).
# Stages run in strict dependency order; a failed stage aborts the run (set -e).
set -euo pipefail
cd "$(dirname "$0")"
PY=./env/bin/python
MODE="${1:-analysis}"

run() { echo; echo "=== $* ==="; "$@"; }

if [ "$MODE" = "full" ]; then
  run $PY smoke_test.py                       # 0. env sanity (no model load)
  run $PY run_inference.py                    # 1. Phase 1: extract hidden states (HEAVY, resumable)
  run $PY collate.py                          # 2. build dataset.npz
fi

# --- analysis stages: pure functions of dataset.npz (+ inference_noise/) ---
run $PY gate0_cv.py --selftest                # leakage / planted-signal self-test
run $PY baselines.py                          # Gate 1   -> baselines_summary.json
run $PY gate0_cv.py                           # Gate 0   -> gate0_cv_summary.json   (honest nested CV)
run $PY gate0_diag.py                         # Gate 0 diagnostic -> gate0_diag_summary.json (gold vs output)

if [ "$MODE" = "full" ]; then
  run $PY run_inference_noise.py              # 3. noise-image inference for G3 candidates (HEAVY, resumable)
fi

run $PY gate3_construct.py                    # Gate 3   -> gate3_construct_summary.json (construct validity)
run $PY layer_profile.py                      # Gate 3 addendum -> layer_profile_summary.json (depth profile)

if [ "$MODE" = "full" ]; then
  run $PY gate4_patching.py --selftest        # harness validity (baseline reproduces wrong answer)
  run $PY gate4_patching.py                   # Gate 4   -> gate4_patching_summary.json (causal usage)
else
  echo; echo "NOTE: Gate 4 (gate4_patching.py) needs the live model — run in 'full' mode."
fi

echo; echo "=== DONE ($MODE). Results in outputs/*.json ==="
