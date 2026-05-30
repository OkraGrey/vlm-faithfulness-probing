# 3B baseline results (Qwen2.5-VL-3B) — frozen for the 7B comparison

These JSONs are the canonical Qwen2.5-VL-**3B** results from the 2026-05-30 study, copied
here from `outputs/` (which is git-ignored). They are the **comparison baseline** for the 7B
run. Do NOT overwrite them — the 7B run writes to `outputs/` and should be saved to a sibling
`results_7b/` folder so both models can be compared side by side.

| File | Gate | 3B headline |
|---|---|---|
| `baselines_summary.json` | Gate 1 (single-split) | 2/14 "pass" — **later RETRACTED** by Gate 0 |
| `gate0_cv_summary.json` | Gate 0 (honest nested CV) | Forensic/Object_Localization fail honest CV |
| `gate0_diag_summary.json` | Gate 0 diagnostic | 2 GOLD-readers (Jigsaw, Multi-view), 9 OUTPUT-readers (PAPO-consistent) |
| `gate3_construct_summary.json` | Gate 3 (noise ablation) | Jigsaw + Multi-view gold-signal is visual (collapses under noise) |
| `layer_profile_summary.json` | Gate 3 addendum | gold decodable flat-from-L0 → shallow cue, NOT deep grounding |
| `gate4_patching_summary.json` | Gate 4 (causal steering) | direction is causally INERT → H1 strong form REJECTED at 3B |

**3B verdict:** supports PAPO for 9/14 tasks; the 2 "sees-but-can't-say" tasks have a
shallow/early, causally-inert signal. See `../../docs/00_start_here.md` and
`../../docs/03_methodology_log.md` (Gates 0/3/4) for full reasoning.

**The 7B question:** does the gold signal *emerge with depth* (genuine grounding) and become
*causally usable*, or stay shallow + inert as at 3B? Run the identical pipeline at 7B
(`export QWEN_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`) and compare against these files.
