# Measured results

## NL2SH-ALFA (300 tasks, execution + mxbai-embed judge, InterCode-ALFA 0.3.6)

Scored with `tests/alfa_eval.py`, the judge the paper used for its model table (Westenfelder et al., NAACL 2025).

| System | Score | Note |
|---|---|---|
| Reference commands (`bash` column) | 297/300 (0.990) | checks the harness on this machine |
| **kural 0.2** (2026-11-16) | **55/300 (0.183)** | answered 263, declined 37; of 96 answers at confidence ≥ 0.95 only 38 were right |
| GPT-4o (published) | 0.74 | |
| Qwen2.5-Coder-7B (published, base / best wrapper) | 0.61 / 0.62 | |
| Llama-3.1-8B (published, base / best) | 0.46 / 0.56 | |
| Qwen2.5-Coder-1.5B (published, base / best) | 0.21 / 0.50 | |
| Llama-3.2-1B (published, base / best) | 0.12 / 0.37 | |

Per-task results: `alfa_kural_0.2.json`. What it shows: kural 0.2's 290 building blocks cover only part of this
benchmark, its wording coverage is limited to its training patterns, and its confidence is not calibrated. These are
the ladder's steps 3 (data, open bash form) and 4 (execution checks, confidence head) in `docs/ARCHITECTURE.md`.
