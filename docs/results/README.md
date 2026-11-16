# Measured results

## NL2SH-ALFA (300 tasks, execution + mxbai-embed judge, InterCode-ALFA 0.3.6)

Scored with `tests/alfa_eval.py`, the judge the paper used for its model table (Westenfelder et al., NAACL 2025).

| System | Score | Note |
|---|---|---|
| Reference commands (`bash` column) | 297/300 (0.990) | checks the harness on this machine |
| **kural 0.3** (2026-11-16, + 36,222 round-trip-filtered local-model rewordings) | **97/300 (0.323)** | |
| kural 0.2 (2026-11-16) | 91/300 (0.303) | answered 263, declined 37. (First published as 0.183: the scoring script did not pass the judge's 0.75 threshold, so every non-identical output scored 0. Fixed in 75f162c and re-judged.) |
| GPT-4o (published) | 0.74 | |
| Qwen2.5-Coder-7B (published, base / best wrapper) | 0.61 / 0.62 | |
| Llama-3.1-8B (published, base / best) | 0.46 / 0.56 | |
| Qwen2.5-Coder-1.5B (published, base / best) | 0.21 / 0.50 | |
| Llama-3.2-1B (published, base / best) | 0.12 / 0.37 | |

Per-task results: `alfa_kural_0.2.json`, `alfa_kural_0.3.json`. In each run about 3 of 300 judgments errored
(an output longer than the embedding model's context, or a container that failed to reset) and count as 0.

## Rewordings held out from training (2,736, written by qwen2.5:7b, never filtered by any parser)

| Model | Right | In top 3 | Declined |
|---|---|---|---|
| kural 0.2 | 64.5% | 70.9% | 309 |
| **kural 0.3** | **75.4%** | **80.5%** | 156 |

## Repo test sets (kural 0.2 → 0.3)

v2_final 185 → 185/188, v2_dev 258 → 258/261, once_outside 36 → 37/39, natural_dev 137 → 135/140,
natural_test 107 → 106/109, natural_final 73 → 73/74, outside_v2 declined 40 → 40/40.

Engine and run-time parity after the change: Rust and NumPy agree on all 851 test requests; the Rust run time
matches Python on all 851; 51/51 executed commands. What it shows: kural 0.2's 290 building blocks cover only part of this
benchmark, its wording coverage is limited to its training patterns, and its confidence is not calibrated. These are
the ladder's steps 3 (data, open bash form) and 4 (execution checks, confidence head) in `docs/ARCHITECTURE.md`.
