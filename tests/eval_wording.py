"""Accuracy on held-out rewordings (data_tools/paraphrase.py heldout.jsonl): requests reworded by a local model,
checked only for value alignment and test overlap, never used for training or filtered by any parser.

  python tests/eval_wording.py HELDOUT.jsonl      (uses KURAL_MODEL / KURAL_LEXICON when set)
"""
import json
import sys

sys.path.insert(0, ".")
from kural import cli
from kural import compile as C

rows = [json.loads(l) for l in open(sys.argv[1])]
env = C.Env(live=False, last="ls")
right = top3 = declined = 0
for r in rows:
    cands = [c["command"] for c in cli.translate(r["text"], env.fresh())]
    right += bool(cands) and cands[0] == r["command"]
    top3 += r["command"] in cands[:3]
    declined += not cands
n = len(rows)
print(f"{sys.argv[1]}: {right}/{n} right ({100 * right / n:.1f}%), in top 3 {top3}/{n} ({100 * top3 / n:.1f}%), declined {declined}")
