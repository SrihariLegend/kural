#!/usr/bin/env bash
# One unattended run: local-model rewordings -> word table -> training -> export -> every evaluation.
# Usage: data_tools/overnight_paraphrase.sh OUT_DIR [N_REQUESTS]
# Needs: Ollama with qwen2.5:7b-instruct on 127.0.0.1:11435 (GPU) and mxbai-embed-large on 127.0.0.1:11434 (CPU),
# Docker, data/eval/icalfa-venv (see tests/alfa_eval.py), target/release/kural.
set -u
cd "$(dirname "$0")/.."
O=$1; N=${2:-12000}
mkdir -p "$O/cand"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/run.log"; }
FROM=${FROM:-reword}            # FROM=train resumes after the rewordings and word table
if [ "$FROM" = reword ]; then
log "rewording $N requests"
python3 data_tools/paraphrase.py "$O/para" --n "$N" >> "$O/run.log" 2>&1
log "stats: $(cat "$O/para/stats.json" | tr -d '\n ')"
# free the GPU for training
for p in $(pgrep -x ollama); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "OLLAMA_HOST=127.0.0.1:11435" && kill "$p"; done
log "word table"
KURAL_LEXICON="$O/cand/lexicon.json" KURAL_LEXICON_EXTRA="$O/para/train.jsonl" python3 -m kural.lexicon >> "$O/run.log" 2>&1
fi
log "training"
.venv/bin/python -m kural.train --n 1600000 --epochs 12 --extra "$O/para/train.jsonl" --extra-repeat 3 --out "$O/cand/weights.npz" >> "$O/run.log" 2>&1
python3 -m kural.export --weights "$O/cand/weights.npz" --out "$O/cand/model.kural" >> "$O/run.log" 2>&1
log "evaluating"
python3 tests/report.py --final > "$O/report_base.txt" 2>&1
KURAL_MODEL="$O/cand/model.kural" KURAL_LEXICON="$O/cand/lexicon.json" KURAL_WEIGHTS="$O/cand/weights.npz" python3 tests/report.py --final > "$O/report_cand.txt" 2>&1
python3 tests/eval_wording.py "$O/para/heldout.jsonl" > "$O/wording_base.txt" 2>&1
KURAL_MODEL="$O/cand/model.kural" KURAL_LEXICON="$O/cand/lexicon.json" python3 tests/eval_wording.py "$O/para/heldout.jsonl" > "$O/wording_cand.txt" 2>&1
KURAL_MODEL="$O/cand/model.kural" KURAL_LEXICON="$O/cand/lexicon.json" python3 - "$O" <<'PY'
import json, subprocess, sys
o = sys.argv[1]
rows = [json.loads(l) for l in open("data/eval/nl2sh_alfa/test.jsonl")]
out = []
for r in rows:
    p = subprocess.run(["./target/release/kural", "--shell", "--cwd", "/testbed", r["nl"]], capture_output=True, text=True)
    out.append({"id": r["id"], "command": p.stdout.split("\n")[0] if p.returncode == 0 and p.stdout.strip() else None})
json.dump(out, open(f"{o}/alfa_cand_commands.json", "w"))
PY
K=$(pwd)
(cd "$O" && "$K/data/eval/icalfa-venv/bin/python" "$K/tests/alfa_eval.py" alfa_cand_commands.json alfa_cand_scores.json > alfa_cand.txt 2>&1)
log "done: $(grep final "$O/alfa_cand.txt")"
