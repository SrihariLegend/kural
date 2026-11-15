#!/usr/bin/env bash
# Rebuild the whole pretraining corpus under data/ (see data_tools/README.md).
# Every step is resumable: re-running skips finished tasks. Logs go to data/logs/.
# Expect several hours (Reddit streaming dominates) and up to ~110 GB of peak disk.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=(.venv/bin/python -I)
mkdir -p data/logs data/models
export HF_HOME="$ROOT/data/hf_home" HF_HUB_DISABLE_TELEMETRY=1

if [ ! -s data/models/lid.176.bin ]; then
  curl -sfL -o data/models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
fi

data_tools/fetch_hf.sh                                    2>&1 | tee data/logs/fetch_hf.log
"${PY[@]}" data_tools/eval_sets.py                        2>&1 | tee data/logs/eval_sets.log   # before any source!

# Reddit streaming is network-bound and runs alongside the CPU-bound steps.
"${PY[@]}" data_tools/reddit.py extract > data/logs/reddit_extract.log 2>&1 &
REDDIT=$!

"${PY[@]}" data_tools/wikipedia.py                        > data/logs/wikipedia.log 2>&1
"${PY[@]}" data_tools/stackexchange.py all                > data/logs/stackexchange.log 2>&1
"${PY[@]}" data_tools/web.py fineweb_edu                  > data/logs/fineweb_edu.log 2>&1
"${PY[@]}" data_tools/web.py finemath                     > data/logs/finemath.log 2>&1
"${PY[@]}" data_tools/docs.py all                         > data/logs/docs.log 2>&1
"${PY[@]}" data_tools/code.py all                         > data/logs/code.log 2>&1

wait "$REDDIT"
"${PY[@]}" data_tools/reddit.py threads                   > data/logs/reddit_threads.log 2>&1
"${PY[@]}" data_tools/reddit.py finalize                  >> data/logs/reddit_threads.log 2>&1
rm -rf data/work/reddit_extract data/hf_home/hub

"${PY[@]}" data_tools/manifest.py
du -sh data
