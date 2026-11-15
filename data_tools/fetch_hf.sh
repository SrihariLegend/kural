#!/usr/bin/env bash
# Download the fixed-size Hugging Face inputs (everything except the sources that are
# streamed shard by shard: Stack Overflow, Reddit, github-code-clean).
# Usage: data_tools/fetch_hf.sh            (re-runnable: hf download skips finished files)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
export HF_HOME="$ROOT/data/hf_home" HF_HUB_DISABLE_TELEMETRY=1
HF="$ROOT/.venv/bin/hf"

dl() { # repo  local_dir  include-patterns...
  local repo=$1 dir=$2; shift 2
  local args=(); for p in "$@"; do args+=(--include "$p"); done
  echo "[$(date +%T)] $repo -> $dir ($*)"
  "$HF" download "$repo" --repo-type dataset "${args[@]}" --local-dir "$RAW/$dir" >/dev/null
}

dl wikimedia/wikipedia wikipedia '20231101.en/*'
dl HuggingFaceFW/fineweb-edu fineweb_edu 'sample/10BT/000_00000.parquet' 'sample/10BT/001_00000.parquet' \
   'sample/10BT/002_00000.parquet' 'sample/10BT/003_00000.parquet'
dl HuggingFaceTB/finemath finemath 'finemath-4plus/train-0000[0-7]-*'
SITES="unix.stackexchange.com askubuntu.com superuser.com serverfault.com math.stackexchange.com mathoverflow.net
 codereview.stackexchange.com softwareengineering.stackexchange.com cs.stackexchange.com english.stackexchange.com
 ell.stackexchange.com vi.stackexchange.com emacs.stackexchange.com raspberrypi.stackexchange.com
 retrocomputing.stackexchange.com networkengineering.stackexchange.com stats.stackexchange.com"
pats=(); for s in $SITES; do pats+=("$s/*"); done
dl HuggingFaceTB/stackexchange_2025_md se_md "${pats[@]}"
dl westenfelder/NL2SH-ALFA nl2sh_alfa '*'
echo "[$(date +%T)] done"
