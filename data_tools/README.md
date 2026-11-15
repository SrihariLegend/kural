# data_tools: building kural's pretraining corpus

These scripts download, clean, deduplicate and decontaminate the text used to pretrain
kural's next model (docs/ARCHITECTURE.md, section 4.1), and build the English-to-command
task data from Stack Exchange. Everything they produce lives under `data/`, which is
git-ignored. `data/MANIFEST.md` lists what was built: per source the origin, licence,
filters, document count, bytes and estimated tokens.

## Rebuild

```bash
uv pip install --python .venv/bin/python datasets pyarrow zstandard datasketch huggingface_hub \
    hf_xet fasttext-numpy2-wheel pyahocorasick lxml regex xxhash orjson requests
data_tools/run_all.sh            # everything, in order; several hours, mostly network-bound
```

Each step can also be run on its own (all are resumable: a finished task is skipped, so an
interrupted run continues; delete `data/work/<source>/` to redo a source from scratch):

| step | command | output |
|---|---|---|
| inputs from Hugging Face | `data_tools/fetch_hf.sh` | `data/raw/...` |
| eval sets, decontamination list | `python -I data_tools/eval_sets.py` | `data/eval/`, `data/pairs/nl2sh_alfa_train.jsonl` |
| Wikipedia | `python -I data_tools/wikipedia.py` | `clean/wikipedia` |
| Stack Exchange + pairs | `python -I data_tools/stackexchange.py all` (or `sites`, `so`, `finalize`) | `clean/stackexchange`, `clean/stackoverflow_general`, `clean/stackexchange_math`, `pairs/` |
| FineWeb-Edu, FineMath | `python -I data_tools/web.py fineweb_edu` / `finemath` | `clean/fineweb_edu`, `clean/finemath` |
| man, tldr, GNU, Arch Wiki | `python -I data_tools/docs.py all` (or `man`, `tldr`, `gnu`, `archwiki`) | `clean/docs_*` |
| code | `python -I data_tools/code.py all` (or `curated`, `github`, `finalize`) | `clean/code` |
| Reddit | `python -I data_tools/reddit.py all` (or `extract`, `threads`, `finalize`) | `clean/reddit_tech`, `clean/reddit_general` |
| manifest | `python -I data_tools/manifest.py` | `data/MANIFEST.md` |

Run `eval_sets.py` before any source: the sources read `data/eval/decontam_patterns.txt`
while filtering, and a source built without it is not decontaminated.

## Layout

```
data/
  raw/            downloads (deleted after processing; Reddit and Stack Overflow are streamed)
  work/           per-task intermediate parts + dedup signatures (deleted by finalize)
  clean/<source>/shard-NNNNN.jsonl.zst   one JSON object per line: {"text", "source", "meta"}
  clean/<source>/stats.json              counts, drop reasons, origin, licence
  pairs/          stackexchange_pairs.jsonl, heldout.jsonl (never train on it), nl2sh_alfa_train.jsonl
  eval/           NL2Bash, InterCode-Bash, NL2SH-ALFA test, decontam_patterns.txt (never train on these)
  models/         lid.176.bin (fastText language ID)
  MANIFEST.md
```

Shards hold about 256 MB of text each, zstd level 6. Read them with
`common.read_jsonl_zst(path)` or `zstdcat shard-00000.jsonl.zst | head`.

## How the pipeline works (common.py)

1. Each source script splits its input into tasks and runs them in a process pool (all cores
   but two). A worker parses its task into documents, then applies the shared filters:
   length bounds, fastText English ID, Gopher-style prose checks (lenient: typos and casual
   writing pass, symbol walls and repeated lines do not), spam/link-dump checks and
   decontamination. Survivors go to `data/work/<source>/part-<task>.jsonl.zst` with an exact
   hash and 14 MinHash band hashes per document.
2. `finalize()` loads all signatures, keeps the first occurrence of each exact duplicate and
   drops any document that shares a MinHash band with an earlier one (Jaccard >= ~0.75 on
   5-word shingles), applies an optional byte budget, and writes the shards.

Token counts are estimated as UTF-8 bytes / 4.

## Held-out Stack Exchange questions

A question is held out when `xxh64("<site>:<id>") % 100 < 2` (`stackexchange.is_heldout`).
Held-out questions are excluded from every pretraining source and from
`stackexchange_pairs.jsonl`; the shell-relevant ones are written to `data/pairs/heldout.jsonl`.
Use the same function on any later Stack Exchange-derived data.

## Adding a source

Write `produce(task_arg, drops)` yielding `{"text", "source", "meta"}`, pick a `Filters`,
call `run_tasks(name, tasks, produce, filters)` and `finalize(name, task_ids)`, and add a
description to `NOTES` / `GROUPS` in `manifest.py`.
