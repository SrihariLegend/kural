"""Score commands on the NL2SH-ALFA test set (300 tasks) exactly as Westenfelder et al. (NAACL 2025) scored the
models in their results table: InterCode-ALFA 0.3.6, execution in its Docker environments, outputs compared with
the mxbai-embed-large embedding model (the "exec + mxbai-embed" judge, eval_mode="embed").

Setup (once): Docker running; Ollama serving mxbai-embed-large on localhost:11434 (CPU is fine:
`CUDA_VISIBLE_DEVICES= ollama serve`, then `ollama pull mxbai-embed-large`); and the benchmark package in its own
environment: `uv venv --python 3.12 data/eval/icalfa-venv && uv pip install --python data/eval/icalfa-venv/bin/python icalfa==0.3.6`.

  data/eval/icalfa-venv/bin/python tests/alfa_eval.py COMMANDS.json OUT.json [--limit N]

COMMANDS.json: a list of {"id": task index, "command": bash or null}. A null command (declined) scores 0, as an
empty answer would. Published scores on this set: GPT-4o 0.74, Qwen2.5-Coder-7B 0.61-0.62, Llama-3.1-8B 0.46-0.56.
"""
import json
import os
import sys
import time

from icalfa import submit_command


def main():
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    items = json.load(open(src))[:limit]
    done = {}
    if os.path.exists(dst):
        done = {r["id"]: r for r in json.load(open(dst))}          # resume
    if "--rescore-zero" in sys.argv:                                # judge again every answered task that scored 0
        done = {k: r for k, r in done.items() if r["score"] or not r["command"]}
    t0 = time.time()
    for n, it in enumerate(items):
        if it["id"] in done:
            continue
        cmd = it.get("command")
        # eval_param is the cosine threshold (the paper's 0.75); without it icalfa compares against its default
        # "gpt-4-0613" string, raises, and scores the task 0 whenever the two outputs differ.
        score = submit_command(index=it["id"], command=cmd, eval_mode="embed", eval_param=0.75) if cmd else 0
        done[it["id"]] = {"id": it["id"], "command": cmd, "score": score}
        if n % 10 == 0 or n == len(items) - 1:
            json.dump(sorted(done.values(), key=lambda r: r["id"]), open(dst, "w"), indent=1)
            s = sum(r["score"] for r in done.values())
            print(f"{len(done)}/{len(items)} scored, {s} right ({s / len(done):.3f}), {time.time() - t0:.0f}s", flush=True)
    json.dump(sorted(done.values(), key=lambda r: r["id"]), open(dst, "w"), indent=1)
    s = sum(r["score"] for r in done.values())
    print(f"final: {s}/{len(done)} = {s / len(done):.3f}")


if __name__ == "__main__":
    main()
