"""Natural rewordings of generated requests, written by a local language model and kept only when they round-trip.

kural's parser is trained on requests built from about 900 hand-written phrase patterns, so it understands wording
close to them. Here a local model (through Ollama) rewrites sampled requests the way different people would type
them, and each rewording is checked:

  1. every copied value (file names, sizes, numbers, search text) must appear in it word for word, so the program's
     pointers can be moved onto the new words;
  2. it must not be (or contain) any request of the test sets;
  3. round trip: today's parser must have the right program among its top three readings.

Kept rewordings become training examples with the original program (AutoQA, EMNLP 2020: the round-trip filter alone
was worth about 18 points). From 5% of the sampled requests, rewordings that pass checks 1-2 are set aside *without*
check 3 as a held-out "wording" test, which measures how well a model handles wording it was not built for.

  python data_tools/paraphrase.py OUT_DIR [--n 12000] --backend pi \
      --models google-vertex/gemini-4-hydrogen:high,azure/gpt-6-luna:max [--batch 20] [--workers 16]

The pi backend sends batches of requests to models through the `pi` CLI (one model per batch, taking turns);
every kept rewording records which model wrote it, and stats.json breaks the keep rate down per model.

Writes OUT_DIR/train.jsonl and OUT_DIR/heldout.jsonl ({words, out, n, text, source, command}) and OUT_DIR/stats.json.
Resumable: OUT_DIR/raw.jsonl keeps every model answer.
"""
import argparse
import subprocess
import glob
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kural import cli, ir, mix, tok  # noqa: E402
from kural import compile as C  # noqa: E402

PROMPT = """Here is a request someone typed to a Linux terminal assistant:

{text}

Rewrite it the way {k} different real people might type it. Vary the wording, word order and tone: casual, terse, \
polite, wordy, a beginner who does not know the jargon, someone in a hurry (one may have a small typo in an ordinary \
word). Keep the meaning exactly the same: do not add or drop any condition or step.
These exact strings must appear unchanged in every version: {values}
Write one version per line, with no numbering, quotes or explanations."""


def spans(out):
    """[(start, end)] word positions copied by the program, in order of appearance."""
    res, i, toks = [], 0, list(out)
    while i < len(toks):
        t = toks[i]; i += 1
        if t in (ir.END, ir.PAD):
            break
        if t == ir.SEP or t >= ir.NV:
            continue
        for slot in ir.BY_NAME[ir.VOCAB[t]].slots:
            v = toks[i]; i += 1
            if slot.kind == "span" and v != ir.NONE:
                e = toks[i]; i += 1
                res.append((v - ir.NV, e - ir.NV))
    return res


def move_spans(out, old_words, new_words):
    """The program's tokens with every copied span moved onto the same words in new_words, or None."""
    res, i, toks = [], 0, list(out)
    while i < len(toks):
        t = toks[i]; i += 1
        res.append(t)
        if t in (ir.END, ir.PAD) or t == ir.SEP or t >= ir.NV:
            continue
        for slot in ir.BY_NAME[ir.VOCAB[t]].slots:
            v = toks[i]; i += 1
            if slot.kind != "span" or v == ir.NONE:
                res.append(v)
                continue
            e = toks[i]; i += 1
            target = old_words[v - ir.NV:e - ir.NV + 1]
            at = next((j for j in range(len(new_words) - len(target) + 1) if new_words[j:j + len(target)] == target), None)
            if at is None:
                return None
            res += [ir.NV + at, ir.NV + at + len(target) - 1]
    return res


def test_requests():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    reqs = set()
    for path in glob.glob(os.path.join(root, "tests", "*.txt")):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith(("#", "!")):
                reqs.add(line.lstrip("+ ").split(" => ")[0].strip().lower())
    alfa = os.path.join(root, "data", "eval", "nl2sh_alfa", "test.jsonl")
    if os.path.exists(alfa):
        reqs |= {json.loads(l)["nl"].strip().lower() for l in open(alfa)}
    return reqs


def sample(n, seed):
    """n generated requests without session context: (words, out tokens, gold command)."""
    r = random.Random(seed)
    env = C.Env(live=False, last="ls")
    out = []
    while len(out) < n:
        words, toks, n_req = mix.example(r)
        if n_req != len(words) or len(words) < 3:
            continue
        program = ir.delinearise(toks, words)
        if any(name in ("unknown", "pref", "again", "last") for pipe in program for name, _ in pipe):
            continue
        try:
            cmd = C.compile_program(program, env.fresh())
        except Exception:
            continue
        out.append((words, toks, cmd))
    return out


BATCH_PROMPT = """Below are requests people typed to a Linux terminal assistant, numbered. For EACH request, rewrite it \
the way {k} different real people might type it. Vary the wording, word order and tone: casual, terse, polite, wordy, \
a beginner who does not know the jargon, someone in a hurry (at most one version per request may have a small typo in \
an ordinary word). Keep the meaning exactly the same: do not add or drop any condition, value or step.
The strings listed after "keep:" must appear unchanged, character for character, in every version of that request.

{items}

Answer with exactly one JSON object per line and nothing else: {{"i": <number>, "v": ["version 1", ...]}}"""


def ask_pi(model, batch, k=4):
    """batch: [(i, text, values)] -> {i: [versions]}"""
    items = "\n".join(f"{i}. {text}\n   keep: {values}" for i, text, values in batch)
    cmd = ["pi", "-p", "--no-tools", "--no-session", "--no-context-files", "--no-skills", "--no-extensions",
           "--model", model, "--system-prompt", "You write training data. Follow the instructions exactly.",
           BATCH_PROMPT.format(k=k, items=items)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd="/tmp", stdin=subprocess.DEVNULL)
    out = {}
    for line in p.stdout.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
            out[int(r["i"])] = [str(v) for v in r.get("v", [])][:k + 2]
        except (ValueError, KeyError, TypeError):
            continue
    return out


def ask(host, model, text, values, k=4):
    body = json.dumps({"model": model, "prompt": PROMPT.format(text=text, k=k, values=values), "stream": False,
                       "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 300}}).encode()
    req = urllib.request.Request(host + "/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--host", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", choices=["ollama", "pi"], default="ollama")
    ap.add_argument("--models", default="google-vertex/gemini-4-hydrogen:high,azure/gpt-6-luna:max")
    ap.add_argument("--batch", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw_path = os.path.join(a.out, "raw.jsonl")
    items = sample(a.n, a.seed)
    done, by_model = {}, {}
    if os.path.exists(raw_path):
        for line in open(raw_path):
            try:
                r = json.loads(line)
                done[r["i"]] = r["answer"]
                by_model[r["i"]] = r.get("model", a.model)
            except ValueError:
                pass

    def job(i):
        words, toks, _ = items[i]
        vals = "; ".join(" ".join(words[s:e + 1]) for s, e in spans(toks)) or "(none)"
        try:
            return i, ask(a.host, a.model, " ".join(words), vals)
        except Exception as e:
            return i, f"<error {e}>"

    todo = [i for i in range(len(items)) if i not in done]
    if a.backend == "pi":
        models = a.models.split(",")
        batches = [todo[j:j + a.batch] for j in range(0, len(todo), a.batch)]

        def pjob(b):
            bi, ids = b
            model = models[bi % len(models)]
            payload = []
            for i in ids:
                words, toks, _ = items[i]
                payload.append((i, " ".join(words), "; ".join(" ".join(words[s:e + 1]) for s, e in spans(toks)) or "(none)"))
            try:
                return model, ids, ask_pi(model, payload)
            except Exception as e:
                print(f"batch {bi} failed: {e}", flush=True)
                return model, ids, {}

        with open(raw_path, "a") as raw, ThreadPoolExecutor(a.workers) as pool:
            for n, (model, ids, res) in enumerate(pool.map(pjob, enumerate(batches))):
                for i in ids:
                    if i in res:
                        done[i] = "\n".join(res[i]); by_model[i] = model
                        raw.write(json.dumps({"i": i, "answer": done[i], "model": model}) + "\n")
                raw.flush()
                if n % 10 == 0:
                    print(f"asked {len(done)}/{len(items)} ({n + 1}/{len(batches)} batches)", flush=True)
    else:
        with open(raw_path, "a") as raw, ThreadPoolExecutor(a.workers) as pool:
            for n, (i, answer) in enumerate(pool.map(job, todo)):
                done[i] = answer
                raw.write(json.dumps({"i": i, "answer": answer, "model": a.model}) + "\n")
                if n % 200 == 0:
                    raw.flush()
                    print(f"asked {len(done)}/{len(items)}", flush=True)

    tests = test_requests()
    env = C.Env(live=False, last="ls")
    stats = {"requests": len(items), "answered": len(done), "rewordings": 0, "values_missing": 0, "test_overlap": 0,
             "duplicate": 0, "roundtrip_failed": 0, "kept": 0, "heldout": 0, "per_model": {}}
    seen = set()
    with open(os.path.join(a.out, "train.jsonl"), "w") as tr, open(os.path.join(a.out, "heldout.jsonl"), "w") as ho:
        for i in range(len(items)):
            words, toks, cmd = items[i]
            held = i % 20 == 0
            pm = stats["per_model"].setdefault(by_model.get(i, "?"), {"rewordings": 0, "kept": 0})
            for line in (done.get(i) or "").split("\n"):
                text = line.strip().strip('"').lstrip("-*0123456789.) ").strip()
                if not text or text.startswith("<error") or len(text) > 300:
                    continue
                stats["rewordings"] += 1
                pm["rewordings"] += 1
                low = text.lower()
                if low in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(low)
                if low in tests or any(t in low for t in tests if len(t) > 25):
                    stats["test_overlap"] += 1
                    continue
                new_words = tok.words(text)
                if not new_words or len(new_words) > mix.MAX_REQUEST:
                    continue
                new_out = move_spans(toks, words, new_words)
                if new_out is None:
                    stats["values_missing"] += 1
                    continue
                rec = {"words": new_words, "out": new_out, "n": len(new_words), "text": text, "source": " ".join(words), "command": cmd,
                       "model": by_model.get(i, "?")}
                if held:
                    stats["heldout"] += 1
                    ho.write(json.dumps(rec) + "\n")
                    continue
                cands = [c["command"] for c in cli.translate(text, env.fresh())[:3]]
                if cmd not in cands:
                    stats["roundtrip_failed"] += 1
                    continue
                stats["kept"] += 1
                pm["kept"] += 1
                tr.write(json.dumps(rec) + "\n")
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
