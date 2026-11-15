"""Which words license which part of the language.

The parser can be confidently wrong about a request it was never built for ("how's the weather looking").
This is a plain check that sits beside it: a reading is only accepted if at least one meaningful word of the
request has been seen, in training, in requests for some stage of that reading. A request whose words are all
foreign to the reading is declined instead.

  python -m plainsh.lexicon [N]      rebuild plainsh/lexicon.json from N generated requests (default 400000)

The table is built from the same generator the parser is trained on, so the two always agree on vocabulary.
"""
import difflib
import json
import os
import re
import sys

from . import ir

PATH = os.path.join(os.path.dirname(__file__), "lexicon.json")
_LEX = None
_WORD = re.compile(r"[a-z][a-z0-9+\-]*")


def content(words):
    """Lower-case word stems worth checking: letters only, at least 3 long."""
    out = []
    for w in words:
        out += [m for m in _WORD.findall(w.lower()) if len(m) >= 3]
    return out


def load():
    global _LEX
    if _LEX is None:
        try:
            with open(PATH) as f:
                raw = json.load(f)
            _LEX = {"generic": set(raw["generic"]), "stages": {k: set(v) for k, v in raw["stages"].items()}}
        except (OSError, ValueError, KeyError):
            _LEX = {"generic": set(), "stages": {}}
    return _LEX


def _known(word, vocab):
    if word in vocab:
        return True
    if len(word) >= 5:                                   # allow a typo, or an ending the table has not seen
        stem = word[:5]
        if any(v.startswith(stem) for v in vocab):
            return True
        return bool(difflib.get_close_matches(word, vocab, n=1, cutoff=0.8))
    return False


def support(program, words, used):
    """-> (meaningful words that fit the reading, meaningful words in the request), or None if nothing can be checked.
    words: the request's words; used: positions the program copied as arguments."""
    lex = load()
    if not lex["stages"]:
        return None
    literal = [w for i, w in enumerate(words) if i not in used]
    cand = [w for w in content(literal) if w not in lex["generic"]]
    if not cand:
        return None
    vocab = set()
    for pipe in program:
        for name, _ in pipe:
            vocab |= lex["stages"].get(name, set())
    return sum(_known(w, vocab) for w in cand), len(cand)


def build(n=400_000, seed=77):
    import random
    from collections import Counter
    from . import mix
    r = random.Random(seed)
    seen, total = {}, Counter()
    made = 0
    while made < n:
        words, out, n_req = mix.example(r)
        toks = [t for t in out if t != ir.END]
        if ir.SEP in toks:
            continue                                     # several steps: words cannot be attributed to one of them
        made += 1
        used = ir.spans_used(out)
        literal = set(content([w for i, w in enumerate(words[:n_req]) if i not in used]))
        for name in {ir.VOCAB[t] for t in toks if t < ir.NV and ir.VOCAB[t] in ir.BY_NAME}:
            total[name] += 1
            seen.setdefault(name, Counter()).update(literal)
    stages = {}
    for name, c in seen.items():
        floor = max(4, 0.005 * total[name])              # drop typos and one-off filler words
        stages[name] = sorted(w for w, k in c.items() if k >= floor)
    df = Counter(w for ws in stages.values() for w in ws)
    generic = sorted(w for w, k in df.items() if k >= 0.2 * len(stages))
    gs = set(generic)
    stages = {k: [w for w in v if w not in gs] for k, v in stages.items()}
    with open(PATH, "w") as f:
        json.dump({"generic": generic, "stages": stages}, f, separators=(",", ":"))
    return generic, stages


if __name__ == "__main__":
    generic, stages = build(int(sys.argv[1]) if len(sys.argv) > 1 else 400_000)
    print(f"{len(stages)} stages, {sum(len(v) for v in stages.values())} words, {len(generic)} generic words, {os.path.getsize(PATH) // 1024} KB")
    print("generic:", " ".join(generic[:80]))
    for name in ("tree", "timer", "reboot", "svc_restart", "delete", "lrange", "unknown"):
        print(f"{name}: {' '.join(stages.get(name, [])[:40])}")
