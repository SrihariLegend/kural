"""Check data written by kural-gen against the Python reference.

  python tests/check_rustgen.py SHARD_DIR [N]

1. Every example's program is valid under the grammar (kural/ir.py) and compiles to bash (kural/compile.py),
   features match kural/feat.py, and copied spans point inside the words.
2. Distribution: share of examples using each stage, lengths and context share, Rust shards vs mix.dataset.
"""
import collections
import statistics as st
import sys

sys.path.insert(0, ".")
from kural import compile as C
from kural import feat, ir, mix, shards


def grammar_ok(toks, n):
    g = ir.Grammar(n)
    for t in toks:
        if t not in g.allowed():
            return False
        g.advance(t)
    return g.done


def stats(rows):
    use = collections.Counter()
    for words, out, n_req in rows:
        use.update({ir.VOCAB[t] for t in out if t < ir.NV and ir.VOCAB[t] in ir.BY_NAME})
    n = len(rows)
    return {
        "use": {k: v / n for k, v in use.items()},
        "req": st.mean(r[2] for r in rows),
        "total": st.mean(len(r[0]) for r in rows),
        "out": st.mean(len(r[1]) for r in rows),
        "ctx": sum(len(r[0]) > r[2] for r in rows) / n,
        "multi": sum(ir.SEP in r[1] for r in rows) / n,
        "unknown": sum(r[1][0] == ir.TOK["unknown"] for r in rows) / n,
    }


def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    rows = list(shards.examples(path, n))
    print(f"read {len(rows)} examples from {path}")
    # ---- validity
    bad_grammar = bad_compile = unknown = compiled = 0
    errors = collections.Counter()
    for words, out, n_req in rows[:50_000]:
        if not grammar_ok(out, len(words)):
            bad_grammar += 1
            continue
        prog = ir.delinearise(out, words)
        try:
            C.compile_program(prog, C.Env(live=False, last="ls"))
            compiled += 1
        except C.Unsupported:
            if any(name == "unknown" for pipe in prog for name, _ in pipe):
                unknown += 1
            else:
                bad_compile += 1
                errors["Unsupported " + " | ".join(n for n, _ in prog[0])] += 1
        except Exception as e:
            bad_compile += 1
            errors[f"{type(e).__name__}: {e}"[:90]] += 1
    checked = min(len(rows), 50_000)
    print(f"validity on {checked}: grammar-invalid {bad_grammar}, compiled {compiled}, declined (unknown) {unknown}, failed to compile {bad_compile}")
    for k, v in errors.most_common(8):
        print(f"   {v:5d}  {k}")
    # ---- features: the shard's precomputed word features must equal kural/feat.py
    s = shards.Shard(shards.files(path)[0])
    mism = 0
    for i in range(1, min(s.wid.shape[0], 20_000)):
        w = s.word(i)
        tris, wid, shape = feat.word_features(w)
        t = list(s.tri[i][:len(tris)])
        bits = int(s.shape[i])
        if t != tris or any(x != feat.N_TRI for x in s.tri[i][len(tris):]) or int(s.wid[i]) != wid or [bool(bits >> j & 1) for j in range(16)] != [x > 0.5 for x in shape]:
            mism += 1
            if mism <= 3:
                print("   feature mismatch:", repr(w))
    print(f"features checked on {min(s.wid.shape[0], 20_000) - 1} distinct words: {mism} mismatches")
    # ---- distribution
    py = mix.dataset(len(rows), 4242)
    a, b = stats(rows), stats(py)
    print(f"\n{'':24} {'rust':>9} {'python':>9}")
    for k in ("req", "total", "out", "ctx", "multi", "unknown"):
        print(f"{k:24} {a[k]:9.4f} {b[k]:9.4f}")
    names = set(a["use"]) | set(b["use"])
    diffs = sorted(((a["use"].get(k, 0) - b["use"].get(k, 0), k) for k in names), key=lambda x: -abs(x[0]))
    print("\nlargest differences in the share of examples using a stage (rust - python):")
    for d, k in diffs[:15]:
        print(f"   {k:16} rust {100 * a['use'].get(k, 0):6.2f}%  python {100 * b['use'].get(k, 0):6.2f}%  diff {100 * d:+.2f}")
    missing = [k for k in b["use"] if k not in a["use"]]
    print("stages python makes but rust never does:", missing)
    print("stages rust makes but python never does:", [k for k in a["use"] if k not in b["use"]])


if __name__ == "__main__":
    main()
