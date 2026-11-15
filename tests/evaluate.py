"""Score the parser on hand-written requests.

  python tests/evaluate.py tests/v2_dev.txt [-v]

A request counts as right when the top answer compiles to exactly the same command as the expected program.
Also reports how often the right answer is among the top 3, and how well confidence separates right from wrong.

File format:  request => stage key=value | stage ... ;; next pipeline      ([brackets] around a value with spaces)
  "+ request => ..."   a follow-up: read with the previous line's expected program as its context
  "! command"          a command the user ran by hand; "+" lines after it follow that
"""
import re
import sys
import time

sys.path.insert(0, ".")
from kural import compile as C
from kural import cli, ir, tok


def parse_gold(text):
    program = []
    for pipe in text.split(";;"):
        stages = []
        for st in pipe.split("|"):
            st = st.strip()
            name = st.split()[0]
            args = {m.group(1): m.group(2).strip("[]") for m in re.finditer(r"(\w+)=(\[[^\]]*\]|\S+)", st[len(name):])}
            for slot in ir.BY_NAME[name].slots:
                args.setdefault(slot.name, None)
            stages.append((name, args))
        program.append(stages)
    return program


def load(path):
    """-> rows of (section, request, expected command, context words or None, last command or None)"""
    section, rows = "", []
    ctx = last = None
    for line in open(path):
        line = line.strip()
        if line.startswith("# ----"):
            section = line.strip("# -").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("! "):
            last = line[2:].strip()
            ctx = ["@raw"] + tok.words(last)[:12]
            continue
        follow = line.startswith("+ ")
        req, gold = line[2 if follow else 0:].split(" => ")
        if not follow:
            ctx = last = None
        if gold.strip() == "DECLINE":
            rows.append((section, req.strip(), None, ctx, last))
            continue
        program = parse_gold(gold)
        want = C.compile_program(program, C.Env(live=False, last=last))
        rows.append((section + (" (follow-up)" if follow and "follow" not in section else ""), req.strip(), want, ctx, last))
        if not any(n in ("again", "last", "watch_last", "time_last") for pipe in program for n, _ in pipe):
            ctx, last = ir.context_words(program)[0], want
    return rows


def main():
    path = sys.argv[1]
    verbose = "-v" in sys.argv
    rows = load(path)
    by, right_conf, wrong_conf, top3 = {}, [], [], 0
    t0 = time.time()
    for section, req, want, ctx, last in rows:
        cands = [(c["command"], c["confidence"]) for c in cli.translate(req, C.Env(live=False, last=last), ctx)]
        ok = (not cands) if want is None else bool(cands) and cands[0][0] == want
        top3 += any(c == want for c, _ in cands)
        s = by.setdefault(section, [0, 0])
        s[0] += ok; s[1] += 1
        (right_conf if ok else wrong_conf).append(cands[0][1] if cands else 0.0)
        if verbose and not ok:
            after = f"   [after: {' '.join(ctx)}]" if ctx else ""
            print(f"WRONG  {req}{after}\n   want: {want or '(decline)'}\n   got:  {cands[0][0] if cands else '(nothing)'}  ({100 * (cands[0][1] if cands else 0):.0f}%)")
    n = len(rows)
    hit = sum(v[0] for v in by.values())
    print(f"\n{path}: {hit}/{n} right ({100 * hit / n:.1f}%), right answer in top 3: {top3}/{n} ({100 * top3 / n:.1f}%), {1000 * (time.time() - t0) / n:.0f} ms per request")
    for sec, (h, t) in by.items():
        print(f"  {sec:<34} {h}/{t}")
    for thr in (0.5, 0.8, 0.9, 0.95):
        kept_r = sum(c >= thr for c in right_conf); kept_w = sum(c >= thr for c in wrong_conf)
        kept = kept_r + kept_w
        print(f"  answers with confidence >= {thr:.2f}: {kept}/{n} kept, {100 * kept_r / max(1, kept):.1f}% of those right, {kept_w} wrong slipped through")


if __name__ == "__main__":
    main()
