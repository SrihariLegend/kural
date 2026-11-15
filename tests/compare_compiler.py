"""Compile many generated programs with the Python compiler and the Rust one and compare everything.

  python tests/compare_compiler.py [N]        (default 30000 programs from kural/mix.py)

Compares the command (or which kind of failure), risk, reading, the session-context words and the JSON the
session file stores. Uses the offline environment, so nothing depends on this machine.
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, ".")
from kural import compile as C
from kural import ir, mix

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "kural")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    r = random.Random(int(os.environ.get("SEED", "1234")))
    lasts = [None, "ls -la /tmp", "cat notes.txt", "make test"]
    progs = []
    while len(progs) < n:
        words, out, _ = mix.example(r)
        progs.append((ir.delinearise(out, words), r.choice(lasts)))
    want = []
    for prog, last in progs:
        env = C.Env(live=False, last=last)
        try:
            cmd, err = C.compile_program(prog, env), None
        except C.Unsupported:
            cmd, err = None, "unsupported"
        except Exception:
            cmd, err = None, "other"
        reading = C.describe(prog) + "".join(f"  ({x})" for x in env.notes)
        want.append({"command": cmd, "error": err, "risk": C.risk(prog), "reading": reading,
                     "context": ir.context_words(prog)[0], "pyjson": json.dumps(prog, ensure_ascii=False)})
    inp = "".join(json.dumps({"program": p, "last": l}) + "\n" for p, l in progs)
    got = [json.loads(x) for x in subprocess.run([BIN, "--compile-batch"], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()]
    bad = 0
    for (prog, last), w, g in zip(progs, want, got):
        diff = [k for k in w if w[k] != g.get(k)]
        if diff:
            bad += 1
            if bad <= 12:
                print("DIFFERENT", diff, json.dumps(prog))
                for k in diff:
                    print(f"   {k}: py  {w[k]!r}\n   {' ' * len(k)}  rs  {g.get(k)!r}")
    print(f"{n} generated programs: {n - bad} identical, {bad} different")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
