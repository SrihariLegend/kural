"""Check that the Rust run time (the `kural` binary) gives exactly what the Python run time gives.

  python tests/compare_runtime.py [-v]

Every request of the test sets, with its session context and previous command, goes through Python
cli.translate(req, Env(live=False, last=last), ctx) and through `kural --batch`. All candidates are compared:
command, risk, reading, and confidence (to 1e-5).
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
import evaluate
from kural import cli
from kural import compile as C

BIN = os.environ.get("KURAL_BIN_RS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "kural")


def rows():
    out = []
    for f in ("v2_dev", "v2_final", "once_outside", "natural_dev", "natural_test", "natural_final"):
        out += [(req, ctx, last) for _, req, _, ctx, last in evaluate.load(f"tests/{f}.txt")]
    out += [(ln.strip(), None, None) for ln in open("tests/outside_v2.txt") if ln.strip() and not ln.startswith("#")]
    return out


def main():
    verbose = "-v" in sys.argv
    rs = rows()
    t = time.time()
    py = [[(c["command"], c["risk"], c["reading"], c["confidence"]) for c in cli.translate(req, C.Env(live=False, last=last), ctx)] for req, ctx, last in rs]
    tpy = time.time() - t
    inp = "".join(json.dumps({"text": req, "context": ctx, "last": last}) + "\n" for req, ctx, last in rs)
    t = time.time()
    p = subprocess.run([BIN, "--batch"], input=inp, capture_output=True, text=True, check=True)
    trs = time.time() - t
    rust = [[(c["command"], c["risk"], c["reading"], c["confidence"]) for c in json.loads(line)] for line in p.stdout.splitlines()]
    assert len(rust) == len(py), (len(rust), len(py))
    same_best = same_all = 0
    for (req, ctx, last), a, b in zip(rs, py, rust):
        best = (bool(a) == bool(b)) and (not a or a[0][:3] == b[0][:3] and abs(a[0][3] - b[0][3]) < 1e-5)
        allc = len(a) == len(b) and all(x[:3] == y[:3] and abs(x[3] - y[3]) < 1e-5 for x, y in zip(a, b))
        same_best += best
        same_all += allc
        if not allc and (verbose or not best):
            print(f"DIFFERENT{'' if best else ' (best)'}: {req!r}" + (f"  [after {' '.join(ctx)}]" if ctx else ""))
            for x in a:
                print("   py:  ", x)
            for y in b:
                print("   rust:", y)
    n = len(rs)
    print(f"{n} requests: same best answer {same_best}/{n}, every candidate identical {same_all}/{n}")
    print(f"total time: Python {tpy:.2f} s, Rust {trs:.2f} s (one process for all requests)")
    return 0 if same_all == n else 1


if __name__ == "__main__":
    sys.exit(main())
