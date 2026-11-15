"""Score every test set and print one table.

  python tests/report.py [--final]      --final also scores tests/v2_final.txt (do that once, after all tuning is done)
"""
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
import evaluate
from kural import cli, run
from kural import compile as C


def score(path):
    rows = evaluate.load(path)
    hit = follow = follow_hit = 0
    for section, req, want, ctx, last in rows:
        cands = cli.translate(req, C.Env(live=False, last=last), ctx)
        ok = (not cands) if want is None else bool(cands) and cands[0]["command"] == want
        hit += ok
        if ctx:
            follow += 1; follow_hit += ok
    return hit, len(rows), follow_hit, follow


def declined(path):
    reqs = [ln.strip() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    return sum(not cli.translate(r, C.Env(live=False)) for r in reqs), len(reqs)


def main():
    run.weights()
    sets = [("tests/v2_dev.txt", "new abilities, used to find gaps"), ("tests/once_outside.txt", "written before 0.2 existed, as things 0.1 could not do"),
            ("tests/natural_dev.txt", "0.1 abilities, used for tuning"), ("tests/natural_test.txt", "0.1 abilities, used for tuning"),
            ("tests/natural_final.txt", "0.1 abilities, not used for tuning")]
    if "--final" in sys.argv:
        sets.insert(0, ("tests/v2_final.txt", "new abilities, scored once, never used for tuning"))
    print("| Set | Requests | Right | Note |\n|---|---|---|---|")
    for path, note in sets:
        hit, n, fh, fn = score(path)
        extra = f"; follow-ups {fh}/{fn}" if fn else ""
        print(f"| `{path}` | {n} | {hit} ({100 * hit / n:.1f}%) | {note}{extra} |")
    d, n = declined("tests/outside_v2.txt")
    print(f"| `tests/outside_v2.txt` | {n} | {d} declined ({100 * d / n:.1f}%) | requests it is not built for |")
    reqs = ["delete log files older than 30 days in ~/Downloads", "restart nginx", "what's using port 8080", "show the 5 biggest files in /var/log",
            "replace foo with bar in config.txt", "how many lines of code are in src", "commit everything with message wip and push"]
    t = time.time()
    for _ in range(5):
        for r in reqs:
            cli.translate(r, C.Env(live=False))
    per = 1000 * (time.time() - t) / (5 * len(reqs))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t = time.time()
    subprocess.run([sys.executable, "-m", "kural.cli", "restart nginx"], capture_output=True, env=dict(os.environ, PYTHONPATH="."))
    cold = 1000 * (time.time() - t)
    w = os.path.getsize("kural/weights.npz") / 1e6
    print(f"\nparsing: {per:.0f} ms per request once loaded; one-off call from the shell: {cold:.0f} ms; peak memory of this process: {rss:.0f} MB; weights file: {w:.1f} MB")


if __name__ == "__main__":
    main()
