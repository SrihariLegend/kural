"""Check that the Rust engine and the NumPy reference give the same readings for every test request.

  python tests/compare_engines.py
"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
import evaluate
from kural import native, run

assert native.available(), "build the engine first: cargo build --release --manifest-path engine/Cargo.toml"
P = run.weights()
rows = []
for f in ("v2_dev", "v2_final", "once_outside", "natural_dev", "natural_test", "natural_final"):
    rows += [(req, ctx) for _, req, _, ctx, _ in evaluate.load(f"tests/{f}.txt")]
rows += [(ln.strip(), None) for ln in open("tests/outside_v2.txt") if ln.strip() and not ln.startswith("#")]
same_top = same_all = 0
worst = 0.0
tn = tr = 0.0
for req, ctx in rows:
    t = time.perf_counter(); a = run.parse(req, ctx, P=P); tn += time.perf_counter() - t
    t = time.perf_counter(); b = native.parse(req, ctx); tr += time.perf_counter() - t
    same_top += bool(a) and bool(b) and a[0][0] == b[0][0]
    same_all += [x[0] for x in a] == [x[0] for x in b]
    for x, y in zip(a, b):
        if x[0] == y[0]:
            worst = max(worst, abs(x[1] - y[1]))
    if not (a and b and a[0][0] == b[0][0]):
        print("DIFFERENT:", req, "\n   numpy:", a[0][0] if a else None, "\n   rust: ", b[0][0] if b else None)
n = len(rows)
print(f"{n} requests: same best reading {same_top}/{n}, same top-3 list {same_all}/{n}, largest probability difference {worst:.2e}")
print(f"time per request: NumPy {1000 * tn / n:.2f} ms, Rust {1000 * tr / n:.2f} ms (both include their own word splitting)")
