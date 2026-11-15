"""Run the same terminal session through the Python and the Rust run time and compare every answer and the
session files they write: follow-ups, a corrected offer asked again, preferences, preview requests, a change of folder.

  python tests/compare_session.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, ".")
from kural import cli, session

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "kural")

STEPS = [
    ("ask", "find pdf files bigger than 10mb in docs"), ("ask", "only the ones from this week"), ("ask", "how many are there"),
    ("ask", "delete them"), ("ask", "is the nginx service running"), ("ask", "restart it"), ("ask", "show its logs"),
    ("run", "make test", 2), ("ask", "again with sudo"), ("ask", "save that to out.txt"),
    ("ask", "show free disk space"), ("run", "df -h /tmp", 0), ("ask", "show free disk space"),
    ("ask", "use the trash instead of deleting"), ("ask", "delete all log files"), ("ask", "be verbose"), ("ask", "move txt files to backup"),
    ("preview", "delete the tmp files"), ("ask", "list python files"), ("cd", "sub"), ("ask", "delete them"),
    ("ask", "kill firefox"), ("ask", "whats taking all my memory"), ("ask", "make it rain"), ("ask", "search for a package called ripgrep"),
    ("ask", "install it"), ("choose", "find large files", 1),
]


def run_line(path, cmd, status, cwd):
    with open(path, "a") as f:
        f.write(f"run\t{int(time.time())}\t{status}\t{session.esc(cwd)}\t{session.esc(cmd)}\n")


def strip_times(text):
    return re.sub(r"^(ask|run)\t\d+\t", r"\1\tT\t", text, flags=re.M)


def main():
    base = tempfile.mkdtemp(prefix="kural-sess-")
    os.makedirs(os.path.join(base, "docs")); os.makedirs(os.path.join(base, "sub"))
    spy, srs = os.path.join(base, "s.py"), os.path.join(base, "s.rs")
    for p in (spy, srs):
        open(p, "w").write("kural-session\t1\t0\n")
    cwd, bad, n = base, 0, 0
    for step in STEPS:
        kind = step[0]
        if kind == "run":
            for p in (spy, srs):
                run_line(p, step[1], step[2], cwd)
            continue
        if kind == "cd":
            cwd = os.path.join(base, step[1])
            continue
        text = step[1]
        final = kind != "preview"
        choose = step[2] if kind == "choose" else 0
        a = cli.answer(text, cwd=cwd, session_path=spy, final=final, choose=choose)
        args = [BIN, "--hook", "--session", srs, "--cwd", cwd] + (["--preview"] if not final else []) + (["--choose", str(choose)] if choose else []) + ["--", text]
        out = subprocess.run(args, capture_output=True, text=True).stdout.rstrip("\n").split("\x1f")
        b = [(out[1 + 4 * i], int(out[2 + 4 * i]), out[3 + 4 * i], out[4 + 4 * i]) for i in range(int(out[0]))]
        want = [(r["command"], r["risk"], f"{r['confidence']:.3f}", r["reading"]) for r in a]
        n += 1
        ok = want == b
        bad += not ok
        print(("ok   " if ok else "DIFF ") + text + ("" if ok else f"\n   py:   {want}\n   rust: {b}"))
        if a and b:
            print("       ->", b[0][0])
    same_file = strip_times(open(spy).read()) == strip_times(open(srs).read())
    if not same_file:
        print("session files differ:\n--- python\n" + strip_times(open(spy).read()) + "--- rust\n" + strip_times(open(srs).read()))
    shutil.rmtree(base, ignore_errors=True)
    print(f"{n} requests: {n - bad} identical; session files {'identical' if same_file else 'DIFFERENT'}")
    return 0 if not bad and same_file else 1


if __name__ == "__main__":
    sys.exit(main())
