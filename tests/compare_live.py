"""Compare the Python and Rust compilers when they look at a real folder: name matching (case, missing extension,
typos, several candidates), commands that create things, project detection, installed tools, preferences.

  python tests/compare_live.py
"""
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, ".")
from kural import compile as C

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "kural")

FOLDERS = {
    "rust": ["Cargo.toml", "README.md", "notes.txt", "config.yaml", "Documents-old/", "src/main.rs", "report.PDF", "data.csv",
             "data.json", "My Photos/a.jpg", "photos/b.jpg", ".venv/bin/activate", "script.sh", "archive.tar.gz"],
    "node": ["package.json", "readme.txt", "notes.md", "Notes.txt", "src/", "build/", "build.log", "builder.py"],
    "plain": ["Makefile", "a.txt", "b.txt", "todo", "TODO.md"],
}
P = lambda *st: [list(st)]
F = lambda **kw: ("files", {"what": "anyfile", "in": None, **kw})
PROGRAMS = [
    P(("view", {"file": "readme"})), P(("view", {"file": "the makefile"})), P(("view", {"file": "src"})), P(("view", {"file": "confg.yaml"})),
    P(("view", {"file": "notes"})), P(("view", {"file": "todo"})), P(("view", {"file": "dat"})), P(("view", {"file": "data"})),
    P(("ls", {"path": "notes.txt"})), P(("ls", {"path": "documents-old"})), P(("ls", {"path": "photos"})), P(("ls", {"path": "my photos"})),
    P(("edit", {"file": "confg.yaml"})), P(("edit", {"file": "new.txt"})), P(("cd", {"path": "documents-old"})), P(("cd", {"path": "SRC"})),
    P(("proj_test", {})), P(("proj_build", {})), P(("proj_run", {})), P(("open", {"target": "firefox"})), P(("open", {"target": "notes.txt"})),
    P(("open", {"target": "report.pdf"})), P(("touch", {"file": "readme.md"})), P(("mkdir", {"path": "src2"})),
    P(("paths", {"names": "src"}), ("delete", {})), P(("paths", {"names": "readme.md"}), ("delete", {})), P(("paths", {"names": "notes2.txt"}), ("delete", {})),
    P(("rename", {"old": "Notes.txt", "new": "readme.md"})), P(("lines", {"of": "note.txt"}), ("match", {"text": "x"}), ("save", {"to": "readme"})),
    P(("lines", {"of": "note.txt"}), ("match", {"text": "x"}), ("lshow", {})), P(("run_file", {"file": "script"})), P(("run_file", {"file": "builder"})),
    P(("venv_on", {"path": None})), P(("backup", {"file": "config"})), P(("extract", {"archive": "archive", "into": "out"})),
    P(F(), ("ext", {"ext": "log"}), ("delete", {})), P(F(in_="x"), ("fshow", {})) if False else P(F(**{"in": "documents-old"}), ("fshow", {})),
    P(F(**{"in": "Src"}), ("ext", {"ext": "rs"}), ("fgrep", {"text": "main"})), P(F(), ("ext", {"ext": "txt"}), ("move", {"to": "documents-old"})),
    P(("paths", {"names": "a.txt and b.txt"}), ("move", {"to": "docs"})), P(("paths", {"names": "report.pdf"}), ("move", {"to": "final.pdf"})),
    P(("paths", {"names": "readme"}), ("copy", {"to": "build"})), P(("paths", {"names": "notes.txt"}), ("chmod", {"mode": "read only"})),
    P(("download", {"url": "https://x.org/a.zip", "to": "src"})), P(("download", {"url": "https://x.org/a.zip", "to": "out.zip"})),
    P(("procs", {}), ("pname", {"name": "NetworkManager"}), ("kill", {"force": "soft"})), P(("procs", {}), ("pname", {"name": "firefox"}), ("kill", {"force": "soft"})),
    P(("clipboard", {}), ("lcount", {})), P(("lines", {"of": "data.csv"}), ("clip", {})), P(("tree", {"path": None, "depth": "2"})),
    P(("dns", {"host": "example.com"})), P(("port", {"n": "8080"}), ("kill", {"force": "hard"})), P(("paths", {"names": "src"}), ("trash", {})),
    P(("pkg_owner", {"file": "/usr/bin/ls"})), P(("pkg_owner", {"file": "notes.txt"})), P(("pkg_install", {"names": "htop and vim"})),
    [[("mkdir", {"path": "newdir"})], [F(), ("ext", {"ext": "txt"}), ("move", {"to": "newdir"})]],
    [[("mkdir", {"path": "documents-oldx"})], [F(), ("ext", {"ext": "txt"}), ("copy", {"to": "documents-oldx"})]],
    P(F(), ("ext", {"ext": "txt"}), ("delete", {})), P(F(), ("ext", {"ext": "txt"}), ("fsort", {"key": "size", "order": "desc"}), ("delete", {})),
    P(F(), ("ext", {"ext": "jpg"}), ("move", {"to": "photos"})), P(F(), ("ext", {"ext": "jpg"}), ("rn_lower", {})),
]
PREFS = [[], ["trash"], ["confirm", "verbose"], ["trash", "confirm", "verbose"]]


def make(kind):
    d = tempfile.mkdtemp(prefix="kural-live-")
    for name in FOLDERS[kind]:
        p = os.path.join(d, name)
        if name.endswith("/"):
            os.makedirs(p, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()
    return d


def main():
    total = bad = 0
    for kind in FOLDERS:
        d = make(kind)
        for prog, prefs in itertools.product(PROGRAMS, PREFS):
            env = C.Env(cwd=d, live=True, prefs={k: True for k in prefs}, last="ls")
            try:
                want, werr = C.compile_program(prog, env), None
            except Exception as e:
                want, werr = None, type(e).__name__
            p = subprocess.run([BIN, "--compile-json", json.dumps(prog), "--live", "--cwd", d, "--last", "ls", "--prefs", ",".join(prefs)],
                               capture_output=True, text=True)
            got = p.stdout.rstrip("\n") if p.returncode == 0 else None
            notes = [l[len("notes: "):] for l in p.stderr.splitlines() if l.startswith("notes: ")]
            got_notes = notes[0].split("; ") if notes else []
            total += 1
            if got != want or (want is not None and got_notes != env.notes):
                bad += 1
                print(f"DIFFERENT [{kind} {prefs}] {json.dumps(prog)}\n   py:  {want!r} {env.notes} {werr or ''}\n   rs:  {got!r} {got_notes} {p.stderr.strip()[:80]}")
        shutil.rmtree(d, ignore_errors=True)
    print(f"{total} live compilations: {total - bad} identical, {bad} different")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
