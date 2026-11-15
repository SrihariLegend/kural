"""Run the commands kural generates, for real, in a scratch folder, and check what they did.

  python tests/run_commands.py

This tests the compiler (program -> bash), not the parser: every program here is written by hand.
File names include spaces, quotes, a leading dash and upper case, because that is where shell commands break.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, ".")
from kural import compile as C

FILES = {
    "a.txt": "apple\nbanana\napple\n\ncherry\n", "B.TXT": "TODO one\nfoo bar\n", "notes one.txt": "foo\nfoo\nbaz\n",
    "it's (1).txt": "x\n", "-dash.txt": "dash\n", "data.csv": "name,qty\nnut,3\nbolt,4\n", "nums.txt": "5\n10\n20\n",
    "IMG 01.JPG": "j1", "img02.jpg": "j2", "src/main.py": "import os\n# TODO fix\nprint(1)\n", "src/util.py": "def f():\n    pass\n",
    "src/deep/x.py": "x = 1\n", ".hidden": "h\n", "empty.log": "", "old.log": "old\n", "big.bin": "x" * 5000, "dup1.txt": "same\n",
    "dup2.txt": "same\n", "draft-report.md": "draft\n",
}


def fresh():
    d = tempfile.mkdtemp(prefix="kural-run-")
    for name, text in FILES.items():
        p = os.path.join(d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
    old = time.time() - 40 * 86400
    os.utime(os.path.join(d, "old.log"), (old, old))
    return d


def tree(d):
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.relpath(os.path.join(root, f), d) for f in files]
    return sorted(out)


F = lambda **kw: ("files", {"what": "anyfile", "in": None, **kw})
CHECKS = [
    # (name, program, check(stdout, folder) -> bool)
    ("rename to lowercase", [[F(), ("flat", {}), ("ext", {"ext": "jpg"}), ("rn_lower", {})]],
     lambda o, d: {"img 01.jpg", "img02.jpg"} <= set(tree(d)) and "IMG 01.JPG" not in tree(d)),
    ("rename spaces to underscores", [[F(), ("flat", {}), ("rn_spaces", {})]],
     lambda o, d: {"notes_one.txt", "IMG_01.JPG", "it's_(1).txt"} <= set(tree(d))),
    ("rename replacing text", [[F(), ("rn_replace", {"old": "draft", "new": "final"})]], lambda o, d: "final-report.md" in tree(d)),
    ("rename with prefix", [[F(), ("ext", {"ext": "csv"}), ("rn_prefix", {"text": "old_"})]], lambda o, d: "old_data.csv" in tree(d)),
    ("rename with suffix before the extension", [[F(), ("ext", {"ext": "csv"}), ("rn_suffix", {"text": "_v2"})]], lambda o, d: "data_v2.csv" in tree(d)),
    ("change extension", [[F(), ("ext", {"ext": "md"}), ("rn_ext", {"ext": "txt"})]], lambda o, d: "draft-report.txt" in tree(d)),
    ("replace text inside files", [[F(), ("ext", {"ext": "txt"}), ("replace", {"old": "foo", "new": "qux"})]],
     lambda o, d: open(f"{d}/notes one.txt").read() == "qux\nqux\nbaz\n" and "qux bar" in open(f"{d}/B.TXT").read()),
    ("delete lines containing a word, in place", [[("lines", {"of": "a.txt"}), ("nomatch", {"text": "apple"}), ("inplace", {})]],
     lambda o, d: open(f"{d}/a.txt").read() == "banana\n\ncherry\n"),
    ("keep only matching lines, in place", [[("lines", {"of": "a.txt"}), ("match", {"text": "an"}), ("inplace", {})]],
     lambda o, d: open(f"{d}/a.txt").read() == "banana\n"),
    ("replace in one file, in place", [[("lines", {"of": "notes one.txt"}), ("lreplace", {"old": "foo", "new": "a/b&c"}), ("inplace", {})]],
     lambda o, d: open(f"{d}/notes one.txt").read() == "a/b&c\na/b&c\nbaz\n"),
    ("sort and dedupe in place", [[("lines", {"of": "a.txt"}), ("lsort", {"key": "alpha", "order": "asc"}), ("uniq", {}), ("inplace", {})]],
     lambda o, d: open(f"{d}/a.txt").read() == "\napple\nbanana\ncherry\n"),
    ("dedupe in place keeps order", [[("lines", {"of": "a.txt"}), ("uniq", {}), ("inplace", {})]],
     lambda o, d: open(f"{d}/a.txt").read() == "apple\nbanana\n\ncherry\n"),
    ("delete line 2 in place", [[("lines", {"of": "nums.txt"}), ("ldel", {"n": "2"}), ("inplace", {})]], lambda o, d: open(f"{d}/nums.txt").read() == "5\n20\n"),
    ("delete the last line in place", [[("lines", {"of": "nums.txt"}), ("ldel", {"n": "last"}), ("inplace", {})]], lambda o, d: open(f"{d}/nums.txt").read() == "5\n10\n"),
    ("drop blank lines in place", [[("lines", {"of": "a.txt"}), ("lblank", {}), ("inplace", {})]], lambda o, d: "\n\n" not in open(f"{d}/a.txt").read()),
    ("lines 2 to 3", [[("lines", {"of": "a.txt"}), ("lrange", {"a": "2", "b": "3"}), ("lshow", {})]], lambda o, d: o == "banana\napple\n"),
    ("lines given as 2-3", [[("lines", {"of": "a.txt"}), ("lrange", {"a": "2-3", "b": "2-3"}), ("lshow", {})]], lambda o, d: o == "banana\napple\n"),
    ("one line by number", [[("lines", {"of": "a.txt"}), ("lline", {"n": "5"}), ("lshow", {})]], lambda o, d: o == "cherry\n"),
    ("second column of a csv", [[("lines", {"of": "data.csv"}), ("col", {"n": "second", "sep": "comma"}), ("lshow", {})]], lambda o, d: o == "qty\n3\n4\n"),
    ("add up numbers", [[("lines", {"of": "nums.txt"}), ("lsum", {}), ("lshow", {})]], lambda o, d: o.strip() == "35"),
    ("most common lines", [[("lines", {"of": "a.txt"}), ("lfreq", {}), ("ltake", {"n": "1"}), ("lshow", {})]], lambda o, d: o.split() == ["2", "apple"]),
    ("count words", [[("lines", {"of": "B.TXT"}), ("wcount", {})]], lambda o, d: o.strip() == "4"),
    ("upper case to a new file", [[("lines", {"of": "nums.txt"}), ("lupper", {}), ("save", {"to": "out file.txt"})]], lambda o, d: open(f"{d}/out file.txt").read() == "5\n10\n20\n"),
    ("count matching lines in a file with a leading dash", [[("lines", {"of": "./-dash.txt"}), ("match", {"text": "dash"}), ("lcount", {})]], lambda o, d: o.strip() == "1"),
    ("delete files older than 30 days", [[F(), ("ext", {"ext": "log"}), ("older", {"age": "30 days"}), ("delete", {})]],
     lambda o, d: "old.log" not in tree(d) and "empty.log" in tree(d)),
    ("list files that are not txt, top level", [[F(), ("flat", {}), ("not", {}), ("ext", {"ext": "txt"}), ("fcount", {})]], lambda o, d: o.strip() == "8"),
    ("hidden files", [[F(), ("hidden", {}), ("fshow", {})]], lambda o, d: o.strip() == "./.hidden"),
    ("empty files", [[F(), ("empty", {}), ("fshow", {})]], lambda o, d: o.strip() == "./empty.log"),
    ("files containing a word", [[F(), ("contains", {"text": "TODO"}), ("fcount", {})]], lambda o, d: o.strip() == "2"),
    ("python files not containing a word", [[F(), ("ext", {"ext": "python"}), ("not", {}), ("contains", {"text": "TODO"}), ("fcount", {})]], lambda o, d: o.strip() == "2"),
    ("search file contents", [[F(), ("ext", {"ext": "python"}), ("fgrep", {"text": "todo"})]], lambda o, d: "main.py:2:# TODO fix" in o),
    ("depth limit", [[F(), ("depth", {"n": "2"}), ("ext", {"ext": "py"}), ("fcount", {})]], lambda o, d: o.strip() == "2"),
    ("lines of code", [[F(), ("ext", {"ext": "py"}), ("fcat", {}), ("lcount", {})]], lambda o, d: o.strip() == "6"),
    ("count by extension", [[F(), ("by_ext", {})]], lambda o, d: "8 txt" in " ".join(o.split()) and "hidden" not in o),
    ("duplicates", [[F(), ("ext", {"ext": "txt"}), ("dupes", {})]], lambda o, d: "dup1.txt" in o and "dup2.txt" in o and "a.txt" not in o),
    ("save a list of names", [[F(), ("ext", {"ext": "py"}), ("fnames", {}), ("lsort", {"key": "alpha", "order": "asc"}), ("save", {"to": "list.txt"})]],
     lambda o, d: open(f"{d}/list.txt").read().split() == ["./src/deep/x.py", "./src/main.py", "./src/util.py"]),
    ("the second biggest file", [[F(), ("fsort", {"key": "size", "order": "desc"}), ("fpick", {"n": "second"}), ("fnames", {}), ("lshow", {})]],
     lambda o, d: o.strip() == "./src/main.py"),
    ("biggest file with sizes", [[F(), ("fsort", {"key": "size", "order": "desc"}), ("ftake", {"n": "1"}), ("fshow", {})]], lambda o, d: "big.bin" in o),
    ("move files with odd names into a folder", [[("mkdir", {"path": "dest box"})], [F(), ("flat", {}), ("ext", {"ext": "txt"}), ("move", {"to": "dest box"})]],
     lambda o, d: {"dest box/it's (1).txt", "dest box/-dash.txt", "dest box/notes one.txt"} <= set(tree(d))),
    ("copy named files", [[("mkdir", {"path": "bk"})], [("paths", {"names": "a.txt and data.csv"}), ("copy", {"to": "bk"})]],
     lambda o, d: {"bk/a.txt", "bk/data.csv", "a.txt"} <= set(tree(d))),
    ("delete a named file with a leading dash", [[("paths", {"names": "-dash.txt"}), ("delete", {})]], lambda o, d: "-dash.txt" not in tree(d)),
    ("rename one file", [[("rename", {"old": "notes one.txt", "new": "notes.txt"})]], lambda o, d: "notes.txt" in tree(d)),
    ("zip then list the archive", [[F(), ("ext", {"ext": "py"}), ("compress", {"fmt": "zip", "to": "code.zip"})], [("archive_list", {"archive": "code.zip"})]],
     lambda o, d: "src/main.py" in o),
    ("tarball of a folder then extract elsewhere", [[("paths", {"names": "src"}), ("compress", {"fmt": "tgz", "to": "src.tar.gz"})], [("mkdir", {"path": "out"})],
                                                    [("extract", {"archive": "src.tar.gz", "into": "out"})]], lambda o, d: "out/src/deep/x.py" in tree(d)),
    ("make read only", [[("paths", {"names": "a.txt"}), ("chmod", {"mode": "read only"})]], lambda o, d: not os.access(f"{d}/a.txt", os.W_OK) or os.getuid() == 0),
    ("back up a file", [[("backup", {"file": "a.txt"})]], lambda o, d: "a.txt.bak" in tree(d)),
    ("write then append", [[("write", {"text": "one", "file": "w.txt"})], [("append", {"text": "it's two", "file": "w.txt"})]],
     lambda o, d: open(f"{d}/w.txt").read() == "one\nit's two\n"),
    ("symlink", [[("link", {"target": "a.txt", "name": "link to a"})]], lambda o, d: os.path.islink(f"{d}/link to a")),
    ("total size", [[F(), ("ext", {"ext": "bin"}), ("total", {})]], lambda o, d: "total" in o),
    ("calculator", [[("calc", {"expr": "15 percent of 240"})]], lambda o, d: o.strip().startswith("36")),
    ("line 2 of the previous output", [[("last", {}), ("lline", {"n": "2"}), ("lshow", {})]], lambda o, d: o.strip() == "10"),
]


def main():
    fails = 0
    for name, program, check in CHECKS:
        d = fresh()
        env = C.Env(cwd=d, live=True, last="cat nums.txt")
        try:
            cmd = C.compile_program(program, env)
            p = subprocess.run(["bash", "-c", cmd], cwd=d, capture_output=True, text=True, timeout=20)
            ok = bool(check(p.stdout, d))
        except Exception as e:
            cmd, ok, p = f"(error: {e!r})", False, None
        if not ok:
            fails += 1
            print(f"FAIL  {name}\n      {cmd}")
            if p:
                print(f"      exit {p.returncode}  stdout={p.stdout[:200]!r}  stderr={p.stderr[:200]!r}")
        else:
            print(f"ok    {name}")
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(CHECKS) - fails}/{len(CHECKS)} commands did what they should")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
