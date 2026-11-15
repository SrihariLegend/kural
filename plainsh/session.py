"""The session file: one throwaway text file per terminal session.

It holds what happened in this terminal, one event per line, tab-separated:

  run   <time>  <exit status>  <folder>  <command>      every command that was run (written by the shell hook)
  ask   <time>  <folder>  <request>                     a plainsh request that was accepted
  got   <program as JSON>                               how plainsh read it
  cmd   <command>                                       the command it offered
  fix   <request>  <offered>  <command actually run>    the user edited the offer before running it
  pref  <key>  <on|off>                                 a preference set during the session

Nothing else is remembered between requests, and the file is gone when the session's runtime folder is cleaned.
"""
import difflib
import json
import os
import time

from . import ir, tok

MAX_RAW_WORDS = 12


def runtime_dir():
    base = os.environ.get("XDG_RUNTIME_DIR")
    d = os.path.join(base, "plainsh") if base and os.path.isdir(base) else f"/tmp/plainsh-{os.getuid()}"
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"t": "\t", "n": "\n", "\\": "\\"}.get(s[i + 1], s[i + 1])); i += 2
        else:
            out.append(s[i]); i += 1
    return "".join(out)


def norm(request):
    return " ".join(tok.words(request.lower()))


class State:
    """What the next request may build on."""

    def __init__(self):
        self.context = None      # words describing the previous request or command (see ir.context_words)
        self.last = None         # the last command line that was run
        self.prefs = {}
        self.fixes = {}          # normalised request -> the command the user ran instead of the one offered
        self.asks = []           # earlier requests, oldest first


def _config_prefs():
    prefs = {}
    path = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "plainsh", "config")
    try:
        for line in open(path):
            k, _, v = line.strip().partition("=")
            if k in ir.ENUMS["prefkey"]:
                prefs[k] = v.strip().lower() in ("on", "1", "true", "yes")
    except OSError:
        pass
    return prefs


def read(path, cwd=None):
    st = State()
    st.prefs = _config_prefs()
    if not path:
        return st
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    except OSError:
        return st
    offer = None           # (program, command, folder) of the latest accepted request
    pending = {}
    newest = None          # "offer" or "run": which came last
    run = None             # (command, folder)
    for line in lines:
        f = line.split("\t")
        kind = f[0]
        if kind == "run" and len(f) >= 5:
            run = (unesc(f[4]), unesc(f[3]))
            newest = "run"
        elif kind == "ask" and len(f) >= 4:
            pending = {"cwd": unesc(f[2]), "request": unesc(f[3])}
            st.asks.append(pending["request"])
        elif kind == "got" and len(f) >= 2:
            try:
                pending["program"] = [[(n, a) for n, a in pipe] for pipe in json.loads(f[1])]
            except ValueError:
                pending.pop("program", None)
        elif kind == "cmd" and len(f) >= 2 and "program" in pending:
            offer = (pending["program"], unesc(f[1]), pending.get("cwd"))
            newest = "offer"
        elif kind == "fix" and len(f) >= 4:
            offered, ran = unesc(f[2]), unesc(f[3])
            same_tool = offered.split()[:1] == ran.split()[:1]
            if same_tool or difflib.SequenceMatcher(None, offered, ran).ratio() >= 0.5:
                st.fixes[norm(unesc(f[1]))] = ran
        elif kind == "pref" and len(f) >= 3 and f[1] in ir.ENUMS["prefkey"]:
            st.prefs[f[1]] = f[2] == "on"
    if run:
        st.last = run[0]
    use_program = offer is not None and (newest == "offer" or (run and run[0] == offer[1]))
    if use_program and cwd and offer[2] and os.path.normpath(cwd) != os.path.normpath(offer[2]):
        use_program = False          # the folder changed: relative names in the old request no longer mean the same
    if use_program and not any(n in ("unknown", "pref") for pipe in offer[0] for n, _ in pipe):
        words, _ = ir.context_words(offer[0])
        if len(words) <= 36:
            st.context = words
    if st.context is None and run:
        st.context = ["@raw"] + tok.words(run[0])[:MAX_RAW_WORDS]
    return st


def _append(path, lines):
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))


def record(path, request, cwd, program, command):
    if not path:
        return
    try:
        _append(path, [f"ask\t{int(time.time())}\t{esc(cwd or '')}\t{esc(request)}",
                       "got\t" + json.dumps(program, ensure_ascii=False), f"cmd\t{esc(command)}"])
    except OSError:
        pass


def set_pref(path, key, state):
    if not path:
        return
    try:
        _append(path, [f"pref\t{key}\t{state}"])
    except OSError:
        pass
