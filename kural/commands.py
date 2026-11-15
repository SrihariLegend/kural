"""Stand-alone commands: one table (kural/lang/commands.txt, shared with the Rust code) that holds, for each command, its arguments, the bash it becomes,
and many ways of asking for it in English.

Format of TABLE:
  @ name  slot:type  slot:type?  !risk        a new command; "?" marks an optional slot; risk 0-2 (default 0)
  $ bash template                             {slot} is replaced by the safely quoted value;
                                              "[ ... ]" is dropped when a slot inside it was not given;
                                              "$$" means the command is built by a function in compile.py
  every other line                            an English phrasing: (a|b) picks one, [a|b] picks one or nothing,
                                              {slot} is where the argument goes

Keeping the three together means adding a command is a few lines in one place.
"""
import os


TABLE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lang", "commands.txt"), encoding="utf-8").read()


def parse_table():
    """-> list of dicts: name, slots [(name, type, optional)], risk, bash (str or None for a function), phrases [str]"""
    out, cur = [], None
    for raw in TABLE.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@ "):
            parts = line[2:].split()
            cur = {"name": parts[0], "slots": [], "risk": 0, "bash": None, "phrases": []}
            for p in parts[1:]:
                if p.startswith("!"):
                    cur["risk"] = int(p[1:])
                else:
                    n, t = p.split(":")
                    cur["slots"].append((n, t.rstrip("?"), t.endswith("?")))
            out.append(cur)
        elif line.startswith("$"):
            cur["bash"] = None if line.strip() == "$$" else line[1:].strip()
        else:
            cur["phrases"].append(line)
    return out


COMMANDS = parse_table()
BY_NAME = {c["name"]: c for c in COMMANDS}


def _first_words(p):
    """Every word a phrasing can start with."""
    p = p.strip()
    if not p:
        return set()
    if p[0] in "([":
        close = {"(": ")", "[": "]"}[p[0]]
        depth, parts, cur = 0, [], ""
        for i, ch in enumerate(p):
            if ch in "([":
                depth += 1
                if depth == 1:
                    continue
            elif ch in ")]":
                depth -= 1
                if depth == 0:
                    parts.append(cur)
                    rest = p[i + 1:]
                    break
            elif ch == "|" and depth == 1:
                parts.append(cur); cur = ""
                continue
            cur += ch
        out = set()
        for part in parts:
            out |= _first_words(part) if part.strip() else _first_words(rest)
        if close == "]":
            out |= _first_words(rest)
        return out
    w = p.split()[0]
    return set() if w.startswith("{") else {w}


def verbs(c):
    """The words a command's phrasings start with, shortened to stems: the request must contain one of them."""
    out = set()
    for p in c["phrases"]:
        for w in _first_words(p):
            w = "".join(ch for ch in w.lower() if ch.isalnum())
            if len(w) >= 2:
                out.add(w[:5])
    return tuple(sorted(out))
