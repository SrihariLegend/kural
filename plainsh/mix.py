"""Whole training requests: every kind of request plainsh understands, mixed in the proportions used for
training, plus follow-ups that build on the previous request ("delete them", "same but in ~/src", "go there").

gen.py builds file, line and process pipelines. This module adds the stand-alone commands (from the
table in commands.py), requests about named files, the extra line and process requests, follow-ups,
and the requests plainsh should decline.
"""
import random

from . import commands, gen, ir, tok
from .gen import JOIN, POLITE, TAILS, Req, emit, finish, put, val

# --------------------------------------------------------------------------- phrase templates
def parse_template(s):
    """'(a|b) [c] {slot}' -> nested nodes: ("lit", text) | ("slot", name) | ("alt", [sequences], optional)"""
    pos = 0

    def seq(stop):
        nonlocal pos
        alts, cur, lit = [], [], []

        def flush():
            t = "".join(lit).strip()
            if t:
                cur.append(("lit", t))
            lit.clear()

        while pos < len(s):
            c = s[pos]
            if stop and c == stop:
                break
            if c == "|" and stop:
                flush(); alts.append(cur); cur = []; pos += 1
            elif c in "([":
                flush(); pos += 1
                inner = seq(")" if c == "(" else "]"); pos += 1
                cur.append(("alt", inner, c == "["))
            elif c == "{":
                flush(); j = s.index("}", pos)
                cur.append(("slot", s[pos + 1:j])); pos = j + 1
            else:
                lit.append(c); pos += 1
        flush(); alts.append(cur)
        return alts

    return seq("")[0]


def flatten(r, nodes, out):
    for node in nodes:
        if node[0] == "alt":
            if node[2] and r.random() < 0.5:
                continue
            flatten(r, r.choice(node[1]), out)
        else:
            out.append(node)
    return out


TEMPLATES = {c["name"]: [parse_template(p) for p in c["phrases"]] for c in commands.COMMANDS}
# which kind of value to sample for a slot, where the table's type is too broad
KIND = {
    ("pkg_info", "name"): "pkg", ("pkg_has", "name"): "pkg", ("pkg_files", "name"): "pkg", ("pkg_search", "term"): "pkg",
    ("pkg_owner", "file"): "binfile", ("git_tag", "name"): "tag", ("alias_make", "name"): "alias", ("alias_make", "cmd"): "aliascmd",
    ("var_set", "value"): "varvalue", ("write", "text"): "sentence", ("append", "text"): "sentence", ("write", "file"): "textfile",
    ("append", "file"): "textfile", ("run_file", "file"): "script_file", ("open", "target"): "target", ("path_add", "dir"): "bindir",
    ("venv_make", "path"): "venv", ("venv_on", "path"): "venv", ("tree", "depth"): "smallcount", ("extract", "archive"): "archive",
    ("archive_list", "archive"): "archive", ("view", "file"): "textfile", ("page", "file"): "textfile", ("edit", "file"): "textfile",
    ("follow", "file"): "textfile", ("make_exec", "file"): "script_file", ("dk_build", "tag"): "image", ("git_commit", "msg"): "msg",
    ("git_commit_all", "msg"): "msg", ("watch_last", "n"): "smallcount", ("git_blame", "file"): "textfile",
    ("git_log_file", "file"): "textfile", ("git_add", "file"): "textfile", ("git_discard_file", "file"): "textfile",
    ("diff", "a"): "textfile", ("diff", "b"): "textfile",
}
WEIGHT = {"pkg_install": 3, "pkg_remove": 2, "cd": 2.5, "mkdir": 2.5, "view": 2.5, "git_commit": 2, "git_checkout": 1.5, "extract": 2,
          "download": 2, "ssh": 1.5, "edit": 2, "open": 2, "rename": 2.5, "ls": 2, "sys_update": 1.5, "disk_free": 1.5, "mem_free": 1.5,
          "svc_restart": 1.5, "svc_status": 1.5, "svc_start": 1.5, "svc_stop": 1.5, "again": 1.5, "git_status": 1.5, "follow": 1.3,
          "touch": 1.5, "append": 1.5, "write": 1.3, "diff": 1.3, "git_log": 1.3, "ping": 1.3, "man": 1.3, "run_file": 1.5}
_NAMES = [c["name"] for c in commands.COMMANDS]
_WTS = [WEIGHT.get(n, 1.0) for n in _NAMES]
_FOLDER_NOUNS = ("folder", "directory", "dir")


def say(q, name, only=None, pronoun=None):
    """Write one phrasing of a table command into q. Returns its slot keys, or None if a required slot was left out.
    pronoun: {slot: text} writes that text instead of a value (the slot is then filled from the context)."""
    r = q.r
    c = commands.BY_NAME[name]
    nodes = flatten(r, r.choice(only or TEMPLATES[name]), [])
    keys, texts = {}, {}
    for i, node in enumerate(nodes):
        if node[0] == "lit":
            q.lit(node[1])
            continue
        slot = node[1]
        if slot in texts:                      # mentioned twice ("copy x to x.bak"): repeat the same words
            q.lit(texts[slot])
            continue
        if pronoun and slot in pronoun:
            q.lit(pronoun[slot]); texts[slot] = pronoun[slot]
            continue
        kind = KIND.get((name, slot)) or next(t for n, t, _ in c["slots"] if n == slot)
        text, quote, pre, post = val(q, kind)
        before = nodes[i - 1][1].split()[-1].lower() if i and nodes[i - 1][0] == "lit" else ""
        after = nodes[i + 1][1].split()[0].lower() if i + 1 < len(nodes) and nodes[i + 1][0] == "lit" else ""
        if pre and before not in ("the", "my", "a", "called", "named", "to", "in", "into", "at"):
            q.lit(pre)
        elif pre and before in ("to", "in", "into", "at") and after not in _FOLDER_NOUNS:
            q.lit(pre)
        k = q.key()
        q.chunks.append((f'"{text}"' if quote else text, k))
        if post and after not in _FOLDER_NOUNS and (pre or before in ("the", "my")):
            q.lit(post)
        keys[slot], texts[slot] = k, text
    for n, _, optional in c["slots"]:
        if n not in keys and not optional and not (pronoun and n in pronoun):
            return None
    return {n: keys.get(n) for n, _, _ in c["slots"]}


def command_request(q, name=None):
    r = q.r
    name = name or r.choices(_NAMES, _WTS)[0]
    if r.random() < 0.22:
        q.lit(r.choice(POLITE))
    mark = len(q.chunks)
    if name == "var_set" and r.random() < 0.35:            # "export FOO=bar": one word holds both parts
        k = q.key()
        q.lit(r.choice(["export", "set", "", "export", "define"]))
        q.chunks.append((f"{val(q, 'envname')[0]}={val(q, 'varvalue')[0]}", k))
        return [("var_set", {"name": k, "value": k})]
    if name == "alias_make" and r.random() < 0.4:
        a, cmd = val(q, "alias")[0], val(q, "aliascmd")[0]
        q.lit("alias")
        if " " in cmd or r.random() < 0.5:
            k1, k2 = q.key(), q.key()
            q.chunks.append((a + "=", k1)); q.chunks.append((f"'{cmd}'", k2))
            return [("alias_make", {"name": k1, "cmd": k2})]
        k = q.key()
        q.chunks.append((f"{a}={cmd}", k))
        return [("alias_make", {"name": k, "cmd": k})]
    for _ in range(8):
        del q.chunks[mark:]
        args = say(q, name)
        if args is not None:
            break
    else:
        raise RuntimeError(f"no phrasing of {name} has all its required slots")
    if r.random() < 0.08:
        q.lit(r.choice(TAILS))
    return [(name, args)]


# --------------------------------------------------------------------------- files the user names outright
def paths_request(q):
    r = q.r
    folder = r.random() < 0.3
    n = r.choice([1, 1, 1, 1, 2, 2, 3])
    names = [(r.choice(gen.REAL_DIRS + gen.DIR_WORDS) if folder else q.filename()) for _ in range(n)]
    text = names[0] if n == 1 else gen.pkg_list(r, names)
    lead = r.choice(["", "", "", "the", "the folder", "the directory", "my"] if folder else
                    ["", "", "", "the file", "file", "my", "the files" if n > 1 else "the file"])
    sink = r.choices(["delete", "trash", "move", "copy", "compress", "sizes", "chmod", "chown", "upload"], [30, 7, 16, 14, 8, 9, 6, 4, 6])[0]
    if r.random() < 0.22:
        q.lit(r.choice(POLITE))
    k = None

    def target():
        nonlocal k
        q.lit(lead)
        k = q.key()
        q.chunks.append((text, k))

    args = {}
    if sink == "delete":
        q.lit(r.choice(gen.DELETE)); target()
    elif sink == "trash":
        form = r.randrange(4)
        q.lit(["trash", "move", "put", "send"][form]); target()
        q.lit(["", "to the trash", "in the bin", "to the trash"][form])
    elif sink in ("move", "copy"):
        q.lit(r.choice(gen.MOVE if sink == "move" else gen.COPY)); target()
        q.lit(r.choice(["to", "into", "to", "over to", "in"]))
        if not folder and n == 1 and r.random() < 0.2:
            args["to"] = q.key(); q.chunks.append((q.filename(), args["to"]))      # a new name
        else:
            args["to"] = put(q, "path")
    elif sink == "compress":
        fmt = r.choice(["tgz", "zip"])
        q.lit(r.choice(gen.ZIP if fmt == "zip" else gen.TGZ)); target()
        args = {"fmt": fmt, "to": None}
        if r.random() < 0.5:
            name = q.word() + r.choice([".zip"] if fmt == "zip" else [".tar.gz", ".tgz", ""])
            q.lit(r.choice(["into", "as", "to", "called", "named"]))
            args["to"] = q.key(); q.chunks.append((name, args["to"]))
    elif sink == "sizes":
        form = r.randrange(6)
        q.lit(["how big is", "size of", "how much space does", "how large is", "du", "what's the size of"][form]); target()
        q.lit(["", "", r.choice(["take", "take up", "use"]), "", "", ""][form])
    elif sink == "chmod":
        m = r.choice([w for w in gen.MODE_WORDS if w not in ("executable", "+x", "u+x")])
        mk = q.key()
        form = r.randrange(4)
        if form == 0:
            q.lit("make"); target(); q.chunks.append((m, mk))
        elif form == 1:
            q.lit("chmod"); q.chunks.append((m, mk)); target()
        elif form == 2:
            q.lit(r.choice(["set the permissions of", "change the permissions of", "change the mode of"])); target(); q.lit("to"); q.chunks.append((m, mk))
        else:
            q.lit("set"); target(); q.lit("to"); q.chunks.append((m, mk))
        args = {"mode": mk}
    elif sink == "chown":
        o = r.choice(gen.OWNERS)
        ok = q.key()
        form = r.randrange(3)
        if form == 0:
            q.lit(r.choice(["change the owner of", "set the owner of", "change ownership of"])); target(); q.lit("to"); q.chunks.append((o, ok))
        elif form == 1:
            q.lit("chown"); q.chunks.append((o, ok)); target()
        else:
            q.lit("give"); target(); q.lit("to"); q.chunks.append((o, ok))
        args = {"owner": ok}
    else:
        q.lit(r.choice(["upload", "send", "rsync", "scp", "copy", "sync", "push", "transfer", "mirror", "back up", "ship"])); target()
        q.lit(r.choice(["to", "to", "over to", "to the server"]))
        args = {"to": put(q, "remote")}
    if r.random() < 0.08:
        q.lit(r.choice(TAILS))
    return [("paths", {"names": k}), (sink, args)]


# --------------------------------------------------------------------------- more line requests
_SEP_WORDS = {
    "space": ["", "", "separated by spaces", "whitespace separated", "split on spaces"],
    "comma": ["separated by commas", "comma separated", "using , as the delimiter", "split on commas", "(comma delimited)"],
    "tab": ["tab separated", "separated by tabs", "split on tabs", "(tab delimited)"],
    "colon": ["separated by colons", "using : as the delimiter", "colon delimited", "split on :"],
    "semi": ["separated by semicolons", "semicolon delimited", "using ; as the separator"],
    "bar": ["separated by pipes", "pipe delimited", "using | as the separator"],
}
_INPLACE = ["in place", "in-place", "and save it back", "and overwrite the file", "and write it back to the file", "permanently",
            "and update the file", "and save the file", "for good", "and save the changes"]
_ORD = ["1st", "2nd", "3rd", "4th", "5th", "first", "second", "third", "fourth", "fifth", "last" if False else "sixth"]


def _phrases(q, options):
    """options: list of sequences; a str is literal text, a tuple is (text, key[, quote])."""
    emit(q, q.r.choice(options))


def lines2_request(q):
    r = q.r
    src = r.choices(["file", "history", "env", "pkgs", "services", "journal", "dmesg", "clipboard"], [44, 13, 7, 10, 6, 10, 6, 4])[0]
    if r.random() < 0.22:
        q.lit(r.choice(POLITE))
    c = r.choice
    if src == "file":
        f = q.filename(["log", "txt", "csv", "md", "json", "conf", "py", "sh", "out", "tsv", "yaml", "list", "dat"])
        fk = q.key(); fa = (f, fk)
        pipe = [("lines", {"of": fk})]
        kind = c(["range", "line", "col", "col", "replace", "replace", "replace_show", "freq", "num", "rev", "shuf", "upper", "lower",
                  "sum", "blank", "wcount", "wcount", "ccount", "clip", "inplace", "inplace", "append_out", "random", "countmatch",
                  "countmatch", "ldel", "ldel"])
        if kind == "range":
            a, b = sorted(r.sample(range(1, 400), 2))
            ka, kb = q.key(), q.key()
            if r.random() < 0.18:
                glued = (f"{a}-{b}", ka)
                _phrases(q, [["show lines", glued, "of", fa], ["lines", glued, "of", fa], ["print lines", glued, "from", fa], [fa, "lines", glued]])
                kb = ka
            else:
                A, B = (str(a), ka), (str(b), kb)
                _phrases(q, [["show lines", A, "to", B, "of", fa], ["lines", A, "through", B, "of", fa], ["print lines", A, "to", B, "from", fa],
                             ["from line", A, "to line", B, "of", fa], [fa, "lines", A, "to", B], ["show", fa, "from line", A, "to", B],
                             ["give me lines", A, "to", B, "in", fa], ["what's on lines", A, "through", B, "of", fa]])
            pipe += [("lrange", {"a": ka, "b": kb}), ("lshow", {})]
        elif kind == "line":
            n = (str(r.randint(1, 500)) if r.random() < 0.7 else c(_ORD), q.key())
            _phrases(q, [["show line", n, "of", fa], ["what's on line", n, "of", fa], ["line", n, "of", fa], ["print line", n, "from", fa],
                         ["print the", n, "line of", fa], ["what is line", n, "in", fa], ["show me line", n, "in", fa], [fa, "line", n]])
            pipe += [("lline", {"n": n[1]}), ("lshow", {})]
        elif kind == "col":
            n = (str(r.randint(1, 9)) if r.random() < 0.6 else c(_ORD[:10]), q.key())
            sep = c(["space", "space", "comma", "comma", "tab", "colon", "semi", "bar"])
            said = c(_SEP_WORDS[sep])
            if f.endswith(".csv") and r.random() < 0.7:
                sep, said = "comma", c(["", "", "comma separated"])
            elif f.endswith(".tsv") and r.random() < 0.7:
                sep, said = "tab", ""
            elif f == "/etc/passwd":
                sep, said = "colon", c(["", "separated by colons"])
            _phrases(q, [["print column", n, "of", fa, said], ["show the", n, "column of", fa, said], ["column", n, "of", fa, said],
                         ["get field", n, "from", fa, said], ["cut column", n, "from", fa, said], ["the", n, "field of each line in", fa, said],
                         ["extract column", n, "from", fa, said], ["show only the", n, "column in", fa, said], ["field", n, "of", fa, said]])
            pipe.append(("col", {"n": n[1], "sep": sep}))
            extra = r.random()
            if extra < 0.12:
                q.lit(c(["and add it up", "and sum it", "and total it", "summed", "and add them up"])); pipe += [("lsum", {}), ("lshow", {})]
            elif extra < 0.22:
                q.lit(c(["sorted", "and sort it", "in sorted order"])); pipe += [("lsort", {"key": "alpha", "order": "asc"}), ("lshow", {})]
            elif extra < 0.3:
                q.lit(c(["unique values only", "without duplicates", "deduplicated"])); pipe += [("uniq", {}), ("lshow", {})]
            elif extra < 0.38:
                q.lit(c(["and count how often each value appears", "with counts", "and count each value"])); pipe += [("lfreq", {}), ("lshow", {})]
            else:
                pipe.append(("lshow", {}))
        elif kind in ("replace", "replace_show"):
            o = val(q, "text"); n = val(q, "text")
            ok, nk = q.key(), q.key()
            O, N = (o[0], ok, o[1]), (n[0], nk, n[1])
            if kind == "replace":
                _phrases(q, [["replace", O, "with", N, "in", fa], ["change", O, "to", N, "in", fa], ["swap", O, "for", N, "in", fa],
                             ["substitute", N, "for", O, "in", fa], ["in", fa, "replace", O, "with", N], ["find and replace", O, "with", N, "in", fa],
                             ["replace every", O, "with", N, "in", fa], ["change all occurrences of", O, "to", N, "in", fa],
                             ["turn every", O, "into", N, "in", fa], ["rename", O, "to", N, "everywhere in", fa]])
                pipe += [("lreplace", {"old": ok, "new": nk}), ("inplace", {})]
            else:
                _phrases(q, [["show", fa, "with", O, "replaced by", N], ["print", fa, "replacing", O, "with", N],
                             ["cat", fa, "but with", O, "changed to", N], [fa, "with", O, "swapped for", N],
                             ["show what", fa, "looks like with", O, "replaced by", N], ["preview", fa, "with", O, "changed to", N]])
                pipe += [("lreplace", {"old": ok, "new": nk}), ("lshow", {})]
        elif kind == "freq":
            if r.random() < 0.35:
                n = (q.count(), q.key())
                _phrases(q, [["top", n, "most frequent lines in", fa], ["the", n, "most common lines in", fa], ["show the", n, "most repeated lines of", fa]])
                pipe += [("lfreq", {}), ("ltake", {"n": n[1]}), ("lshow", {})]
            else:
                _phrases(q, [["most common lines in", fa], ["count how often each line appears in", fa], ["frequency of each line in", fa],
                             ["which lines repeat the most in", fa], ["count duplicates in", fa], ["histogram of lines in", fa],
                             ["how many times does each line occur in", fa], ["tally the lines of", fa]])
                pipe += [("lfreq", {}), ("lshow", {})]
        elif kind == "num":
            _phrases(q, [["show", fa, "with line numbers"], ["number the lines of", fa], ["cat -n", fa], ["print", fa, "with line numbers"],
                         [fa, "with numbered lines"], ["show line numbers for", fa], ["display", fa, "with each line numbered"]])
            pipe += [("lnum", {}), ("lshow", {})]
        elif kind == "rev":
            _phrases(q, [["reverse the lines of", fa], ["show", fa, "backwards"], ["print", fa, "in reverse line order"], ["tac", fa],
                         [fa, "from bottom to top"], ["flip", fa, "upside down"], ["show", fa, "last line first"], ["reverse", fa]])
            pipe += [("lrev", {}), ("lshow", {})]
        elif kind == "shuf":
            if r.random() < 0.4:
                n = (q.count(), q.key())
                _phrases(q, [["pick", n, "random lines from", fa], ["give me", n, "random lines of", fa], [n, "random lines from", fa],
                             ["sample", n, "lines from", fa]])
                pipe += [("lshuf", {}), ("ltake", {"n": n[1]}), ("lshow", {})]
            else:
                _phrases(q, [["shuffle the lines of", fa], ["randomize the order of lines in", fa], [fa, "in random order"], ["shuf", fa],
                             ["scramble the lines of", fa], ["shuffle", fa], ["mix up the lines in", fa]])
                pipe += [("lshuf", {}), ("lshow", {})]
        elif kind in ("upper", "lower"):
            w = c(["uppercase", "upper case", "all caps", "capitals"] if kind == "upper" else ["lowercase", "lower case", "all lowercase", "small letters"])
            _phrases(q, [["convert", fa, "to", w], [fa, "in", w], ["make", fa, "all", w], ["print", fa, "in", w], ["show", fa, "in", w],
                         ["turn the text of", fa, "into", w], [w, "version of", fa]])
            pipe += [("lupper" if kind == "upper" else "llower", {}), ("lshow", {})]
        elif kind == "sum":
            _phrases(q, [["add up the numbers in", fa], ["sum", fa], ["total of the numbers in", fa], ["sum all the values in", fa],
                         ["what do the numbers in", fa, "add up to"], ["add up every line of", fa], ["sum the numbers in", fa]])
            pipe += [("lsum", {}), ("lshow", {})]
        elif kind == "blank":
            _phrases(q, [["show", fa, "without blank lines"], [fa, "with empty lines removed"], ["print", fa, "skipping blank lines"],
                         ["hide empty lines in", fa], ["remove blank lines from", fa], ["strip empty lines from", fa],
                         ["drop the blank lines of", fa], ["squeeze out empty lines in", fa]])
            pipe.append(("lblank", {}))
            if r.random() < 0.3:
                q.lit(c(_INPLACE)); pipe.append(("inplace", {}))
            else:
                pipe.append(("lshow", {}))
        elif kind == "wcount":
            _phrases(q, [["how many words in", fa], ["word count of", fa], ["count the words in", fa], ["wc -w", fa], ["number of words in", fa],
                         ["how many words does", fa, "have"], ["count words in", fa], ["how many words are in", fa], ["words in", fa]])
            pipe.append(("wcount", {}))
        elif kind == "ccount":
            _phrases(q, [["how many characters in", fa], ["character count of", fa], ["count the characters in", fa], ["number of chars in", fa],
                         ["how long is", fa, "in characters"], ["wc -m", fa], ["how many letters are in", fa], ["count chars in", fa]])
            pipe.append(("ccount", {}))
        elif kind == "clip":
            _phrases(q, [["copy", fa, "to the clipboard"], ["copy the contents of", fa, "to my clipboard"], ["put", fa, "on the clipboard"],
                         ["copy what's in", fa], [fa, "to clipboard"], ["copy the text of", fa], ["send", fa, "to the clipboard"]])
            pipe.append(("clip", {}))
        elif kind == "inplace":
            op = c(["nomatch", "nomatch", "match", "uniq", "sort", "sortu", "lower", "upper"])
            if op in ("nomatch", "match"):
                t = val(q, "text"); tk = q.key(); T = (t[0], tk, t[1])
                if op == "nomatch":
                    _phrases(q, [["remove lines containing", T, "from", fa], ["delete every line with", T, "in", fa],
                                 ["drop lines mentioning", T, "from", fa], ["strip lines with", T, "out of", fa],
                                 ["delete all lines that contain", T, "from", fa], ["get rid of lines with", T, "in", fa]])
                else:
                    _phrases(q, [["keep only lines containing", T, "in", fa], ["delete every line of", fa, "that doesn't contain", T],
                                 ["remove all lines without", T, "from", fa], ["keep only the lines with", T, "in", fa]])
                pipe.append((op, {"text": tk}))
            elif op == "uniq":
                _phrases(q, [["remove duplicate lines from", fa], ["dedupe", fa], ["deduplicate", fa], ["delete repeated lines in", fa]])
                pipe.append(("uniq", {}))
            elif op in ("sort", "sortu"):
                _phrases(q, [["sort", fa], ["sort the lines of", fa], ["put the lines of", fa, "in order"], ["alphabetize", fa]])
                pipe.append(("lsort", {"key": "alpha", "order": "asc"}))
                if op == "sortu":
                    q.lit(c(["and remove duplicates", "and dedupe", "without duplicates", "and drop repeats"])); pipe.append(("uniq", {}))
            else:
                w = c(["uppercase", "all caps"] if op == "upper" else ["lowercase", "all lowercase"])
                _phrases(q, [["convert", fa, "to", w], ["make", fa, "all", w]])
                pipe.append(("lupper" if op == "upper" else "llower", {}))
            q.lit(c(_INPLACE)); pipe.append(("inplace", {}))
        elif kind == "countmatch":
            t = val(q, "text"); tk = q.key(); T = (t[0], tk, t[1])
            _phrases(q, [["how many lines in", fa, c(["contain", "mention", "have", "include"]), T], ["how many lines of", fa, "mention", T],
                         ["count the lines with", T, "in", fa], ["how many times does", T, c(["appear in", "show up in", "occur in"]), fa],
                         ["number of lines containing", T, "in", fa], ["count occurrences of", T, "in", fa],
                         ["how often does", T, "appear in", fa], ["how many", T, "lines are in", fa]])
            pipe += [("match", {"text": tk}), ("lcount", {})]
        elif kind == "ldel":
            n = (c([str(r.randint(1, 300)), str(r.randint(1, 40)), "first", "last", "1st", "2nd", "3rd", "second", "third"]), q.key())
            if r.random() < 0.75:
                _phrases(q, [["delete line", n, "from", fa], ["remove line", n, "of", fa], ["delete the", n, "line of", fa],
                             ["drop line", n, "from", fa], ["get rid of line", n, "in", fa], ["remove the", n, "line from", fa],
                             ["cut line", n, "out of", fa], ["delete line number", n, "in", fa]])
                pipe += [("ldel", {"n": n[1]}), ("inplace", {})]
            else:
                _phrases(q, [["show", fa, "without line", n], ["print", fa, "skipping line", n], [fa, "minus line", n],
                             ["show", fa, "without the", n, "line"], ["print", fa, "except line", n]])
                pipe += [("ldel", {"n": n[1]}), ("lshow", {})]
        elif kind == "append_out":
            t = val(q, "text"); tk = q.key(); T = (t[0], tk, t[1])
            out = q.filename(["txt", "log", "out", "md"]); ok = q.key()
            _phrases(q, [["append lines containing", T, "from", fa, "to", (out, ok)], ["add the lines with", T, "in", fa, "to the end of", (out, ok)],
                         ["find", T, "in", fa, "and append the matches to", (out, ok)], ["grep", T, fa, ">>", (out, ok)],
                         ["take lines mentioning", T, "from", fa, "and add them to", (out, ok)]])
            pipe += [("match", {"text": tk}), ("lappend", {"to": ok})]
        else:
            n = (q.count(), q.key())
            _phrases(q, [["a random line from", fa], ["pick a random line of", fa], ["one random line from", fa], ["random line of", fa]])
            pipe += [("lshuf", {}), ("ltake", {"n": None}), ("lshow", {})]
        return pipe

    T = None
    if src == "history":
        pipe = [("history", {})]
        kind = c(["show", "match", "match", "match", "last", "count", "matchlast"])
        cmd = c(gen.CMDNAMES + gen.PKGS[:20] + ["docker run", "git push", "ssh", "sudo"])
        tk = q.key(); T = (cmd, tk, " " in cmd)
        n = (q.count(), q.key())
        if kind == "show":
            q.lit(c(["show my command history", "history", "what commands did i run", "list my previous commands", "show shell history",
                     "show my bash history", "what have i run so far", "list the commands i ran"]))
        elif kind == "match":
            _phrases(q, [["search my history for", T], ["history | grep", T], ["find", T, "in my history"], ["what", T, "commands did i run"],
                         ["commands i ran with", T, "in them"], ["when did i last run", T], ["show history entries with", T],
                         ["grep my history for", T], ["did i run", T, "before"], ["look through my history for", T],
                         ["how did i run", T, "last time"], ["what was that", T, "command i ran"]])
            pipe.append(("match", {"text": tk}))
        elif kind == "last":
            _phrases(q, [["last", n, "commands"], ["show my last", n, "commands"], ["the", n, "most recent commands i ran"], ["previous", n, "commands"],
                         ["what were my last", n, "commands"]])
            pipe.append(("llast", {"n": n[1]}))
        elif kind == "count":
            _phrases(q, [["how many times did i run", T], ["count", T, "commands in my history"], ["how often have i run", T]])
            pipe += [("match", {"text": tk}), ("lcount", {})]
            return pipe
        else:
            _phrases(q, [["last", n, T, "commands"], ["my last", n, "commands with", T], ["the", n, "most recent", T, "commands i ran"]])
            pipe += [("match", {"text": tk}), ("llast", {"n": n[1]})]
    elif src == "env":
        pipe = [("env", {})]
        kind = c(["show", "show", "match", "match", "count", "save"])
        t = c(gen.ENVNAMES + ["proxy", "java", "path", "key", "token", "home", "xdg", "lang", "python"])
        tk = q.key(); T = (t, tk)
        if kind == "show":
            q.lit(c(["show environment variables", "list env vars", "printenv", "show all environment variables", "env",
                     "what environment variables are set", "list my environment", "show my env", "dump the environment"]))
        elif kind == "match":
            _phrases(q, [["env vars containing", T], ["environment variables with", T, "in them"], ["show env vars matching", T],
                         ["is there an env var with", T], ["env | grep", T], ["which environment variables mention", T],
                         ["find environment variables related to", T], ["any env vars about", T]])
            pipe.append(("match", {"text": tk}))
        elif kind == "count":
            q.lit(c(["how many environment variables are set", "count my env vars", "number of environment variables"]))
            return pipe + [("lcount", {})]
        else:
            out = q.filename(["txt", "env", "log"]); ok = q.key()
            _phrases(q, [["save my environment to", (out, ok)], ["dump env vars into", (out, ok)], ["write all environment variables to", (out, ok)]])
            return pipe + [("save", {"to": ok})]
    elif src == "pkgs":
        pipe = [("pkgs", {})]
        kind = c(["show", "match", "match", "count", "count", "save"])
        t = c(gen.PKGS + ["python", "lib", "font", "qt", "gtk", "kde", "linux", "nvidia", "perl", "rust"])
        tk = q.key(); T = (t, tk)
        if kind == "show":
            q.lit(c(["list installed packages", "what packages are installed", "show all installed packages", "installed packages",
                     "what do i have installed", "list everything that is installed", "show my installed packages"]))
        elif kind == "match":
            _phrases(q, [["which installed packages have", T, "in the name"], ["installed packages matching", T],
                         ["list installed packages containing", T], ["what", T, "packages do i have installed"],
                         ["show installed packages with", T], ["do i have any", T, "packages installed"],
                         ["installed packages related to", T], ["which", T, "packages are installed"]])
            pipe.append(("match", {"text": tk}))
        elif kind == "count":
            q.lit(c(["how many packages are installed", "count installed packages", "number of installed packages", "how many packages do i have",
                     "how many programs are installed", "total number of installed packages"]))
            return pipe + [("lcount", {})]
        else:
            out = q.filename(["txt", "list", "log"]); ok = q.key()
            _phrases(q, [["save the list of installed packages to", (out, ok)], ["export installed packages to", (out, ok)],
                         ["write my package list to", (out, ok)], ["dump the installed packages into", (out, ok)]])
            return pipe + [("save", {"to": ok})]
    elif src == "services":
        pipe = [("services", {})]
        kind = c(["show", "show", "match", "count"])
        t = c(gen.UNITS + ["network", "blue", "print", "sound", "docker", "ssh"])
        tk = q.key(); T = (t, tk)
        if kind == "show":
            q.lit(c(["list running services", "what services are running", "show running services", "which services are active",
                     "running systemd services", "show all active services", "what daemons are running", "list active services"]))
        elif kind == "match":
            _phrases(q, [["running services matching", T], ["is there a running service with", T, "in the name"],
                         ["which running services mention", T], ["list running services containing", T], ["any running services about", T]])
            pipe.append(("match", {"text": tk}))
        else:
            q.lit(c(["how many services are running", "count running services", "number of active services"]))
            return pipe + [("lcount", {})]
    elif src == "journal":
        unit = None
        kind = c(["show", "last", "last", "match"])
        n = (q.count(), q.key())
        t = val(q, "text"); tk = q.key(); T = (t[0], tk, t[1])
        if r.random() < 0.7:
            u = val(q, "unit")[0]; unit = q.key(); U = (u, unit)
            if kind == "show":
                _phrases(q, [["show the logs of", U], [U, "logs"], ["logs for the", U, "service"], ["what is", U, "logging"],
                             ["show me the", U, "service logs"], ["journalctl -u", U], ["see the journal for", U],
                             ["show logs from", U], ["what did", U, "log"], ["check the", U, "logs"]])
            elif kind == "last":
                _phrases(q, [["last", n, "lines of the", U, "logs"], ["show the last", n, "log lines for", U],
                             ["the", n, "most recent", U, "log entries"], ["tail the", U, "logs", n, "lines"],
                             ["show the", n, "latest log lines from", U]])
            else:
                _phrases(q, [[U, "logs mentioning", T], ["search the", U, "logs for", T], ["lines with", T, "in the", U, "logs"],
                             ["grep the", U, "logs for", T], ["find", T, "in the logs of", U]])
        else:
            if kind == "show":
                q.lit(c(["show the system log", "system journal", "show the journal", "journalctl", "what's in the system log",
                         "show system logs", "show me the journal from this boot"]))
            elif kind == "last":
                _phrases(q, [["last", n, "lines of the system log"], ["show the last", n, "journal entries"], ["the", n, "latest system log lines"]])
            else:
                _phrases(q, [["search the journal for", T], ["journal entries mentioning", T], ["find", T, "in the system log"],
                             ["system log lines with", T], ["grep the journal for", T]])
        pipe = [("journal", {"unit": unit})]
        if kind == "last":
            pipe.append(("llast", {"n": n[1]}))
        elif kind == "match":
            pipe.append(("match", {"text": tk}))
    elif src == "dmesg":
        pipe = [("dmesg", {})]
        kind = c(["show", "match", "match", "last"])
        t = c(["usb", "error", "nvidia", "wifi", "bluetooth", "ata", "thermal", "oom", "firmware", "acpi", "nvme", "i915", "fail"])
        tk = q.key(); T = (t, tk)
        n = (q.count(), q.key())
        if kind == "show":
            q.lit(c(["show kernel messages", "dmesg", "kernel log", "show the kernel log", "what is the kernel saying",
                     "show the kernel ring buffer", "print kernel messages"]))
        elif kind == "match":
            _phrases(q, [["kernel messages about", T], ["dmesg | grep", T], ["search the kernel log for", T],
                         ["kernel log lines mentioning", T], ["any", T, "messages in dmesg"], ["what does the kernel say about", T],
                         ["grep dmesg for", T]])
            pipe.append(("match", {"text": tk}))
        else:
            _phrases(q, [["last", n, "kernel messages"], ["the", n, "most recent kernel messages"], ["tail of dmesg", n, "lines"],
                         ["show the last", n, "lines of the kernel log"]])
            pipe.append(("llast", {"n": n[1]}))
    else:
        pipe = [("clipboard", {})]
        kind = c(["show", "show", "save", "count", "words"])
        if kind == "show":
            q.lit(c(["show what's in the clipboard", "paste the clipboard", "print my clipboard", "what's on my clipboard",
                     "clipboard contents", "what did i copy", "show the clipboard"]))
        elif kind == "save":
            out = q.filename(["txt", "md", "log", "json"]); ok = q.key()
            _phrases(q, [["save the clipboard to", (out, ok)], ["paste the clipboard into", (out, ok)], ["write what i copied to", (out, ok)],
                         ["dump the clipboard into", (out, ok)], ["put my clipboard in", (out, ok)]])
            return pipe + [("save", {"to": ok})]
        elif kind == "count":
            q.lit(c(["how many lines are in the clipboard", "count lines in my clipboard", "how many lines did i copy"]))
            return pipe + [("lcount", {})]
        else:
            q.lit(c(["how many words are in my clipboard", "word count of the clipboard", "count the words i copied"]))
            return pipe + [("wcount", {})]
    # a common tail for the listing sources
    extra = r.random()
    if extra < 0.07:
        out = q.filename(["txt", "log", "out"]); ok = q.key()
        emit(q, [c(["and save it to", "and write that to", "into", "and put it in", "> "]), (out, ok)])
        pipe.append(("save", {"to": ok}))
    elif extra < 0.13:
        q.lit(c(["and count them", "and tell me how many", "- how many are there", "and count"])); pipe.append(("lcount", {}))
    elif extra < 0.17:
        q.lit(c(["sorted", "in alphabetical order", "and sort them"])); pipe += [("lsort", {"key": "alpha", "order": "asc"}), ("lshow", {})]
    elif extra < 0.2:
        q.lit(c(["and copy it to the clipboard", "and copy that"])); pipe.append(("clip", {}))
    else:
        pipe.append(("lshow", {}))
    return pipe


# --------------------------------------------------------------------------- more process requests
def procs2_request(q):
    r = q.r
    c = r.choice
    if r.random() < 0.22:
        q.lit(c(POLITE))
    if r.random() < 0.55:
        n = (val(q, "port")[0], q.key())
        kind = c(["show", "show", "show", "kill", "kill"])
        if kind == "show":
            _phrases(q, [["what's using port", n], ["which process is listening on port", n], ["who is on port", n], ["what is running on port", n],
                         ["what's listening on", n], ["show the process on port", n], ["which program has port", n],
                         ["find what's using port", n], ["what is bound to port", n], ["who has port", n, "open"], ["lsof port", n]])
            return [("port", {"n": n[1]}), ("pshow", {})]
        force = r.random() < 0.25
        if force:
            _phrases(q, [["force kill whatever is on port", n], ["kill -9 the process on port", n], ["forcibly kill the thing using port", n]])
        else:
            _phrases(q, [["kill whatever is on port", n], ["free up port", n], ["kill the process using port", n],
                         ["stop whatever is listening on port", n], ["kill port", n], ["terminate the thing on port", n],
                         ["close whatever has port", n], ["kill the server on port", n]])
        return [("port", {"n": n[1]}), ("kill", {"force": "hard" if force else "soft"})]
    me = r.random() < 0.35
    kind = c(["show", "show", "kill", "count", "top"])
    u = (c(["my", "i"]) if me else c(gen.OWNERS[:10] + [q.word()]), q.key())
    pipe = [("procs", {}), ("puser", {"user": u[1]})]
    if kind == "show":
        if me:
            u = ("my", u[1]) if r.random() < 0.6 else ("i", u[1])
            _phrases(q, [["show", u, "processes"], ["list", u, "processes"], ["what processes are", u, "own" if False else "running"]] if u[0] == "my"
                     else [["what am", u, "running"], ["which processes do", u, "own"], ["what processes am", u, "running"]])
        else:
            _phrases(q, [["show processes owned by", u], ["what is", u, "running"], ["processes of user", u], ["processes belonging to", u],
                         ["ps for user", u], ["list everything", u, "is running"], ["which processes does", u, "own"]])
        pipe.append(("pshow", {}))
    elif kind == "kill":
        force = r.random() < 0.25
        v = c(["force kill", "kill -9"] if force else ["kill", "stop", "terminate", "end"])
        if me:
            u = ("my", u[1])
            _phrases(q, [[v, "all", u, "processes"], [v, "all of", u, "processes"]])
        else:
            _phrases(q, [[v, "all processes of user", u], [v, "everything", u, "is running"], [v, "all processes owned by", u],
                         [v, "every process belonging to", u]])
        pipe.append(("kill", {"force": "hard" if force else "soft"}))
    elif kind == "count":
        if me:
            u = ("i", u[1])
            _phrases(q, [["how many processes am", u, "running"], ["how many processes do", u, "have"]])
        else:
            _phrases(q, [["how many processes does", u, "have"], ["count processes owned by", u], ["number of processes run by", u]])
        pipe.append(("pcount", {}))
    else:
        key = c(["mem", "cpu"])
        res = c({"mem": ["memory", "ram"], "cpu": ["cpu", "processor"]}[key])
        n = (q.count(), q.key())
        if me:
            u = ("my", u[1])
            _phrases(q, [["top", n, "of", u, "processes by", res], [u, n, "biggest", res, "users"]])
        else:
            _phrases(q, [["top", n, "processes of user", u, "by", res], ["the", n, "processes of", u, "using the most", res]])
        pipe += [("psort", {"key": key, "order": "desc"}), ("ptake", {"n": n[1]}), ("pshow", {})]
    return pipe


# --------------------------------------------------------------------------- session preferences
PREFS = {
    ("trash", "on"): ["use the trash instead of deleting", "always move deleted files to the trash", "send deleted files to the trash from now on",
                      "don't delete permanently, use the trash", "prefer the trash over rm", "from now on trash things instead of deleting them",
                      "use trash", "turn on trash mode", "deletes should go to the trash", "never rm, always trash"],
    ("trash", "off"): ["stop using the trash", "delete for real from now on", "delete permanently again", "don't use the trash anymore",
                       "turn off trash mode", "go back to real deletes", "no more trash, just delete"],
    ("confirm", "on"): ["ask me before deleting", "always confirm before overwriting", "ask before you delete or overwrite anything",
                        "use interactive mode", "always ask first", "confirm every deletion", "add -i to rm and mv", "make deletes interactive",
                        "prompt me before removing files"],
    ("confirm", "off"): ["stop asking me to confirm", "don't ask before deleting", "no more confirmations", "turn off interactive mode",
                         "stop confirming", "don't prompt me anymore"],
    ("verbose", "on"): ["be verbose", "show what gets moved or deleted", "always use verbose mode", "print every file you touch",
                        "add -v to commands", "verbose mode on", "tell me each file as you go", "make commands verbose"],
    ("verbose", "off"): ["stop being verbose", "quiet mode", "turn off verbose mode", "don't print every file", "be quiet", "verbose mode off"],
}


def pref_request(q):
    key = q.r.choice(list(PREFS))
    if q.r.random() < 0.2:
        q.lit(q.r.choice(POLITE))
    q.lit(q.r.choice(PREFS[key]))
    return [("pref", {"key": key[0], "state": key[1]})]


# --------------------------------------------------------------------------- requests plainsh cannot do
OUTSIDE = """convert {f} to {e}|convert {f} to a gif|resize {f} to {n}x{n}|trim the first {n} seconds of {f}|merge all pdfs into one
turn {f} into a pdf|rotate {f} by {n} degrees|extract the audio from {f}|make a thumbnail of {f}|compress the video {f}|crop {f}
join {f} and {f2} into one video|speed up {f} by {n}x|make {f} black and white|ocr {f}|print {f} on the printer|scan a document
set the wallpaper to {f}|change the wallpaper|take a screenshot|record the screen|record my voice|play some music|play {f}|pause the music
skip this song|what song is this|open port {n} in the firewall|block port {n}|block {h}|allow ssh through the firewall|set up a vpn
add a user called {w}|delete the user {w}|change my password|change the password for {w}|add {w} to the sudo group|give {w} sudo access
create a group called {w}|lock the account {w}|format the drive|format {d} as ext4|partition the disk|mount the usb drive|unmount {d}
eject the disk|check the disk for errors|repair the filesystem|encrypt {f}|decrypt {f}|sign {f} with gpg|generate an ssh key
generate a password|make a random password|create a qr code for {u}|hello|hi there|who are you|tell me a joke|what's the weather
what's the weather in {w}|thanks|thank you|good morning|how are you|what can you do|help|help me|are you there|you are great|nice
what is the meaning of life|sing a song|write a poem about {w}|make me a sandwich|make it rain|make coffee|make {w} great again
find love|find me a girlfriend|find the meaning of {w}|kill time|kill the vibe|open sesame|show me the money|move on|copy that
delete my ex from my memory|go to hell|run away|stop it|stop|go|run|do it|yes|no|ok|maybe|why|what|test|asdf|hmm|lol|never mind
install confidence|remove all doubt|search the web for {w}|google {w}|look up {w} on wikipedia|translate {w} to french|what does {w} mean
define {w}|how do you spell {w}|explain {f}|summarize {f}|summarise this file|what does this script do|what does this error mean
why did that fail|fix it|fix the bug|fix the error in {f}|debug {f}|refactor {f}|optimize {f}|write a script that {w}s every {n} minutes
write a python script to {w}|write a function that sorts a list|write a bash loop over {e} files|generate a makefile|write a dockerfile
write unit tests for {f}|add a comment to {f}|what is a symlink|what is the difference between {w} and {w2}|how does dns work
what is {n} feet in meters|convert {n} miles to km|how many days until christmas|what time is it in tokyo|what day was {n} days ago
how old is linux|who wrote git|when was {w} released|schedule a job at {n}|run {f} every day at {n}|add a cron job that runs {f} hourly
run this every night at midnight|remind me to {w} tomorrow|set an alarm for {n} am|send an email to {w}|email {f} to {w}|text {w}
call {w}|tweet {w}|post {w} to slack|open a ticket|book a meeting|create a database called {w}|run this sql query|select all users from {w}
drop the table {w}|back up the database|restore the database from {f}|kubectl get pods|deploy to production|deploy the app
scale the deployment to {n}|terraform apply|create an s3 bucket|list my ec2 instances|upload {f} to s3|launch a vm|start the virtual machine
compile {f}|compile {f} with optimizations|link {f} against {w}|disassemble {f}|benchmark {f}|profile {f}|strace {p}|attach gdb to {p}
change the hostname to {w}|set the timezone to {w}|set the keyboard layout to {w}|change the system language|set the date to {n}
enable dark mode|change the theme|make the font bigger|rotate the screen|connect my bluetooth headphones|pair with {w}
switch to the external monitor|turn off the screen|overclock the cpu|update the bios|flash the firmware|install windows|install the nvidia driver
rename it|delete it|move it|do that for all of them|the other one|not that one|undo|undo that|undo the last command|revert that
go back in time|restore what i deleted|recover {f}|bring back the file i deleted|is this safe|are you sure|what would that do
split {f} into {n} parts|merge {f} and {f2}|join the lines of {f}|wrap {f} at {n} columns|indent {f}|format {f}|lint {f}|prettify {f}
convert {f} from csv to json|parse {f} as json|validate {f}|pretty print {f}|base64 encode {f}|base64 decode {w}|url encode {w}
hex dump of {f}|count the vowels in {f}|spell check {f}|find typos in {f}|translate {f} to spanish|read {f} out loud
how's the weather looking|is it going to rain|will it snow tomorrow|what's the forecast|launch the browser|open the browser
open a new terminal|open the file manager|open the settings|who won the {w} cup|who won the game|what's the capital of {w}
who is the president of {w}|why is my computer so slow|why is {p} crashing|why does {p} keep freezing|swap tabs for spaces in {f}
convert tabs to spaces in {f}|indent {f} with tabs|plug in the external drive and mount it|mount {d}|safely remove the usb stick
what is {w}|tell me about {w}|how tall is {w}|shrink {f} to half size|make {f} smaller|make {f} louder|reverse the video {f}
set a reminder|what is my name|where do i live|how much money do i have|order a pizza|buy {w}|what should i eat|i'm bored|i am tired""".replace("\n", "|").split("|")


def outside_request(q):
    r = q.r
    t = r.choice(OUTSIDE)
    text = t.format(f=q.filename(), f2=q.filename(), e=r.choice(gen.EXTS), w=q.word(), w2=q.word(),
                    n=r.choice([1, 2, 5, 10, 30, 50, 80, 100, 644, 755, 8080]), h=r.choice(gen.HOSTS), u=r.choice(gen.URLS),
                    p=r.choice(gen.PROCS + gen.PKGS), d=r.choice(gen.REAL_DIRS))
    if r.random() < 0.2:
        q.lit(r.choice(POLITE))
    q.lit(text)
    if r.random() < 0.06:
        q.lit(r.choice(TAILS))
    return [("unknown", {})]


# --------------------------------------------------------------------------- one request
SUDO_PRE = ["sudo", "as root", "with sudo", "as superuser", "with root privileges", "as admin", "using sudo"]
KINDS = ["files", "lines", "lines2", "procs", "procs2", "single", "fileq", "paths", "pref", "outside"]
KIND_WT = [32, 6, 9, 5, 2, 33, 3, 5, 0.6, 5]
_GEN = {"files": gen.files_request, "lines": gen.lines_request, "lines2": lines2_request, "procs": gen.procs_request,
        "procs2": procs2_request, "single": command_request, "fileq": gen.file_question, "paths": paths_request,
        "pref": pref_request, "outside": outside_request}


def one_pipeline(q, kinds=KINDS, weights=KIND_WT):
    r = q.r
    kind = r.choices(kinds, weights)[0]
    as_root = kind in ("files", "lines", "lines2", "procs", "paths") and r.random() < 0.035
    front = as_root and r.random() < 0.5
    if front:
        q.lit(r.choice(SUDO_PRE[:3]))
    pipe = _GEN[kind](q)
    if as_root and not front:
        q.lit(r.choice(SUDO_PRE))
    return [("sudo", {})] + pipe if as_root else pipe


# --------------------------------------------------------------------------- follow-ups
def keep(pipe, where, i=0):
    """The previous pipeline again, with every argument pointing at its word in the context."""
    out = []
    for j, (name, args) in enumerate(pipe):
        a = {}
        for slot in ir.BY_NAME[name].slots:
            if slot.kind == "span":
                a[slot.name] = ("ctx", where[(i, j, slot.name)]) if (i, j, slot.name) in where else None
            else:
                a[slot.name] = args[slot.name]
        out.append((name, a))
    return out


def fill(q, phrase, kind=None):
    """Write a phrase; "{}" marks where a fresh value of `kind` goes. Returns that value's key (or None)."""
    if "{}" not in phrase:
        q.lit(phrase)
        return None
    a, b = phrase.split("{}")
    q.lit(a)
    k = put(q, kind)
    q.lit(b)
    return k


_VAL_KIND = {"size": "size", "dur": "dur", "count": "count", "text": "text", "glob": "word", "ext": "ext", "date": "date", "path": "path",
             "owner": "owner", "mode": "mode", "file": "textfile", "remote": "remote", "word": "word", "port": "port", "unit": "unit"}
FOLLOW_SINK = {
    "delete": ["delete them", "now delete them", "remove those", "delete those files", "get rid of them", "ok delete them all", "wipe them",
               "rm them", "delete it", "nuke them", "delete all of those", "and delete them", "remove them", "erase them", "delete these",
               "go ahead and delete them", "delete them all", "remove all of them", "now remove those", "delete that"],
    "trash": ["trash them", "move them to the trash", "put those in the bin", "send them to the trash", "trash it", "bin them"],
    "fcount": ["how many are there", "count them", "how many is that", "how many", "count those", "how many of them", "what's the count",
               "give me a count", "how many files is that"],
    "total": ["how big are they in total", "how much space do they take", "total size", "what's their combined size", "how much do they add up to",
              "how much space is that", "how much space would that free"],
    "sizes": ["how big is each", "show their sizes", "with sizes", "how big are they", "list them with sizes", "and their sizes"],
    "details": ["show details", "with permissions", "ls -l them", "show me the details of those", "who owns them", "list them with details"],
    "fopen": ["open them", "open it", "open those", "open that one", "open them all"],
    "fedit": ["edit it", "edit them", "open them in my editor", "let me edit those"],
    "fhash": ["checksum them", "hash those", "sha256 them"],
    "rn_lower": ["rename them to lowercase", "lowercase their names", "make their names lowercase"],
    "rn_spaces": ["replace the spaces in their names with underscores", "remove spaces from their names"],
    "by_ext": ["break them down by type", "count them by extension", "how many of each type"],
    "dupes": ["are any of them duplicates", "find duplicates among them", "which of those are identical"],
    "fshow": ["show them", "list them", "which ones", "show me them", "list those", "let me see them", "show them again", "which files"],
    "cat_count": ["how many lines do they have", "count their lines", "how many lines in total", "count the lines in those"],
}
FOLLOW_SINK_ARG = {
    "move": (["move them to {}", "put those in {}", "now move them into {}", "move it to {}", "send them to {}", "move those to {}",
              "shift them over to {}", "stick them in {}", "move them all to {}"], "to", "path"),
    "copy": (["copy them to {}", "back them up to {}", "copy those into {}", "copy it to {}", "make copies in {}", "copy them all to {}"], "to", "path"),
    "chmod": (["make them {}", "chmod {} them", "set them to {}", "make those {}"], "mode", "mode"),
    "chown": (["give them to {}", "change their owner to {}", "chown them to {}"], "owner", "owner"),
    "upload": (["upload them to {}", "send those to {}", "rsync them to {}", "scp them to {}", "copy them to {}"], "to", "remote"),
    "fgrep": (["search them for {}", "grep them for {}", "look for {} in those", "which lines mention {}", "find {} in them", "search those for {}"], "text", "text"),
    "rn_prefix": (["prefix them with {}", "add {} to the start of their names"], "text", "affix"),
    "rn_ext": (["change their extension to {}", "rename them to {}"], "ext", "ext"),
    "names_save": (["save that list to {}", "write those names to {}", "save the list to {}", "put that list in {}"], "to", "textfile"),
}
FOLLOW_ZIP = [("zip them up", "zip", None), ("compress them", "tgz", None), ("tar them up", "tgz", None), ("zip those", "zip", None),
              ("make a tarball of them", "tgz", None), ("put them in a zip called {}", "zip", ".zip"), ("archive those as {}", "tgz", ".tar.gz"),
              ("zip them into {}", "zip", ".zip"), ("tar them into {}", "tgz", ".tar.gz")]
FOLLOW_SORT = [("sort them by size", "size", "desc", 0), ("biggest first", "size", "desc", 0), ("largest first", "size", "desc", 0),
               ("newest first", "time", "desc", 0), ("oldest first", "time", "asc", 0), ("sort by date", "time", "desc", 0),
               ("sorted by name", "name", "asc", 0), ("order them by size", "size", "desc", 0), ("smallest first", "size", "asc", 0),
               ("just the top {}", None, None, 1), ("only the {} largest", "size", "desc", 1), ("show the {} newest of those", "time", "desc", 1),
               ("the biggest one", "size", "desc", 2), ("top {}", None, None, 1), ("first {} only", None, None, 1),
               ("just the {} biggest", "size", "desc", 1), ("only the {} oldest", "time", "asc", 1), ("the newest one", "time", "desc", 2),
               ("only the {} smallest", "size", "asc", 1), ("limit it to {}", None, None, 1), ("the oldest one", "time", "asc", 2)]
FOLLOW_LOC = ["same but in {}", "now in {}", "do the same for {}", "what about {}", "and in {}", "same thing for {}", "now do {}",
              "repeat for {}", "same for {}", "try {} instead", "how about in {}", "now check {}", "what about in {}", "same in {}"]
FOLLOW_VAL = ["no, {}", "i meant {}", "actually {}", "make it {}", "make that {}", "sorry, {}", "oops {}", "change it to {}", "{} instead",
              "use {} instead", "no i meant {}", "i said {}", "should be {}", "wait, {}", "not that, {}", "rather {}", "change that to {}"]
FOLLOW_EXT = ["same for {}", "what about {}", "now {}", "and for {} files", "do the same with {}", "same but {}", "how about {}"]
FOLLOW_SUDO = ["with sudo", "as root", "try again as root", "do it with sudo", "same but with sudo", "sudo", "use sudo",
               "permission denied, use sudo", "run that as root", "again as root", "needs sudo", "that needs root"]
FOLLOW_PICK = [("open the {} one", "fopen"), ("delete the {} one", "delete"), ("show me the {}", "fshow"), ("just the {} one", "fshow"),
               ("edit the {} one", "fedit"), ("trash the {} one", "trash"), ("how big is the {} one", "sizes"), ("open the {}", "fopen"),
               ("the {} one", "fshow"), ("delete the {}", "delete")]
_ORDER_STAGES = ("fsort", "ftake", "fpick")
_SOURCE_OR_FILTER = set(ir.FILTER_ORDER) | {"files", "paths", "not", "sudo"}


def _split_files(K):
    body = [st for st in K if st[0] in _SOURCE_OR_FILTER]
    order = [st for st in K if st[0] in _ORDER_STAGES]
    sink = [st for st in K if st[0] not in _SOURCE_OR_FILTER and st[0] not in _ORDER_STAGES]
    return body, order, sink


def files_followup(q, A, where):
    r = q.r
    K = keep(A, where)
    body, order, sink = _split_files(K)
    dirs = any(n == "files" and a.get("what") == "dirs" for n, a in K)
    edit = r.choices(["sink", "filter", "sort", "loc", "value", "pick", "sudo"], [42, 20, 10, 9, 11, 5, 3])[0]
    if edit == "sink":
        roll = r.random()
        if roll < 0.62:
            name = r.choice(list(FOLLOW_SINK))
            q.lit(r.choice(FOLLOW_SINK[name]))
            return [body + order + gen.frame_stages(name, {})]
        if roll < 0.9:
            name = r.choice(list(FOLLOW_SINK_ARG))
            phrases, slot, kind = FOLLOW_SINK_ARG[name]
            k = fill(q, r.choice(phrases), kind)
            return [body + order + gen.frame_stages(name, {slot: k})]
        phrase, fmt, ext = r.choice(FOLLOW_ZIP)
        k = None
        if ext:
            a, b = phrase.split("{}")
            q.lit(a); k = q.key(); q.chunks.append((q.word() + r.choice([ext, ext, ""]), k)); q.lit(b)
        else:
            q.lit(phrase)
        return [body + order + [("compress", {"fmt": fmt, "to": k})]]
    if edit == "filter":
        have = {n for n, _ in body}
        for _ in range(12):
            stages, pre, post, noun = gen.file_filters(q, dirs)
            names = [n for n in stages if n != "__not__"]
            if names and len(names) <= 2 and not (set(names) & have):
                break
        else:
            return None
        bare = bool(pre) or noun is not None
        lead = r.choice(["only", "just", "but only", "now only", "filter to", "limit it to", "only show", "narrow it to"] if bare else
                        ["only the ones", "just the ones", "only those", "keep only the ones", "narrow it to the ones", "of those only the ones",
                         "which of them are", "which of those are", "the ones", "only show the ones", "but only the ones", "just those"])
        q.lit(lead)
        for p in pre:
            emit(q, p)
        if noun is not None:
            emit(q, [noun])
        elif bare:
            q.lit(r.choice(["ones", "files", "ones"]))
        gen.tail(q, post)
        merged = {n: a for n, a in body if n in ir.FILTER_ORDER}
        negs = {body[i + 1][0] for i, (n, _) in enumerate(body) if n == "not"}
        merged.update({n: a for n, a in stages.items() if n != "__not__"})
        negs |= set(stages.get("__not__", ()))
        if negs:
            merged["__not__"] = negs
        head = [st for st in body if st[0] in ("sudo", "files", "paths")]
        return [head + gen.filter_stages(merged) + order + sink]
    if edit == "sort":
        phrase, key, direction, mode = r.choice(FOLLOW_SORT)
        k = fill(q, phrase, "count")
        new = [st for st in order if st[0] == "fsort"] if key is None else [("fsort", {"key": key, "order": direction})]
        if mode == 1:
            new.append(("ftake", {"n": k}))
        elif mode == 2:
            new.append(("ftake", {"n": None}))
        else:
            new += [st for st in order if st[0] == "ftake"]
        return [body + new + sink]
    if edit == "loc":
        if not any(n == "files" for n, _ in body):
            return None
        k = fill(q, r.choice(FOLLOW_LOC), "path")
        return [[(n, dict(a, **{"in": k})) if n == "files" else (n, a) for n, a in body] + order + sink]
    if edit == "value":
        spots = [(j, slot) for j, (n, a) in enumerate(K) for slot in ir.BY_NAME[n].slots
                 if slot.kind == "span" and a.get(slot.name) is not None and n not in ("files", "paths") and slot.value in _VAL_KIND]
        if not spots:
            return None
        j, slot = r.choice(spots)
        k = fill(q, r.choice(FOLLOW_EXT if slot.value == "ext" and r.random() < 0.6 else FOLLOW_VAL), _VAL_KIND[slot.value])
        return [[(n, dict(a, **{slot.name: k})) if i == j else (n, a) for i, (n, a) in enumerate(K)]]
    if edit == "pick":
        phrase, name = r.choice(FOLLOW_PICK)
        a, b = phrase.split("{}")
        q.lit(a); k = q.key(); q.chunks.append((r.choice(gen.ORDINALS), k)); q.lit(b)
        return [body + [st for st in order if st[0] != "fpick"] + [("fpick", {"n": k}), (name, {})]]
    if K[0][0] == "sudo":
        return None
    q.lit(r.choice(FOLLOW_SUDO))
    return [[("sudo", {})] + K]


FOLLOW_LSINK = {
    "lcount": ["count them", "how many", "how many lines", "count those lines", "how many is that", "how many are there", "count that"],
    "clip": ["copy that to the clipboard", "copy them", "put that on my clipboard", "copy that"],
    "wcount": ["how many words", "count the words", "how many words is that"],
    "lshow": ["show them", "show me", "let me see them"],
}
FOLLOW_LSINK_ARG = {"save": (["save that to {}", "save them to {}", "write that to {}", "put the result in {}", "save it as {}", "save those to {}"], "to"),
                    "lappend": (["append that to {}", "add those to {}", "append them to {}"], "to")}
FOLLOW_LOP = [
    (["only lines with {}", "just the ones containing {}", "filter for {}", "only those mentioning {}", "grep that for {}", "narrow it to {}",
      "only the ones with {}"], "match", "text", "text"),
    (["without {}", "drop the ones with {}", "exclude {}", "hide lines with {}", "but not the ones mentioning {}"], "nomatch", "text", "text"),
    (["remove duplicates", "dedupe them", "unique ones only", "without duplicates"], "uniq", None, None),
    (["just the first {}", "only the top {}", "first {} only", "limit that to {}"], "ltake", "n", "count"),
    (["just the last {}", "only the last {}", "the last {} of those"], "llast", "n", "count"),
    (["which are most common", "count how often each appears", "most frequent first"], "lfreq", None, None),
    (["with line numbers", "number them"], "lnum", None, None),
]
FOLLOW_LSORT = [("sort them", "alpha", "asc"), ("sorted", "alpha", "asc"), ("sort that", "alpha", "asc"), ("in alphabetical order", "alpha", "asc"),
                ("sort them in reverse", "alpha", "desc"), ("sort numerically", "num", "asc"), ("biggest number first", "num", "desc")]
FOLLOW_FILE = ["same for {}", "now do {}", "what about {}", "same thing with {}", "and in {}", "try {} instead", "do that for {} too"]


def lines_followup(q, A, where):
    r = q.r
    K = keep(A, where)
    ops, sink = K[:-1], K[-1:]
    if sink[0][0] not in ("lshow", "lcount", "wcount", "ccount", "save", "lappend", "clip", "inplace"):
        ops, sink = K, [("lshow", {})]
    edit = r.choices(["sink", "op", "sort", "file", "value", "pron", "sudo"], [30, 32, 8, 9, 8, 10, 3])[0]
    if edit == "sink":
        if r.random() < 0.6:
            name = r.choice(list(FOLLOW_LSINK))
            q.lit(r.choice(FOLLOW_LSINK[name]))
            return [ops + [(name, {})]]
        name = r.choice(list(FOLLOW_LSINK_ARG))
        phrases, slot = FOLLOW_LSINK_ARG[name]
        return [ops + [(name, {slot: fill(q, r.choice(phrases), "textfile")})]]
    if edit == "op":
        phrases, name, slot, kind = r.choice(FOLLOW_LOP)
        k = fill(q, r.choice(phrases), kind)
        return [ops + [(name, {slot: k} if slot else {})] + sink]
    if edit == "sort":
        phrase, key, direction = r.choice(FOLLOW_LSORT)
        q.lit(phrase)
        return [ops + [("lsort", {"key": key, "order": direction})] + sink]
    src = next((j for j, (n, _) in enumerate(K) if n == "lines"), None)
    if edit == "file":
        if src is None:
            return None
        k = fill(q, r.choice(FOLLOW_FILE), "textfile")
        return [[(n, {"of": k}) if j == src else (n, a) for j, (n, a) in enumerate(K)]]
    if edit == "value":
        spots = [(j, slot) for j, (n, a) in enumerate(K) for slot in ir.BY_NAME[n].slots
                 if slot.kind == "span" and a.get(slot.name) is not None and n != "lines" and slot.value in _VAL_KIND]
        if not spots:
            return None
        j, slot = r.choice(spots)
        k = fill(q, r.choice(FOLLOW_VAL), _VAL_KIND[slot.value])
        return [[(n, dict(a, **{slot.name: k})) if i == j else (n, a) for i, (n, a) in enumerate(K)]]
    if edit == "pron":
        if src is None:
            return None
        return pronoun_followup(q, "file", K[src][1]["of"])
    if K[0][0] == "sudo":
        return None
    q.lit(r.choice(FOLLOW_SUDO))
    return [[("sudo", {})] + K]


def procs_followup(q, A, where):
    r = q.r
    K = keep(A, where)
    body = K[:-1]
    edit = r.choices(["kill", "count", "show", "top", "name"], [50, 12, 8, 15, 15])[0]
    if edit == "kill":
        force = r.random() < 0.3
        q.lit(r.choice(["force kill it", "kill -9 it", "kill it hard", "force quit it", "nuke it", "force kill them", "kill -9 them"] if force else
                       ["kill it", "kill them", "kill that", "stop it", "terminate them", "end it", "kill those", "ok kill it", "now kill it",
                        "kill them all", "stop them", "terminate it"]))
        return [body + [("kill", {"force": "hard" if force else "soft"})]]
    if edit == "count":
        q.lit(r.choice(["how many", "count them", "how many are there", "how many is that"]))
        return [body + [("pcount", {})]]
    if edit == "show":
        q.lit(r.choice(["show them", "list them", "show me", "which ones"]))
        return [body + [("pshow", {})]]
    if edit == "top":
        if K[0][0] == "port":
            return None
        key = r.choice(["mem", "cpu"])
        res = r.choice({"mem": ["memory", "ram"], "cpu": ["cpu", "processor"]}[key])
        base = [st for st in body if st[0] not in ("psort", "ptake")]
        if r.random() < 0.5:
            k = fill(q, r.choice([f"top {{}} by {res}", f"just the {{}} using the most {res}", f"only the top {{}} by {res}"]), "count")
            return [base + [("psort", {"key": key, "order": "desc"}), ("ptake", {"n": k})] + K[-1:]]
        q.lit(r.choice([f"sort by {res}", f"sort them by {res}", f"order by {res} usage", f"by {res}"]))
        return [base + [("psort", {"key": key, "order": "desc"})] + [st for st in body if st[0] == "ptake"] + K[-1:]]
    spot = next(((j, s) for j, (n, a) in enumerate(K) for s in ("name", "n") if n in ("pname", "port") and s in a), None)
    if spot is None:
        return None
    j, slot = spot
    kind = "port" if K[j][0] == "port" else None
    a, b = r.choice(["same for {}", "what about {}", "now {}", "and {}", "do the same for {}", "i meant {}", "no, {}"]).split("{}")
    q.lit(a); k = q.key()
    q.chunks.append((val(q, "port")[0] if kind else r.choice(gen.PROCS), k)); q.lit(b)
    return [[(n, dict(a2, **{slot: k})) if i == j else (n, a2) for i, (n, a2) in enumerate(K)]]


def _c(name, slot):
    return lambda c, q: [(name, {slot: c})]


PRON = {
    "path": [
        (["go there", "cd there", "cd into it", "take me there", "go into it", "enter it", "now go there", "cd to it", "jump there",
          "go to that folder", "cd into that", "switch to it", "and go there", "now cd into it", "go in there"], _c("cd", "path")),
        (["what's in there", "look inside", "list it", "show what's in it", "ls there", "what's inside it", "show me what's in there"], _c("ls", "path")),
        (["list the files there", "what files are in it", "show the files in there", "find the files in it"],
         lambda c, q: [("files", {"what": "anyfile", "in": c}), ("fshow", {})]),
        (["how big is it", "how much space does it take", "what's its size"], lambda c, q: [("paths", {"names": c}), ("sizes", {})]),
        (["delete it", "remove it", "get rid of it", "delete that folder"], lambda c, q: [("paths", {"names": c}), ("delete", {})]),
        (["show it as a tree", "tree of it", "show its structure"], lambda c, q: [("tree", {"path": c, "depth": None})]),
    ],
    "file": [
        (["show it", "print it", "cat it", "what's in it", "show me that file", "read it", "let me see it", "display it"], _c("view", "file")),
        (["edit it", "open it in my editor", "let me edit it", "now edit that", "open it in vim", "edit that file"], _c("edit", "file")),
        (["open it", "open that", "xdg-open it"], _c("open", "target")),
        (["delete it", "remove it", "remove that file", "get rid of it", "rm it", "delete that"], lambda c, q: [("paths", {"names": c}), ("delete", {})]),
        (["move it to {}", "put it in {}", "move that to {}"], lambda c, q, k: [("paths", {"names": c}), ("move", {"to": k})], "path"),
        (["copy it to {}", "copy that into {}"], lambda c, q, k: [("paths", {"names": c}), ("copy", {"to": k})], "path"),
        (["make it executable", "chmod +x it", "let me run it", "mark it executable"], _c("make_exec", "file")),
        (["back it up", "make a backup of it", "keep a copy of it"], _c("backup", "file")),
        (["rename it to {}", "call it {} instead", "change its name to {}"], lambda c, q, k: [("rename", {"old": c, "new": k})], "textfile"),
        (["run it", "execute it", "now run it", "run that", "launch it"], _c("run_file", "file")),
        (["how big is it", "what's its size", "how large is that"], lambda c, q: [("paths", {"names": c}), ("sizes", {})]),
        (["how many lines does it have", "count its lines", "how long is it"], lambda c, q: [("lines", {"of": c}), ("lcount", {})]),
        (["search it for {}", "grep it for {}", "find {} in it", "does it mention {}"],
         lambda c, q, k: [("lines", {"of": c}), ("match", {"text": k}), ("lshow", {})], "text"),
        (["follow it", "tail it", "watch it for changes"], _c("follow", "file")),
        (["who owns it", "when was it last modified", "show its permissions", "stat it"], _c("finfo", "file")),
        (["stage it", "git add it", "add it to git"], _c("git_add", "file")),
        (["checksum it", "hash it", "sha256 of it"], _c("checksum", "file")),
        (["what kind of file is it", "what type is it"], _c("ftype", "file")),
        (["show the first {} lines of it", "first {} lines of it"], lambda c, q, k: [("lines", {"of": c}), ("ltake", {"n": k}), ("lshow", {})], "count"),
        (["show the last {} lines of it", "last {} lines of it"], lambda c, q, k: [("lines", {"of": c}), ("llast", {"n": k}), ("lshow", {})], "count"),
    ],
    "unit": [
        (["restart it", "now restart it", "bounce it", "restart that", "restart that service", "give it a restart"], _c("svc_restart", "unit")),
        (["stop it", "stop that service", "shut it down", "turn it off"], _c("svc_stop", "unit")),
        (["start it", "start it up", "bring it up", "start that service", "turn it on"], _c("svc_start", "unit")),
        (["enable it", "enable it at boot", "make it start on boot", "autostart it"], _c("svc_enable", "unit")),
        (["disable it", "stop it from starting at boot", "disable that service"], _c("svc_disable", "unit")),
        (["check its status", "is it running", "how is it doing", "status of it", "is it up", "did it start"], _c("svc_status", "unit")),
        (["show its logs", "what's it logging", "show me the logs", "logs for it", "check its logs"],
         lambda c, q: [("journal", {"unit": c}), ("lshow", {})]),
        (["follow its logs", "tail its logs", "watch its logs"], _c("svc_follow", "unit")),
    ],
    "pkg": [
        (["install it", "ok install it", "install that", "get it installed", "go ahead and install it"], _c("pkg_install", "names")),
        (["uninstall it", "remove it", "get rid of it"], _c("pkg_remove", "names")),
        (["tell me more about it", "info on it", "show details about it", "what version is it"], _c("pkg_info", "name")),
        (["what files does it install", "list its files"], _c("pkg_files", "name")),
    ],
    "container": [
        (["stop it", "stop that container", "shut it down"], _c("dk_stop", "name")), (["restart it"], _c("dk_restart", "name")),
        (["show its logs", "logs for it", "what's it printing"], _c("dk_logs", "name")),
        (["follow its logs", "tail its logs"], _c("dk_follow", "name")),
        (["remove it", "delete it", "delete that container"], _c("dk_rm", "name")),
        (["get a shell in it", "exec into it", "open a shell inside it", "jump into it"], _c("dk_shell", "name")),
        (["start it", "start it again"], _c("dk_start", "name")),
    ],
    "image": [(["run it", "start a container from it"], _c("dk_run", "image")), (["remove it", "delete that image"], _c("dk_rmi", "image")),
              (["pull it", "pull it again"], _c("dk_pull", "image"))],
    "branch": [
        (["switch to it", "check it out", "go to it", "checkout that branch"], _c("git_checkout", "branch")),
        (["merge it", "merge it in", "merge that into this branch"], _c("git_merge", "branch")),
        (["delete it", "delete that branch", "remove it", "get rid of that branch"], _c("git_branch_del", "branch")),
        (["rebase onto it"], _c("git_rebase", "branch")),
    ],
    "host": [
        (["ssh into it", "log in to it", "connect to it", "ssh there", "get a shell on it"], _c("ssh", "host")),
        (["ping it", "is it up", "is it reachable", "check if it's online"], _c("ping", "host")),
        (["what's its ip", "resolve it", "look up its ip"], _c("dns", "host")),
    ],
    "url": [
        (["download it", "fetch it", "grab it", "save it"], lambda c, q: [("download", {"url": c, "to": None})]),
        (["show its headers", "what status does it return"], _c("http_head", "url")),
        (["curl it", "show the response"], _c("http_get", "url")),
        (["clone it", "git clone it"], lambda c, q: [("git_clone", {"url": c, "into": None})]),
        (["open it", "open it in the browser"], _c("open", "target")),
    ],
    "cmdname": [
        (["show its man page", "how do i use it", "help for it", "man it"], _c("man", "cmd")),
        (["where is it installed", "which one is it", "where does it live"], _c("which", "cmd")),
        (["what version is it", "which version do i have"], _c("ver", "cmd")),
    ],
    "envname": [
        (["what is it set to", "show its value", "print it", "echo it"], _c("var_show", "name")),
        (["unset it", "clear it", "remove it"], _c("var_unset", "name")),
        (["set it to {}", "change it to {}", "make it {}"], lambda c, q, k: [("var_set", {"name": c, "value": k})], "varvalue"),
    ],
    "port": [
        (["what's using it", "who is on that port", "what is listening on it"], lambda c, q: [("port", {"n": c}), ("pshow", {})]),
        (["kill whatever is using it", "free it up", "kill the process on it"], lambda c, q: [("port", {"n": c}), ("kill", {"force": "soft"})]),
    ],
}
_PKG_CMDS = ("pkg_install", "pkg_remove", "pkg_search", "pkg_info", "pkg_has", "pkg_files")
NEXT_STEP = [
    (("git_commit", "git_commit_all", "git_merge", "git_amend", "git_add_all", "git_tag"),
     [(["push it", "now push", "push that", "and push", "push it up"], [("git_push", {})]),
      (["undo that", "undo it", "take that back"], [("git_undo", {})])]),
    (("git_stash",), [(["pop it", "bring it back", "unstash", "undo that"], [("git_stash_pop", {})])]),
    (("git_pull", "git_fetch"), [(["show what changed", "what came in", "what's new"], [("git_log", {"n": None})])]),
    (("compose_up",), [(["show the logs", "logs"], [("compose_logs", {})]), (["bring it down", "stop it", "take it down"], [("compose_down", {})])]),
    (("poweroff", "reboot"), [(["cancel that", "never mind, cancel it", "abort", "cancel the shutdown", "wait, cancel"], [("shutdown_cancel", {})])]),
    (("vol_set", "vol_up", "vol_down", "mute"), [(["louder", "a bit more", "more"], [("vol_up", {})]), (["quieter", "too loud", "less"], [("vol_down", {})])]),
    (("bright_set", "bright_up", "bright_down"), [(["brighter", "a bit more"], [("bright_up", {})]), (["dimmer", "too bright", "less"], [("bright_down", {})])]),
    (("venv_make",), [(["activate it", "now activate it", "turn it on", "enter it"], [("venv_on", {"path": None})])]),
    (("git_add_all", "git_add"), [(["unstage that", "undo that"], [("git_unstage", {"file": None})])]),
]


def pronoun_followup(q, kind, c):
    """A new command about "it": c is the context position of the thing "it" refers to."""
    r = q.r
    entry = r.choice(PRON[kind])
    phrase = r.choice(entry[0])
    if len(entry) == 3:
        k = fill(q, phrase, entry[2])
        return [entry[1](c, q, k)]
    q.lit(phrase)
    return [entry[1](c, q)]


def single_followup(q, A, where):
    r = q.r
    stage = A[1] if A[0][0] == "sudo" else A[0]
    j = 1 if A[0][0] == "sudo" else 0
    name = stage[0]
    if name not in commands.BY_NAME:
        return None
    for names, options in NEXT_STEP:
        if name in names and r.random() < 0.5:
            phrases, prog = r.choice(options)
            q.lit(r.choice(phrases))
            if prog[0][0] == "venv_on" and (0, j, "path") in where:
                prog = [("venv_on", {"path": ("ctx", where[(0, j, "path")])})]
            return [prog]
    cands = []
    for slot, kind, _ in commands.BY_NAME[name]["slots"]:
        if (0, j, slot) not in where:
            continue
        kind = "pkg" if name in _PKG_CMDS else kind
        if kind in PRON:
            cands.append((kind, ("ctx", where[(0, j, slot)])))
    if not cands:
        return None
    kind, c = cands[-1]
    return pronoun_followup(q, kind, c)


def chain_request(q):
    """Two steps in one request where the second refers back to the first: "stop nginx and disable it",
    "make a venv and activate it", "commit with message x and push". The second step points at the first's words."""
    r = q.r
    for _ in range(30):
        mark, nkey = len(q.chunks), q.n
        pipe = command_request(q)
        name, args = pipe[0]
        steps = [o for names, options in NEXT_STEP if name in names for o in options]
        cands = []
        for slot, kind, _ in commands.BY_NAME[name]["slots"]:
            kind = "pkg" if name in _PKG_CMDS else kind
            if args.get(slot) is not None and kind in PRON:
                cands.append((kind, args[slot]))
        if not steps and not cands:
            del q.chunks[mark:]
            continue
        q.lit(r.choice(["and", "and", "and then", "then", ", then", "and", "and after that", "and also", ","]))
        if steps and (not cands or r.random() < 0.4):
            phrases, prog = r.choice(steps)
            q.lit(r.choice(phrases))
            if prog[0][0] == "venv_on":
                prog = [("venv_on", {"path": args.get("path")})]
            return [pipe, prog]
        kind, c = cands[-1]
        return [pipe, pronoun_followup(q, kind, c)[0]]
    raise RuntimeError("no chain")


def find_then(q):
    """"find all pdfs in x and delete them": a listing request, then what to do with what was found."""
    r = q.r
    while True:
        mark, nkey = len(q.chunks), q.n
        pipe = gen.files_request(q)
        if pipe[-1][0] == "fshow" and not any(n in ("fnames", "fcat") for n, _ in pipe):
            break
        del q.chunks[mark:]
    q.lit(r.choice(["and", "and", "and then", "then", ", then", "and", ", and"]))
    body = pipe[:-1]
    roll = r.random()
    if roll < 0.6:
        name = r.choice([n for n in FOLLOW_SINK if n != "fshow"])
        q.lit(r.choice(FOLLOW_SINK[name]))
        return body + gen.frame_stages(name, {})
    if roll < 0.9:
        name = r.choice(list(FOLLOW_SINK_ARG))
        phrases, slot, kind = FOLLOW_SINK_ARG[name]
        return body + gen.frame_stages(name, {slot: fill(q, r.choice(phrases), kind)})
    phrase, fmt, ext = r.choice(FOLLOW_ZIP)
    k = None
    if ext:
        a, b = phrase.split("{}")
        q.lit(a); k = q.key(); q.chunks.append((q.word() + r.choice([ext, ext, ""]), k)); q.lit(b)
    else:
        q.lit(phrase)
    return body + [("compress", {"fmt": fmt, "to": k})]


FOLLOW_AGAIN_SUDO = ["again with sudo", "sudo that", "run that as root", "redo it with sudo", "sudo !!", "same but with sudo", "retry as root",
                     "please do that as root", "with sudo", "as root", "do that again as root", "try again with sudo", "use sudo", "sudo",
                     "run it with sudo", "needs sudo", "permission denied, try sudo"]
FOLLOW_OUT = [
    (["search that for {}", "grep that for {}", "only lines with {}", "filter that for {}", "grep {} in the output", "find {} in that output",
      "just the lines containing {}", "search the output for {}", "show only lines mentioning {}", "pipe that to grep {}"], "match", "text", "text"),
    (["hide lines with {}", "that but without {}", "filter out {} from the output", "exclude lines containing {}"], "nomatch", "text", "text"),
    (["count the lines", "how many lines is that", "count that", "how many lines", "count the output lines", "pipe that to wc -l",
      "how many lines did that print"], "lcount", None, None),
    (["save that to {}", "save the output to {}", "write that to {}", "put the output in {}", "dump that into {}", "redirect that to {}",
      "save the output of that to {}"], "save", "to", "textfile"),
    (["append that to {}", "add the output to {}", "append the output to {}"], "lappend", "to", "textfile"),
    (["just the first {} lines", "only the top {}", "first {} lines of that", "show only the first {}", "head that to {} lines"], "ltake", "n", "count"),
    (["just the last {} lines", "last {} lines of that", "only the last {}", "tail that to {} lines"], "llast", "n", "count"),
    (["copy that to the clipboard", "copy the output", "put that output on my clipboard", "copy that"], "clip", None, None),
    (["with line numbers", "number those lines"], "lnum", None, None),
    (["sort that", "sort the output", "sorted", "sort it alphabetically", "pipe it through sort"], "lsort", None, None),
    (["unique lines of that", "sort that and remove duplicates", "sort and dedupe that"], "sortu", None, None),
]
_ENDS = ("lcount", "save", "lappend", "clip")


def output_followup(q):
    """Follow-ups about the previous command as a whole: run it again, as root, or do something with its output."""
    r = q.r
    roll = r.random()
    if roll < 0.24:
        return command_request(q, r.choice(["again", "again", "again", "watch_last", "time_last"]))
    if roll < 0.4:
        q.lit(r.choice(FOLLOW_AGAIN_SUDO))
        return [("sudo", {}), ("again", {})]
    phrases, name, slot, kind = r.choice(FOLLOW_OUT)
    k = fill(q, r.choice(phrases), kind)
    if name == "lsort":
        ops = [("lsort", {"key": "alpha", "order": "asc"})]
    elif name == "sortu":
        ops = [("lsort", {"key": "alpha", "order": "asc"}), ("uniq", {})]
    else:
        ops = [(name, {slot: k} if slot else {})]
    return [("last", {})] + ops + ([] if name in _ENDS else [("lshow", {})])


_PREV = ["files", "lines", "lines2", "procs", "procs2", "single", "paths"]
_PREV_WT = [38, 8, 6, 7, 2, 30, 6]


def previous(r):
    """A previous request, as the context a follow-up will see: (program with real values, context words, positions)."""
    for _ in range(50):
        qa = Req(r)
        pipe = one_pipeline(qa, _PREV, _PREV_WT)
        words, out, _ = finish(qa, [pipe])
        if len(out) > 40:
            continue
        prog = ir.delinearise(out, words)
        ctx, where = ir.context_words(prog)
        if len(ctx) <= 30:
            return prog, ctx, where
    raise RuntimeError("no short previous request")


def raw_context(r):
    return ["@raw"] + tok.words(r.choice(gen.RAW_COMMANDS))[:12]


def followup(r):
    """-> (words, output tokens, number of request words) for a request that builds on the previous one."""
    for _ in range(40):
        q = Req(r)
        if r.random() < 0.16:
            ctx = raw_context(r)
            prog = [output_followup(q)]
        else:
            A, ctx, where = previous(r)
            pipe = A[0]
            head = pipe[1][0] if pipe[0][0] == "sudo" else pipe[0][0]
            if r.random() < 0.3:
                q.lit(r.choice(["ok", "now", "and", "ok now", "great,", "then", "alright", "cool,", "and then", "next", "also", "please", "so"]))
            if head in ("files", "paths"):
                prog = files_followup(q, pipe, where)
            elif head in ("lines", "history", "env", "pkgs", "services", "journal", "dmesg", "clipboard", "last"):
                prog = lines_followup(q, pipe, where)
            elif head in ("procs", "port"):
                prog = procs_followup(q, pipe, where)
            elif r.random() < 0.14:
                prog = [output_followup(q)]
            else:
                prog = single_followup(q, pipe, where)
            if prog is None:
                continue
        noise(q)
        words, out, n = finish(q, prog, ctx)
        if n and len(out) <= ir.MAX_OUT:
            return words, out, n
    raise RuntimeError("no follow-up could be made")


# --------------------------------------------------------------------------- surface noise and whole examples
FILLERS = ["uh", "like", "just", "really", "actually", "basically", "um", "kinda", "maybe", "quickly"]
DROPPABLE = {"the", "a", "an", "all", "me", "my", "of", "that", "are", "is"}


def noise(q):
    """Small imperfections on the literal (non-argument) words: case, typos, a filler word, a dropped small word."""
    r = q.r
    style = r.random()
    chunks = []
    for text, key in q.chunks:
        if key is None:
            if style < 0.72:
                text = text.lower()
            if r.random() < 0.02:
                ws = text.split()
                if ws:
                    j = r.randrange(len(ws)); ws[j] = gen.typo(r, ws[j]); text = " ".join(ws)
        chunks.append((text, key))
    if 0.72 <= style < 0.92 and chunks and chunks[0][1] is None and chunks[0][0]:
        chunks[0] = (chunks[0][0][0].upper() + chunks[0][0][1:], None)
    if r.random() < 0.04:
        lits = [i for i, (t, k) in enumerate(chunks) if k is None and len(t.split()) > 1]
        if lits:
            i = r.choice(lits)
            ws = chunks[i][0].split()
            drop = [j for j, w in enumerate(ws) if w.lower() in DROPPABLE]
            if drop:
                del ws[r.choice(drop)]
                chunks[i] = (" ".join(ws), None)
    if r.random() < 0.035 and chunks:
        chunks.insert(r.randrange(len(chunks) + 1) if r.random() < 0.5 else 0, (r.choice(FILLERS), None))
    if r.random() < 0.12 and chunks and chunks[-1][1] is None:
        chunks[-1] = (chunks[-1][0] + r.choice(["?", ".", "!", "?"]), None)
    q.chunks = [c for c in chunks if c[0]]


def example(r):
    """One training pair: (words, output tokens, number of request words). Words after the request are the context."""
    if r.random() < 0.3:
        return followup(r)
    q = Req(r)
    roll = r.random()
    if roll < 0.035:
        program = gen.mkdir_then(q)
    elif roll < 0.075:
        program = chain_request(q)
    elif roll < 0.115:
        program = [find_then(q)]
    elif roll < 0.2:
        program = [one_pipeline(q)]
        q.lit(r.choice(JOIN))
        program.append(one_pipeline(q))
        if r.random() < 0.15:
            q.lit(r.choice(JOIN))
            program.append(one_pipeline(q))
    else:
        program = [one_pipeline(q)]
    ctx = None
    if r.random() < 0.3:                      # an unrelated earlier request is in the session: it must be ignored
        ctx = raw_context(r) if r.random() < 0.3 else previous(r)[1]
    noise(q)
    return finish(q, program, ctx)


MAX_REQUEST, MAX_TOTAL = 44, 80


def dataset(n, seed):
    r = random.Random(seed)
    data = []
    while len(data) < n:
        words, out, n_req = example(r)
        if 1 <= n_req <= MAX_REQUEST and len(words) <= MAX_TOTAL and len(out) <= ir.MAX_OUT:
            data.append((words, out, n_req))
    return data


if __name__ == "__main__":
    import sys
    from .compile import Env, Unsupported, compile_program
    r = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    only = sys.argv[3] if len(sys.argv) > 3 else None
    shown = 0
    while shown < (int(sys.argv[2]) if len(sys.argv) > 2 else 25):
        words, out, n = followup(r) if only == "follow" else example(r)
        prog = ir.delinearise(out, words)
        try:
            cmd = compile_program(prog, Env(live=False, last="<last command>"))
        except Unsupported:
            cmd = "(declined)"
        print(" ".join(words[:n]) + ("      [after: " + " ".join(words[n:]) + "]" if len(words) > n else ""))
        print("   ->", cmd or "(preference)")
        shown += 1
