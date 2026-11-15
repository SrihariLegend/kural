"""The typed program language that plain-English requests are parsed into.

A program is one to three pipelines. A pipeline is a source stage, any number of
filter/transform stages, and one final action stage; or a single stand-alone command
(those live in commands.py). Every stage has a fixed list of slots. A slot is filled either
by a span of words (copied from the request or from the session context, never generated)
or by one value from a small fixed set.

The model only ever emits tokens from this language, and `Grammar` restricts each
step to what is valid at that point, so every decoded program is well-formed.
"""
from dataclasses import dataclass

from . import commands

# data types that flow between stages
F, L, P, X = "files", "lines", "procs", "done"
NEG, SUDO = "neg", "sudo"        # grammar states: after "not", after "sudo"

ENUMS = {
    "mode": ["is", "has", "starts", "ends"],
    "fkey": ["size", "time", "name"],
    "lkey": ["alpha", "num"],
    "pkey": ["mem", "cpu"],
    "order": ["asc", "desc"],
    "fmt": ["tgz", "zip"],
    "force": ["soft", "hard"],
    "what": ["anyfile", "dirs"],
    "sep": ["space", "comma", "tab", "colon", "semi", "bar"],
    "prefkey": ["trash", "confirm", "verbose"],
    "onoff": ["on", "off"],
}


@dataclass(frozen=True)
class Slot:
    name: str
    kind: str            # "span" or an ENUMS key
    value: str = ""      # for spans: how the copied words are interpreted (path, size, dur, count, text, ...)
    optional: bool = False


@dataclass(frozen=True)
class Stage:
    name: str
    src: object          # input type, or None for a stage that starts a pipeline
    dst: str             # output type; X ends the pipeline
    slots: tuple = ()


def S(name, src, dst, *slots):
    return Stage(name, src, dst, tuple(slots))


def span(name, value, optional=False):
    return Slot(name, "span", value, optional)


STAGES = [
    # ---- "run it as root": may come first in any pipeline
    S("sudo", None, SUDO),
    # ---- sources
    S("files", None, F, Slot("what", "what"), span("in", "path", True)),
    S("paths", None, F, span("names", "files")),                 # files the user named outright
    S("lines", None, L, span("of", "file")),
    S("procs", None, P),
    S("port", None, P, span("n", "port")),                       # whatever is listening on a port
    S("history", None, L), S("env", None, L), S("pkgs", None, L), S("services", None, L),
    S("journal", None, L, span("unit", "unit", True)),
    S("dmesg", None, L), S("clipboard", None, L),
    S("last", None, L),                                          # the output of the previous command, run again
    # ---- file filters
    S("not", F, NEG),                                            # negates the filter that follows
    S("flat", F, F),
    S("depth", F, F, span("n", "count")),
    S("ext", F, F, span("ext", "ext")),
    S("name", F, F, Slot("mode", "mode"), span("text", "glob")),
    S("larger", F, F, span("size", "size", True)),
    S("smaller", F, F, span("size", "size", True)),
    S("newer", F, F, span("age", "dur", True)),
    S("older", F, F, span("age", "dur", True)),
    S("since", F, F, span("date", "date")),
    S("before", F, F, span("date", "date")),
    S("contains", F, F, span("text", "text")),
    S("empty", F, F),
    S("hidden", F, F),
    S("owner", F, F, span("user", "owner")),
    S("isexec", F, F),
    # ---- line filters and edits
    S("match", L, L, span("text", "text")),
    S("nomatch", L, L, span("text", "text")),
    S("uniq", L, L),
    S("lrange", L, L, span("a", "count"), span("b", "count")),
    S("lline", L, L, span("n", "count")),
    S("ldel", L, L, span("n", "count")),                         # drop one line by its number
    S("col", L, L, span("n", "count"), Slot("sep", "sep")),
    S("lreplace", L, L, span("old", "text"), span("new", "text")),
    S("lfreq", L, L), S("lnum", L, L), S("lrev", L, L), S("lshuf", L, L),
    S("lupper", L, L), S("llower", L, L), S("lsum", L, L), S("lblank", L, L),
    # ---- process filters
    S("pname", P, P, span("name", "word")),
    S("puser", P, P, span("user", "owner")),
    # ---- transforms (same type in and out); one entry per input type keeps the grammar simple
    S("fsort", F, F, Slot("key", "fkey"), Slot("order", "order")),
    S("lsort", L, L, Slot("key", "lkey"), Slot("order", "order")),
    S("psort", P, P, Slot("key", "pkey"), Slot("order", "order")),
    S("ftake", F, F, span("n", "count", True)),
    S("fpick", F, F, span("n", "count")),                        # "the second one"
    S("ltake", L, L, span("n", "count", True)),
    S("ptake", P, P, span("n", "count", True)),
    S("llast", L, L, span("n", "count", True)),
    # ---- files become lines
    S("fnames", F, L),                                           # their paths, one per line
    S("fcat", F, L),                                             # their contents
    # ---- actions that end a pipeline
    S("fshow", F, X), S("lshow", L, X), S("pshow", P, X),
    S("fcount", F, X), S("lcount", L, X), S("pcount", P, X),
    S("total", F, X),
    S("sizes", F, X),
    S("details", F, X),
    S("delete", F, X),
    S("trash", F, X),
    S("move", F, X, span("to", "path")),
    S("copy", F, X, span("to", "path")),
    S("compress", F, X, Slot("fmt", "fmt"), span("to", "file", True)),
    S("fgrep", F, X, span("text", "text")),
    S("fopen", F, X), S("fedit", F, X),
    S("chmod", F, X, span("mode", "mode")),
    S("chown", F, X, span("owner", "owner")),
    S("rn_lower", F, X), S("rn_upper", F, X), S("rn_spaces", F, X),
    S("rn_replace", F, X, span("old", "text"), span("new", "text")),
    S("rn_prefix", F, X, span("text", "text")),
    S("rn_suffix", F, X, span("text", "text")),
    S("rn_ext", F, X, span("ext", "ext")),
    S("replace", F, X, span("old", "text"), span("new", "text")),
    S("fhash", F, X), S("by_ext", F, X), S("dupes", F, X),
    S("upload", F, X, span("to", "remote")),
    S("kill", P, X, Slot("force", "force")),
    S("wcount", L, X), S("ccount", L, X),
    S("save", L, X, span("to", "file")),
    S("lappend", L, X, span("to", "file")),
    S("inplace", L, X),                                          # write the result back into the file it came from
    S("clip", L, X),
    # ---- session preferences
    S("pref", None, X, Slot("key", "prefkey"), Slot("state", "onoff")),
]
# ---- stand-alone commands, from the table in commands.py
STAGES += [S(c["name"], None, X, *[span(n, t, opt) for n, t, opt in c["slots"]]) for c in commands.COMMANDS]
# ---- "this is not something I can do": lets the parser decline instead of guessing
STAGES.append(S("unknown", None, X))
BY_NAME = {s.name: s for s in STAGES}
assert len(BY_NAME) == len(STAGES), "duplicate stage name"

NEGATABLE = ["ext", "name", "newer", "older", "contains", "empty", "hidden", "owner", "isexec"]
FILTER_ORDER = ["flat", "depth", "ext", "name", "larger", "smaller", "newer", "older", "since", "before", "contains",
                "empty", "hidden", "owner", "isexec"]

SPECIAL = ["<pad>", "<bos>", "<end>", ";", "<none>"]
VOCAB = SPECIAL + [s.name for s in STAGES] + sorted({v for vs in ENUMS.values() for v in vs} - {s.name for s in STAGES})
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, END, SEP, NONE = (TOK[t] for t in SPECIAL)
NV = len(VOCAB)
MAX_PIPELINES = 3
MAX_OUT = 72          # longest program in output tokens

_STARTS = [TOK[s.name] for s in STAGES if s.src is None]
_NEXT = {t: [TOK[s.name] for s in STAGES if s.src == t] for t in (F, L, P)}
_NEXT[NEG] = [TOK[n] for n in NEGATABLE]
_NEXT[SUDO] = [t for t in _STARTS if VOCAB[t] not in ("sudo", "unknown", "pref")]


class Grammar:
    """Tracks where we are in a program and says which output tokens are legal next.

    Output tokens are vocabulary ids (< NV) or pointers to input words (NV + position).
    """

    def __init__(self, n_words):
        self.n = n_words
        self.type = None          # data type flowing, None at the start of a pipeline
        self.pending = []         # slots still to fill for the current stage
        self.span_start = None
        self.pipelines = 1
        self.done = False

    def copy(self):
        g = Grammar(self.n)
        g.type, g.pending, g.span_start, g.pipelines, g.done = self.type, list(self.pending), self.span_start, self.pipelines, self.done
        return g

    def allowed(self):
        if self.done:
            return [PAD]
        if self.pending:
            slot = self.pending[0]
            if slot.kind == "span":
                if self.span_start is None:
                    out = [NV + i for i in range(self.n)]
                    return out + [NONE] if slot.optional else out
                return [NV + i for i in range(self.span_start, self.n)]
            return [TOK[v] for v in ENUMS[slot.kind]]
        if self.type is None:
            return _STARTS
        if self.type == X:
            return [END, SEP] if self.pipelines < MAX_PIPELINES else [END]
        return _NEXT[self.type]

    def advance(self, tok):
        if self.pending:
            slot = self.pending[0]
            if slot.kind == "span" and tok >= NV and self.span_start is None:
                self.span_start = tok - NV
                return
            self.span_start = None
            self.pending.pop(0)
            return
        if tok == END:
            self.done = True
        elif tok == SEP:
            self.type, self.pipelines = None, self.pipelines + 1
        else:
            st = BY_NAME[VOCAB[tok]]
            self.type, self.pending = st.dst, list(st.slots)


def linearise(program):
    """program: list of pipelines; pipeline: list of (stage_name, {slot: value}).
    A span value is (start, end) word positions (inclusive) or None; an enum value is its string.
    Returns the output token ids, ending with END."""
    out = []
    for i, pipe in enumerate(program):
        if i:
            out.append(SEP)
        for name, args in pipe:
            out.append(TOK[name])
            for slot in BY_NAME[name].slots:
                v = args.get(slot.name)
                if slot.kind == "span":
                    if v is None:
                        out.append(NONE)
                    else:
                        out += [NV + v[0], NV + v[1]]
                else:
                    out.append(TOK[v])
    return out + [END]


def delinearise(tokens, words):
    """Inverse of linearise, with spans turned back into words (a string)."""
    program, pipe, i = [], [], 0
    while i < len(tokens):
        t = tokens[i]; i += 1
        if t in (END, PAD):
            break
        if t == SEP:
            program.append(pipe); pipe = []
            continue
        st = BY_NAME[VOCAB[t]]
        args = {}
        for slot in st.slots:
            v = tokens[i]; i += 1
            if slot.kind == "span":
                if v == NONE:
                    args[slot.name] = None
                else:
                    e = tokens[i]; i += 1
                    args[slot.name] = " ".join(words[v - NV:e - NV + 1])
            else:
                args[slot.name] = VOCAB[v]
        pipe.append((st.name, args))
    if pipe:
        program.append(pipe)
    return program


def spans_used(tokens):
    """Positions of the input words that the program copied as arguments."""
    used, i = set(), 0
    toks = list(tokens)
    while i < len(toks):
        t = toks[i]; i += 1
        if t in (END, PAD):
            break
        if t == SEP or t >= NV:
            continue
        for slot in BY_NAME[VOCAB[t]].slots:
            v = toks[i]; i += 1
            if slot.kind == "span" and v != NONE:
                e = toks[i]; i += 1
                used.update(range(v - NV, e - NV + 1))
    return used


# ---- a program as context for the next request: stage names and values as a flat list of words.
# Each span value is ONE word (which may contain spaces), so a follow-up can point at it whole.
def context_words(program):
    """-> (words, positions) where positions[(pipe, stage, slot)] is the index of that value's word."""
    words, where = [], {}
    for i, pipe in enumerate(program):
        if i:
            words.append("@;")
        for j, (name, args) in enumerate(pipe):
            words.append("@" + name)
            for slot in BY_NAME[name].slots:
                v = args.get(slot.name)
                if slot.kind != "span":
                    words.append("@" + v)
                elif v is None:
                    words.append("@none")
                else:
                    where[(i, j, slot.name)] = len(words)
                    words.append(str(v))
    return words, where
