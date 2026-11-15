"""The typed program language that plain-English requests are parsed into.

A program is one to three pipelines. A pipeline is a source stage, any number of
filter/transform stages, and one final action stage; or a single stand-alone command
(those live in lang/commands.txt). The stages themselves are listed in lang/stages.txt. Every stage has a fixed list of slots. A slot is filled either
by a span of words (copied from the request or from the session context, never generated)
or by one value from a small fixed set.

The model only ever emits tokens from this language, and `Grammar` restricts each
step to what is valid at that point, so every decoded program is well-formed.
"""
import os
from dataclasses import dataclass

from . import commands

# data types that flow between stages
F, L, P, X = "files", "lines", "procs", "done"
NEG, SUDO = "neg", "sudo"        # grammar states: after "not", after "sudo"

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


def _load_stages():
    """kural/lang/stages.txt (shared with the Rust code), with the commands of commands.txt where it says *commands."""
    enums, lists, stages = {}, {}, []
    types = {"-": None, "files": F, "lines": L, "procs": P, "done": X, "neg": NEG, "sudo": SUDO}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lang", "stages.txt")
    for line in open(path, encoding="utf-8"):
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if f[0] == "%":
            enums[f[1]] = f[2:]
        elif f[0] == "&":
            lists[f[1]] = f[2:]
        elif f[0] == "*commands":
            stages += [S(c["name"], None, X, *[span(n, t, opt) for n, t, opt in c["slots"]]) for c in commands.COMMANDS]
        elif f[0] == "@":
            slots = []
            for spec in f[4:]:
                if "=" in spec:
                    name, kind = spec.split("=")
                    slots.append(Slot(name, kind))
                else:
                    name, kind = spec.split(":")
                    slots.append(span(name, kind.rstrip("?"), kind.endswith("?")))
            stages.append(S(f[1], types[f[2]], types[f[3]], *slots))
    return enums, lists, stages


ENUMS, _LISTS, STAGES = _load_stages()
BY_NAME = {s.name: s for s in STAGES}
assert len(BY_NAME) == len(STAGES), "duplicate stage name"

NEGATABLE = _LISTS["negatable"]
FILTER_ORDER = _LISTS["filter_order"]

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
