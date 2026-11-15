"""Command line: plainsh "delete log files older than 30 days in ~/Downloads"

Prints the bash command for the request. It never runs anything itself.
  plainsh REQUEST...           the command, on stdout
  plainsh -v REQUEST...        also the reading of the request, confidence and alternatives (stderr)
  plainsh --shell REQUEST...   for the shell hook: line 1 command, line 2 risk (0-2), line 3 confidence, line 4 reading
  plainsh --serve              answer requests from shell hooks until idle (started by the hook, not by hand)
Options: --session FILE (the terminal's session file, for follow-ups), --cwd DIR
"""
import os
import sys

from . import commands
from . import compile as C
from . import lexicon, run, session

# A destructive action is only accepted when the request itself contains a word that asks for it.
# This is a plain rule, independent of the model, so a question can never turn into a delete or a kill.
ASKS_FOR = {
    "delete": ("delet", "remov", "rm", "eras", "wip", "trash", "purg", "nuk", "clean", "clear", "discard", "destroy", "drop", "rid", "throw"),
    "kill": ("kill", "stop", "terminat", "end", "clos", "quit", "pkill", "nuk", "murder", "shut", "exit", "free"),
    "move": ("mov", "mv", "transfer", "relocat", "shift", "put", "send", "dump", "stick", "renam"),
    "trash": ("trash", "bin", "recycl", "delet", "remov", "rid", "throw", "send", "put", "mov", "dump", "toss"),
    "replace": ("replac", "chang", "swap", "substitut", "sed", "turn", "renam", "updat", "fix", "switch", "convert"),
    "inplace": ("place", "in-place", "save", "overwrit", "writ", "perman", "updat", "good", "replac", "chang", "swap", "substitut",
                "turn", "renam", "sed", "edit", "modif", "fix"),
}
_RENAME = ("renam", "lower", "upper", "caps", "capital", "replac", "chang", "swap", "prefix", "prepend", "suffix", "append", "add", "put",
           "extens", "space", "underscor", "fix", "turn", "convert", "mv", "make", "switch", "give", "remov", "call")
ASKS_FOR.update({n: _RENAME for n in ("rn_lower", "rn_upper", "rn_spaces", "rn_replace", "rn_prefix", "rn_suffix", "rn_ext")})
# stand-alone commands that remove or stop things: the request must contain one of the verbs their own phrasings start with
ASKS_FOR.update({c["name"]: commands.verbs(c) for c in commands.COMMANDS if c["risk"] >= 2})
SAFE_INSTEAD = {"kill": "pshow", "inplace": "lshow"}      # every other guarded file action falls back to listing


def unasked(program, text, context=()):
    """Names of destructive stages in the program that no word of the request asks for."""
    words = [w.strip(".,!?'\"").lower() for w in text.split()]
    before = {w[1:] for w in context if w.startswith("@")}        # a follow-up may keep what the earlier request asked for
    return [name for pipe in program for name, _ in pipe
            if name in ASKS_FOR and name not in before and not any(w.startswith(stem) for w in words for stem in ASKS_FOR[name])]


def _safe(program):
    """The same selection with every unasked destructive action replaced by a listing, or None."""
    out = []
    for pipe in program:
        new = []
        for name, a in pipe:
            if name not in ASKS_FOR:
                new.append((name, a))
            elif name in SAFE_INSTEAD:
                new.append((SAFE_INSTEAD[name], {}))
            elif name in commands.BY_NAME:
                new = []
                break
            else:
                new.append(("fshow", {}))
        if new:
            out.append(new)
    return out or None


def translate(text, env=None, context=None):
    """-> list of dicts: command, confidence, risk, reading, program. Best first; empty if nothing parsed."""
    env = C.as_env(env)
    out, held, foreign = [], [], False
    for i, (program, prob, words, used) in enumerate(run.parse(text, context)):
        e = env.fresh()
        try:
            if unasked(program, text, context or ()):
                held.append((program, prob))
                continue
            cmd = C.compile_program(program, e)
        except C.Unsupported:
            if i == 0:
                return []          # the parser's best reading is "I can't do that": decline
            continue
        except Exception:
            continue
        fit = lexicon.support(program, words, used)
        if fit is not None and fit[0] == 0:
            if i == 0 and prob < 0.999:
                foreign = True     # none of the request's own words belong to the best reading
            continue
        if fit is not None and fit[0] * 2 < fit[1]:
            prob = min(prob, 0.6)  # most of the request is unexplained: say so
        reading = C.describe(program) + "".join(f"  ({n})" for n in e.notes)
        out.append({"command": cmd, "confidence": prob, "risk": C.risk(program), "reading": reading, "program": program})
    if foreign:
        return []
    if not out and held:
        # every reading was destructive without being asked for: show what it would have acted on instead
        program, prob = held[0]
        safe = _safe(program)
        if safe:
            try:
                out.append({"command": C.compile_program(safe, env.fresh()), "confidence": prob * 0.5, "risk": 0,
                            "reading": C.describe(safe) + "  (listing only)", "program": safe})
            except Exception:
                pass
    return out


def answer(text, cwd=None, session_path=None, final=True, live=True, pm=None, choose=0):
    """One request from a shell: reads the session file for context, and on a final request records what was offered.
    choose: which candidate the user picked (it is moved to the front)."""
    st = session.read(session_path, cwd)
    env = C.Env(cwd=cwd, live=live, pm=pm, prefs=st.prefs, last=st.last, final=final)
    results = translate(text, env, st.context)
    fixed = st.fixes.get(session.norm(text))
    if fixed and not any(r["command"] == fixed for r in results[:1]):
        results.insert(0, {"command": fixed, "confidence": 1.0, "risk": 1, "reading": "what you ran last time you asked this", "program": None})
    results = results[:1] + [r for r in results[1:] if r["confidence"] >= 0.03]       # alternatives worth showing
    if 0 < choose < len(results):
        results.insert(0, results.pop(choose))
    if results:
        best = results[0]
        prog = best["program"]
        if prog and prog[0][0][0] == "pref":
            key, state = prog[0][0][1]["key"], prog[0][0][1]["state"]
            best["reading"] = {"trash": "deletes go to the trash", "confirm": "ask before deleting or overwriting",
                               "verbose": "commands print each file"}[key] + (": on" if state == "on" else ": off") + " (this session)"
            if final:
                session.set_pref(session_path, key, state)
        elif final and prog:
            session.record(session_path, text, cwd, prog, best["command"])
    return results


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    verbose = shell = False
    sess = os.environ.get("PLAINSH_SESSION")
    cwd = None
    while args and args[0].startswith("-") and args[0] != "--":
        flag = args.pop(0)
        if flag == "-v":
            verbose = True
        elif flag == "--shell":
            shell = True
        elif flag == "--serve":
            from . import serve
            return serve.main()
        elif flag == "--session" and args:
            sess = args.pop(0)
        elif flag == "--cwd" and args:
            cwd = args.pop(0)
        else:
            break
    text = " ".join(args).strip()
    if not text:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    results = answer(text, cwd=cwd or os.getcwd(), session_path=sess)
    if not results:
        print("plainsh: that is outside what I can do so far", file=sys.stderr)
        return 1
    best = results[0]
    if shell:
        print(best["command"]); print(best["risk"]); print(f"{best['confidence']:.3f}"); print(best["reading"])
        return 0
    print(best["command"] or f"({best['reading']})")
    if verbose:
        print(f"  reading:    {best['reading']}", file=sys.stderr)
        print(f"  confidence: {100 * best['confidence']:.0f}%", file=sys.stderr)
        for alt in results[1:]:
            print(f"  or ({100 * alt['confidence']:.0f}%): {alt['command']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
