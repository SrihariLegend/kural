"""Turning a parsed program into a bash command line. Deterministic: same program, same machine, same command.

File pipelines pass NUL-separated paths between tools so names with spaces or newlines are safe.
Simple requests compile to the short, familiar form (a single `find`, `grep`, `pkill`, ...).
Facts about the machine (does this file exist, which tool is installed) come from ground.Env.
"""
import re
import shlex

from . import commands
from . import values as V
from .ground import Env, as_env, package_manager  # noqa: F401  (re-exported)

q = shlex.quote

# how careful the shell hook should be before running the command: 2 = removes or stops things
RISK = {"delete": 2, "kill": 2, "replace": 2, "inplace": 2, "trash": 1, "move": 1, "save": 1, "lappend": 1, "chmod": 1,
        "chown": 1, "rn_lower": 1, "rn_upper": 1, "rn_spaces": 1, "rn_replace": 1, "rn_prefix": 1, "rn_suffix": 1,
        "rn_ext": 1, "upload": 1, "sudo": 1}
RISK.update({c["name"]: c["risk"] for c in commands.COMMANDS})

LINE_SOURCES = ("lines", "history", "env", "pkgs", "services", "journal", "dmesg", "clipboard", "last")


class Unsupported(Exception):
    """The request is outside what plainsh can do."""


def _file(text, env):
    return V.quote_path(env.file(text.strip()))


def _new(text):
    """A file the command will create: used exactly as typed."""
    return V.quote_path(text.strip())


def _dir(text, env):
    return V.quote_path(env.folder(V.path(text, env.isdir)))


def _sed_pat(t):
    return re.sub(r"([\\/.*\[\]^$])", r"\\\1", t)


def _sed_rep(t):
    return re.sub(r"([\\/&])", r"\\\1", t)


def _words(text):
    return " ".join(q(w) for w in text.replace(",", " ").split() if w.lower() not in ("and", "&", "plus", "also"))


# --------------------------------------------------------------------------- files
_VAGUE_SORT = {"larger": ("size", "desc"), "smaller": ("size", "asc"), "older": ("time", "asc"), "newer": ("time", "desc")}
_VAGUE_NOTE = {"larger": "no size given: biggest first", "smaller": "no size given: smallest first",
               "older": "no age given: oldest first", "newer": "no age given: newest first"}
_RENAME = {
    "rn_lower": 't="$d/${b,,}"', "rn_upper": 't="$d/${b^^}"', "rn_spaces": 't="$d/${b// /_}"',
    "rn_replace": 't="$d/${b//"$o"/$n}"', "rn_prefix": 't="$d/$p$b"',
    "rn_suffix": 'case $b in *.*) t="$d/${b%.*}$p.${b##*.}";; *) t="$d/$b$p";; esac', "rn_ext": 't="$d/${b%.*}.$p"',
}


def _rename_loop(sink, sa, S, v):
    pre = ""
    if sink == "rn_replace":
        pre = f"o={q(sa['old'])} n={q(sa['new'])}; "
    elif sink in ("rn_prefix", "rn_suffix"):
        pre = f"p={q(sa['text'])}; "
    elif sink == "rn_ext":
        pre = f"p={q(V.extensions(sa['ext'])[0])}; "
    loop = ("while IFS= read -r -d '' f; do d=$(dirname -- \"$f\") b=$(basename -- \"$f\"); " + _RENAME[sink]
            + f'; [ "$f" = "$t" ] || {S}mv -n{v} -- "$f" "$t"; done')
    return pre, loop


def _archive(fmt, to, env, default="archive"):
    name = (to or default).strip()
    if not name.endswith((".zip", ".tar.gz", ".tgz")):
        name += ".zip" if fmt == "zip" else ".tar.gz"
    return V.quote_path(name)


def _fix_fmt(sa):
    to = (sa.get("to") or "").lower()
    if to.endswith(".zip"):
        sa["fmt"] = "zip"
    elif to.endswith((".tar.gz", ".tgz")):
        sa["fmt"] = "tgz"


def _direct(raw, sink, sa, rest, env, sudo):
    """Actions on files the user named outright: the plain command, no find."""
    S = "sudo " if sudo else ""
    v = " -v" if env.prefs.get("verbose") else ""
    i = " -i" if env.prefs.get("confirm") else ""
    files = " ".join(V.quote_path(n) for n in raw)
    anydir = any(env.isdir(n) for n in raw)
    if sink == "delete" and env.prefs.get("trash"):
        sink = "trash"
    if sink in ("fshow", "details"):
        return f"{S}ls -lhd -- {files}"
    if sink == "fcount":
        return f"ls -d -- {files} | wc -l"
    if sink == "total":
        return f"{S}du -ch -- {files} | tail -n 1"
    if sink == "sizes":
        return f"{S}du -sh -- {files}"
    if sink == "delete":
        return f"{S}rm{' -r' if anydir else ''}{i}{v} -- {files}"
    if sink == "trash":
        return f"{'gio trash' if env.has('gio') else 'trash-put'} -- {files}"
    if sink in ("move", "copy"):
        tool = ("mv" if sink == "move" else "cp -r") + i + v
        dest = V.path(sa["to"], env.isdir)
        if len(raw) == 1 and "." in dest.rsplit("/", 1)[-1].lstrip(".") and not env.isdir(dest):
            return f"{S}{tool} -- {files} {V.quote_path(dest)}"          # a new name, not a folder
        return f"{S}{tool} -t {_dir(sa['to'], env)} -- {files}"
    if sink == "compress":
        _fix_fmt(sa)
        name = _archive(sa["fmt"], sa.get("to"), env, default=raw[0].rstrip("/").rsplit("/", 1)[-1])
        return f"{S}zip -qr {name} {files}" if sa["fmt"] == "zip" else f"{S}tar -czf {name} {files}"
    if sink == "fgrep":
        return f"{S}grep -rni -- {q(sa['text'])} {files}"
    if sink == "fopen":
        return f"xdg-open {files}" if len(raw) == 1 else f'for f in {files}; do xdg-open "$f"; done'
    if sink == "fedit":
        return f"{S}${{EDITOR:-nano}} {files}"
    if sink == "chmod":
        return f"{S}chmod {V.mode(sa['mode'])} -- {files}"
    if sink == "chown":
        return f"sudo chown {V.owner(sa['owner'])} -- {files}"
    if sink.startswith("rn_"):
        pre, loop = _rename_loop(sink, sa, S, v)
        return f"{pre}printf '%s\\0' {files} | {loop}"
    if sink == "replace":
        return f"{S}sed -i 's/{_sed_pat(sa['old'])}/{_sed_rep(sa['new'])}/g' -- {files}"
    if sink == "fhash":
        return f"{S}sha256sum -- {files}"
    if sink == "upload":
        return f"rsync -avh --progress {files} {q(sa['to'].strip())}"
    if sink == "fnames":
        return _line_ops(f"printf '%s\\n' {files}", None, rest, env, sudo)
    if sink == "fcat":
        if len(raw) == 1:
            return _line_ops(None, files, rest, env, sudo)
        return _line_ops(f"{S}cat -- {files}", None, rest, env, sudo)
    return None


def _files(pipe, env, sudo):
    head, src = pipe[0]
    explicit = head == "paths"
    if explicit:
        raw = [env.file(n) for n in V.names(src["names"])] or ["."]
        roots, dirs = [V.quote_path(n) for n in raw], False
    else:
        dirs = src["what"] == "dirs"
        roots = [_dir(src["in"], env) if src.get("in") else "."]
    tests, exts_only, depth, contains, ncontains, sort, take, pick, vague = [], [], None, None, None, None, None, None, None
    sink, sa, rest, neg = "fshow", {}, [], False
    stages = pipe[1:]
    for i, (name, a) in enumerate(stages):
        if name == "not":
            neg = True
            continue
        bang, neg = ("! " if neg else ""), False
        if name == "flat":
            depth = 1
        elif name == "depth":
            depth = V.count(a["n"])
        elif name == "ext":
            exts = V.extensions_any(a["ext"])
            if len(exts) == 1:
                tests.append(f"{bang}-iname {q('*.' + exts[0])}")
            else:
                tests.append(bang + "\\( " + " -o ".join(f"-iname {q('*.' + e)}" for e in exts) + " \\)")
            exts_only = exts_only + exts if not bang and exts_only is not None else None
            continue
        elif name == "name":
            t = a["text"]
            pat = t if any(c in t for c in "*?[") else {"is": t, "has": f"*{t}*", "starts": f"{t}*", "ends": f"*{t}"}[a["mode"]]
            tests.append(f"{bang}-iname {q(pat)}")
        elif name in ("larger", "smaller"):
            if a.get("size") is None:
                vague = name
            else:
                tests.append(f"-size {'+' if name == 'larger' else '-'}{V.find_size(a['size'])}")
        elif name in ("newer", "older"):
            if a.get("age") is None:
                vague = name
            else:
                tests.append(bang + V.find_age(a["age"], name == "newer"))
        elif name == "since":
            tests.append(f"-newermt {q(V.date(a['date']))}")
        elif name == "before":
            tests.append(f"! -newermt {q(V.date(a['date']))}")
        elif name == "contains":
            if bang:
                ncontains = a["text"]
            else:
                contains = a["text"]
        elif name == "empty":
            tests.append(bang + "-empty")
        elif name == "hidden":
            tests.append(f"{bang}-name {q('.*')}")
        elif name == "owner":
            tests.append(f"{bang}-user {V.owner(a['user'])}")
        elif name == "isexec":
            tests.append(bang + "-executable")
        elif name == "fsort":
            sort = (a["key"], a["order"])
        elif name == "ftake":
            take = V.count(a.get("n"))
        elif name == "fpick":
            pick = V.count(a["n"])
        elif name in ("fnames", "fcat"):
            sink, rest = name, stages[i + 1:]
            break
        else:
            sink, sa = name, dict(a)
        if name != "flat":
            exts_only = None if name in ("depth", "name", "larger", "smaller", "newer", "older", "since", "before", "empty",
                                         "hidden", "owner", "isexec") else exts_only
    if vague:
        if sink in ("fshow", "sizes", "details") and sort is None:
            sort, take = _VAGUE_SORT[vague], take or V.SHOW_TOP
            env.note(_VAGUE_NOTE[vague])
        else:
            tests.append({"larger": f"-size +{V.BIG}", "smaller": f"-size -{V.SMALL}", "older": f"-mtime +{V.OLD_DAYS - 1}",
                          "newer": f"-mtime -{V.NEW_DAYS}"}[vague])
            env.note({"larger": f"no size given: over {V.BIG}", "smaller": f"no size given: under {V.SMALL}",
                      "older": f"no age given: over {V.OLD_DAYS} days", "newer": f"no age given: last {V.NEW_DAYS} days"}[vague])
            exts_only = None
    plain = contains is None and ncontains is None and sort is None and take is None and pick is None
    if explicit and plain and not tests and depth is None:
        out = _direct(raw, sink, sa, rest, env, sudo)
        if out is not None:
            return out
    if sink == "chown":
        sudo = True
    if sink == "delete" and env.prefs.get("trash"):
        sink = "trash"
    S = "sudo " if sudo else ""
    v = " -v" if env.prefs.get("verbose") else ""
    ask = " -i" if env.prefs.get("confirm") else ""
    rename = sink.startswith("rn_")
    base = " ".join([f"{S}find"] + roots + (["-depth"] if rename and dirs else []) + ([f"-maxdepth {depth}"] if depth else [])
                    + (["-mindepth 1"] if dirs else []) + [f"-type {'d' if dirs else 'f'}"] + tests)

    # ---- the general form: a NUL-separated stream of paths
    chain = None
    keyed = sort is not None and sort[0] in ("size", "time")
    if not plain:
        grep = ""
        if contains is not None:
            grep += f" | xargs -0 -r grep -lZ -- {q(contains)}"
        if ncontains is not None:
            grep += f" | xargs -0 -r grep -LZ -- {q(ncontains)}"
        if keyed:
            if grep:
                stat = "%s\\t%n\\0" if sort[0] == "size" else "%Y\\t%n\\0"
                chain = base + " -print0" + grep + f" | xargs -0 -r stat --printf {q(stat)}"
            else:
                chain = base + " -printf " + q(("%s" if sort[0] == "size" else "%T@") + "\\t%p\\0")
            chain += " | sort -z -n" + ("r" if sort[1] == "desc" else "")
            if take:
                chain += f" | head -z -n {take}"
            chain += " | cut -z -f2-"
        else:
            chain = base + " -print0" + grep
            if sort is not None:
                chain += " | sort -z" + (" -r" if sort[1] == "desc" else "")
            if take:
                chain += f" | head -z -n {take}"
        if pick:
            chain += f" | sed -zn {q(str(pick) + 'p')}"

    def many(cmd, dd=" --"):
        return f"{base} -exec {cmd} {{}} +" if plain else f"{chain} | xargs -0 -r {S}{cmd}{dd}"

    def one(cmd):
        return f"{base} -exec {cmd} {{}} \\;" if plain else f"{chain} | xargs -0 -r -n 1 {S}{cmd}"

    def nul():
        return base + " -print0" if plain else chain

    def newline():
        return base if plain else chain + " | tr '\\0' '\\n'"

    if sink == "fshow":
        if plain:
            return base
        if keyed and sort[0] == "size":
            return chain + " | xargs -0 -r du -sh --"
        return newline()
    if sink == "fcount":
        return base + " | wc -l" if plain else chain + " | tr -cd '\\0' | wc -c"
    if sink == "total":
        return nul() + " | du -ch --files0-from=- | tail -n 1"
    if sink == "sizes":
        if plain:
            if dirs and not tests and not explicit and depth is None:
                return f"{S}du -h --max-depth=1 {roots[0]} | sort -rh"
            return base + " -exec du -sh {} + | sort -rh"
        return chain + " | xargs -0 -r du -sh --"
    if sink == "details":
        return many("ls -lhd")
    if sink == "delete":
        rm = "rm" + (" -r" if dirs else "") + ask + v
        if plain:
            return base + (" -delete" if rm == "rm" else f" -exec {rm} {{}} +")
        return f"{chain} | xargs -0 -r {'-o ' if ask else ''}{S}{rm} --"
    if sink == "trash":
        return many("gio trash" if env.has("gio") else "trash-put")
    if sink in ("move", "copy"):
        tool = ("mv" if sink == "move" else ("cp -r" if dirs else "cp")) + ask + v
        return many(f"{tool} -t {_dir(sa['to'], env)}")
    if sink == "compress":
        _fix_fmt(sa)
        if sa["fmt"] == "zip":
            return nul() + f" | xargs -0 -r zip -q{'r' if dirs else ''} {_archive('zip', sa.get('to'), env)}"
        return nul() + f" | tar --null -czf {_archive('tgz', sa.get('to'), env)} -T -"
    if sink == "fgrep":
        text = q(sa["text"])
        if plain and not dirs and depth is None and not explicit and exts_only is not None and (exts_only or not tests):
            inc = "".join(f" --include={q('*.' + e)}" for e in exts_only)
            return f"{S}grep -rni{inc} -- {text} {roots[0]}"
        return many(f"grep -Hni -- {text}", dd="")
    if sink == "fopen":
        return one("xdg-open")
    if sink == "fedit":
        return f"{base} -exec ${{EDITOR:-nano}} {{}} +" if plain else f"{chain} | xargs -0 -r -o ${{EDITOR:-nano}}"
    if sink == "chmod":
        return many(f"chmod {V.mode(sa['mode'])}")
    if sink == "chown":
        return many(f"chown {V.owner(sa['owner'])}")
    if rename:
        pre, loop = _rename_loop(sink, sa, S, v)
        return f"{pre}{nul()} | {loop}"
    if sink == "replace":
        return many(f"sed -i 's/{_sed_pat(sa['old'])}/{_sed_rep(sa['new'])}/g'")
    if sink == "fhash":
        return many("sha256sum")
    if sink == "by_ext":
        return newline() + " | sed -n 's/.*[^/]\\.\\([^./]*\\)$/\\1/p' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -rn"
    if sink == "dupes":
        return nul() + " | xargs -0 -r sha256sum | sort | uniq -w64 --all-repeated=separate"
    if sink == "upload":
        dest = q(sa["to"].strip())
        if plain:
            return f"{base} -exec rsync -avh --progress {{}} {dest} \\;"
        return f"{chain} | xargs -0 -r -I{{}} rsync -avh --progress {{}} {dest}"
    if sink == "fnames":
        return _line_ops(newline(), None, rest, env, sudo)
    if sink == "fcat":
        return _line_ops(f"{base} -exec cat {{}} +" if plain else f"{chain} | xargs -0 -r {S}cat --", None, rest, env, sudo)
    raise ValueError(sink)


# --------------------------------------------------------------------------- lines
_SEP = {"comma": ",", "tab": None, "colon": ":", "semi": ";", "bar": "|"}
_PKG_LIST = {"pacman": "pacman -Qq", "apt": "apt list --installed 2>/dev/null", "dnf": "dnf list installed",
             "zypper": "rpm -qa", "brew": "brew list"}


def _line_source(head, a, env):
    if head == "history":
        return "history"
    if head == "env":
        return "printenv | sort"
    if head == "pkgs":
        return _PKG_LIST[env.pm]
    if head == "services":
        return "systemctl list-units --type=service --state=running --no-pager --no-legend"
    if head == "journal":
        return "journalctl -b --no-pager" + (f" -u {q(a['unit'].strip())}" if a.get("unit") else "")
    if head == "dmesg":
        return "sudo dmesg"
    if head == "clipboard":
        return env.clipboard(True)
    if head == "last":
        if not env.last:
            raise Unsupported()
        return env.last
    raise ValueError(head)


def _lines(pipe, env, sudo):
    head, a0 = pipe[0]
    stages = pipe[1:]
    if head == "lines":
        return _line_ops(None, _file(a0["of"], env), stages, env, sudo)
    src = _line_source(head, a0, env)
    if head == "journal" and stages and stages[0][0] == "llast":
        src += f" -n {V.count(stages[0][1].get('n'), 10)}"
        stages = stages[1:]
    return _line_ops(src, None, stages, env, sudo)


def _line_ops(src, f, stages, env, sudo):
    """src: a command that prints the lines, or None when they come from the file f (already quoted)."""
    S = "sudo " if sudo else ""
    cmds, redirect, inplace = [], "", False       # cmds: (text, can it take the file as an argument)
    for name, a in stages:
        if name == "match":
            cmds.append((f"grep -i -- {q(a['text'])}", True))
        elif name == "nomatch":
            cmds.append((f"grep -vi -- {q(a['text'])}", True))
        elif name == "uniq":
            if cmds and cmds[-1][0].startswith("sort") and "uniq" not in cmds[-1][0]:
                cmds[-1] = (cmds[-1][0] + " -u", cmds[-1][1])
            else:
                cmds.append(("awk '!seen[$0]++'", True))
        elif name == "lsort":
            flags = ("n" if a["key"] == "num" else "") + ("r" if a["order"] == "desc" else "")
            cmds.append(("sort" + (f" -{flags}" if flags else ""), True))
        elif name == "ltake":
            cmds.append((f"head -n {V.count(a.get('n'), 10)}", True))
        elif name == "llast":
            cmds.append((f"tail -n {V.count(a.get('n'), 10)}", True))
        elif name == "lrange":
            nums = [int(x) for x in re.findall(r"\d+", a["a"] + " " + a["b"])]
            lo, hi = (min(nums[:2]), max(nums[:2])) if a["a"] == a["b"] and len(nums) >= 4 else sorted((V.count(a["a"]), V.count(a["b"])))
            cmds.append((f"sed -n '{lo},{hi}p'", True))
        elif name == "lline":
            cmds.append((f"sed -n '{V.count(a['n'])}p'", True))
        elif name == "ldel":
            cmds.append((f"sed '{_line_no(a['n'])}d'", True))
        elif name == "col":
            n, sep = V.count(a["n"]), _SEP.get(a["sep"], " ")
            if a["sep"] == "space":
                cmds.append((f"awk '{{print ${n}}}'", True))
            else:
                cmds.append((f"cut {'-d' + q(sep) + ' ' if sep else ''}-f{n}", True))
        elif name == "lreplace":
            cmds.append((f"sed 's/{_sed_pat(a['old'])}/{_sed_rep(a['new'])}/g'", True))
        elif name == "lfreq":
            cmds += [("sort", True), ("uniq -c", False), ("sort -rn", False)]
        elif name == "lnum":
            cmds.append(("cat -n", True))
        elif name == "lrev":
            cmds.append(("tac", True))
        elif name == "lshuf":
            cmds.append(("shuf", True))
        elif name == "lupper":
            cmds.append(("tr '[:lower:]' '[:upper:]'", False))
        elif name == "llower":
            cmds.append(("tr '[:upper:]' '[:lower:]'", False))
        elif name == "lsum":
            cmds.append(("awk '{s += $1} END {print s}'", True))
        elif name == "lblank":
            cmds.append(("grep -v '^[[:space:]]*$'", True))
        elif name == "lcount":
            cmds.append(("wc -l", False))
        elif name == "wcount":
            cmds.append(("wc -w", False))
        elif name == "ccount":
            cmds.append(("wc -m", False))
        elif name == "clip":
            cmds.append((env.clipboard(False), False))
        elif name == "save":
            redirect = f" | sudo tee {_new(a['to'])} > /dev/null" if sudo else f" > {_new(a['to'])}"
        elif name == "lappend":
            redirect = f" | sudo tee -a {_new(a['to'])} > /dev/null" if sudo else f" >> {_new(a['to'])}"
        elif name == "inplace":
            inplace = True
    ops = [n for n, _ in stages if n not in ("inplace", "lshow")]
    if inplace:
        if f is None:
            raise Unsupported()
        a = dict(stages[0][1]) if stages else {}
        if ops == ["nomatch"]:
            return f"{S}sed -i '/{_sed_pat(a['text'])}/Id' {f}"
        if ops == ["match"]:
            return f"{S}sed -i '/{_sed_pat(a['text'])}/I!d' {f}"
        if ops == ["lreplace"]:
            return f"{S}sed -i 's/{_sed_pat(a['old'])}/{_sed_rep(a['new'])}/g' {f}"
        if ops == ["ldel"]:
            return f"{S}sed -i '{_line_no(a['n'])}d' {f}"
        if ops == ["lblank"]:
            return f"{S}sed -i '/^[[:space:]]*$/d' {f}"
        if ops in (["lsort"], ["lsort", "uniq"]):
            return f"{S}{cmds[0][0]} -o {f} {f}"
        body = _join(None, f, cmds, S)
        return f'tmp=$(mktemp) && {body} > "$tmp" && {S}cp "$tmp" {f} && rm "$tmp"'
    return _join(src, f, cmds, S) + redirect


def _line_no(text):
    return "$" if text.strip().lower() in ("last", "final", "the last", "bottom") else str(V.count(text))


def _join(src, f, cmds, S):
    if src is not None:
        return " | ".join([S + src if S and not src.startswith("sudo ") else src] + [c for c, _ in cmds])
    if not cmds:
        return f"{S}cat {f}"
    first, takes = cmds[0]
    if takes:
        head = f"{S}{first} {f}"
    elif S:
        head = f"sudo cat {f} | {first}"
    else:
        head = f"{first} < {f}"
    return " | ".join([head] + [c for c, _ in cmds[1:]])


# --------------------------------------------------------------------------- processes
def _procs(pipe, env, sudo):
    S = "sudo " if sudo else ""
    head, a0 = pipe[0]
    pname, user, sort, take, sink, force = None, None, None, None, "pshow", "soft"
    for name, a in pipe[1:]:
        if name == "pname":
            pname = a["name"].strip()
        elif name == "puser":
            user = V.owner(a["user"])
        elif name == "psort":
            sort = (a["key"], a["order"])
        elif name == "ptake":
            take = V.count(a.get("n"))
        elif name == "kill":
            sink, force = "kill", a["force"]
        else:
            sink = name
    sig = " -9" if force == "hard" else ""
    if head == "port":
        n = V.port(a0["n"])
        if sink == "kill":
            return f"{S}kill{sig} $(lsof -ti :{n})" if env.has("lsof") else f"{S}fuser -k {n}/tcp"
        if sink == "pcount":
            return f"{S}lsof -ti :{n} | wc -l"
        return f"{S}lsof -i :{n}" if env.has("lsof") else f"{S}ss -tulpn | grep -w {q(':' + n)}"
    who = f" -u {user}" if user else ""
    order = f" --sort={'' if sort and sort[1] == 'asc' else '-'}%{sort[0]}" if sort else ""
    simple = not sort and not take
    if sink == "kill":
        if pname and simple:
            if not user and force == "soft" and env.is_service(pname):
                env.note(f"{pname} is a running service")
                return f"sudo systemctl stop {q(pname)}"
            return f"{S}pkill{sig}{who} -i -- {q(pname)}"
        if user and not pname and simple:
            return f"{S}pkill{sig}{who}"
        sel = f"-C {q(pname)}" if pname else (f"-u {user}" if user else "-e")
        return f"ps {sel} -o pid={order or ' --sort=-%mem'} | head -n {take or 1} | xargs -r {S}kill{sig}"
    if sink == "pcount":
        if pname or user:
            return f"pgrep -c{who}" + (f" -i -- {q(pname)}" if pname else "")
        return "ps -e --no-headers | wc -l"
    if pname and simple:
        return f"pgrep -a{who} -i -- {q(pname)}"
    sel = f"-C {q(pname)}" if pname else (f"-u {user}" if user else "-e")
    cmd = f"ps {sel} -o pid,user,%cpu,%mem,comm{order}"
    return cmd + (f" | head -n {take + 1}" if take else "")


# --------------------------------------------------------------------------- stand-alone commands
_PKG = {
    "pacman": {"install": "sudo pacman -S {}", "remove": "sudo pacman -Rs {}", "search": "pacman -Ss {}", "update": "sudo pacman -Syu",
               "info": "pacman -Qi {0} 2>/dev/null || pacman -Si {0}", "has": "pacman -Q {}", "owner": "pacman -Qo {}",
               "files": "pacman -Ql {}", "clean": "sudo pacman -Sc", "orphans": "pacman -Qdtq",
               "autoremove": "pacman -Qdtq | sudo pacman -Rns -", "upgradable": "checkupdates"},
    "apt": {"install": "sudo apt install {}", "remove": "sudo apt remove {}", "search": "apt search {}",
            "update": "sudo apt update && sudo apt upgrade", "info": "apt show {}", "has": "dpkg -s {}", "owner": "dpkg -S {}",
            "files": "dpkg -L {}", "clean": "sudo apt clean", "orphans": "apt autoremove --dry-run",
            "autoremove": "sudo apt autoremove", "upgradable": "apt list --upgradable"},
    "dnf": {"install": "sudo dnf install {}", "remove": "sudo dnf remove {}", "search": "dnf search {}", "update": "sudo dnf upgrade",
            "info": "dnf info {}", "has": "rpm -q {}", "owner": "rpm -qf {}", "files": "rpm -ql {}", "clean": "sudo dnf clean all",
            "orphans": "dnf repoquery --unneeded", "autoremove": "sudo dnf autoremove", "upgradable": "dnf check-update"},
    "zypper": {"install": "sudo zypper install {}", "remove": "sudo zypper remove {}", "search": "zypper search {}",
               "update": "sudo zypper update", "info": "zypper info {}", "has": "rpm -q {}", "owner": "rpm -qf {}",
               "files": "rpm -ql {}", "clean": "sudo zypper clean", "orphans": "zypper packages --unneeded",
               "autoremove": "sudo zypper remove --clean-deps", "upgradable": "zypper list-updates"},
    "brew": {"install": "brew install {}", "remove": "brew uninstall {}", "search": "brew search {}", "update": "brew upgrade",
             "info": "brew info {}", "has": "brew list {}", "owner": None, "files": "brew list {}", "clean": "brew cleanup",
             "orphans": "brew autoremove --dry-run", "autoremove": "brew autoremove", "upgradable": "brew outdated"},
}
_PKG_KEY = {"pkg_install": "install", "pkg_remove": "remove", "pkg_search": "search", "sys_update": "update", "pkg_info": "info",
            "pkg_has": "has", "pkg_owner": "owner", "pkg_files": "files", "pkg_clean": "clean", "pkg_orphans": "orphans",
            "pkg_autoremove": "autoremove", "pkg_upgradable": "upgradable"}
_PROJECT = {
    "cargo": ("cargo test", "cargo build", "cargo run"), "npm": ("npm test", "npm run build", "npm start"),
    "go": ("go test ./...", "go build ./...", "go run ."), "python": ("pytest", "python3 -m build", "python3 -m pytest -q"),
    "make": ("make test", "make", "make run"), "cmake": ("ctest --test-dir build", "cmake -B build && cmake --build build", "cmake --build build"),
    "gradle": ("./gradlew test", "./gradlew build", "./gradlew run"), "maven": ("mvn test", "mvn package", "mvn exec:java"),
    "meson": ("meson test -C build", "meson setup build && meson compile -C build", "meson compile -C build"),
    None: ("make test", "make", "make run"),
}
_RUNNERS = {".py": "python3", ".sh": "bash", ".bash": "bash", ".js": "node", ".mjs": "node", ".ts": "npx tsx", ".rb": "ruby",
            ".pl": "perl", ".php": "php", ".lua": "lua", ".go": "go run", ".r": "Rscript"}


def _value(kind, text, env):
    """One argument, quoted for the shell according to what kind of thing it is."""
    t = text.strip()
    if kind == "path":
        return _dir(t, env)
    if kind == "file":
        return _file(t, env)
    if kind == "words":
        return _words(t)
    if kind == "count":
        return str(V.count(t))
    if kind == "port":
        return V.port(t)
    if kind == "pct":
        return str(V.percent(t))
    if kind == "envname":
        return re.sub(r"[^A-Za-z0-9_]", "", t) or "VAR"
    return q(t)


def _fill(c, a, env):
    kinds = {n: k for n, k, _ in c["slots"]}

    def put(m):
        return _value(kinds[m.group(1)], a[m.group(1)], env)

    def optional(m):
        inner = m.group(1)
        if all(a.get(n) is not None for n in re.findall(r"\{(\w+)\}", inner)):
            return re.sub(r"\{(\w+)\}", put, inner)
        return ""

    return re.sub(r"\{(\w+)\}", put, re.sub(r"\[([^\[\]]*)\]", optional, c["bash"]))


def _extract(a, env):
    f, into = a["archive"].strip(), a.get("into")
    low, ff = f.lower(), _file(a["archive"], env)
    if low.endswith(".zip"):
        return f"unzip {ff}" + (f" -d {_dir(into, env)}" if into else "")
    if low.endswith(".7z"):
        return f"7z x {ff}" + (f" -o{_dir(into, env)}" if into else "")
    if low.endswith(".rar"):
        return f"unrar x {ff}" + (f" {_dir(into, env)}/" if into else "")
    if low.endswith(".gz") and not low.endswith(".tar.gz"):
        return f"gunzip -k {ff}"
    return f"tar -xf {ff}" + (f" -C {_dir(into, env)}" if into else "")


def _archive_list(a, env):
    low, ff = a["archive"].strip().lower(), _file(a["archive"], env)
    if low.endswith(".zip"):
        return f"unzip -l {ff}"
    if low.endswith(".7z"):
        return f"7z l {ff}"
    if low.endswith(".rar"):
        return f"unrar l {ff}"
    return f"tar -tf {ff}"


def _download(a, env):
    url, to = q(a["url"].strip()), a.get("to")
    if not to:
        return f"curl -LO {url}" if env.has("curl") else f"wget {url}"
    dest = V.path(to, env.isdir)
    is_dir = env.isdir(dest) or dest.endswith("/") or "." not in dest.rsplit("/", 1)[-1].lstrip(".")
    if not env.has("curl"):
        return f"wget {'-P' if is_dir else '-O'} {V.quote_path(dest)} {url}"
    return f"curl -L --output-dir {V.quote_path(dest)} -O {url}" if is_dir else f"curl -L -o {V.quote_path(dest)} {url}"


def _project(i):
    def build(a, env):
        return _PROJECT[env.project()][i]
    return build


def _run_file(a, env):
    name = env.file(a["file"].strip())
    for ext, tool in _RUNNERS.items():
        if name.lower().endswith(ext):
            return f"{tool} {V.quote_path(name)}"
    return V.quote_path(name if "/" in name or name.startswith("~") else "./" + name)


def _open(a, env):
    t = a["target"].strip()
    if env.live and not env.exists(t) and " " not in t and env.has(t):
        return f"setsid -f {q(t)} >/dev/null 2>&1"
    return f"xdg-open {_file(t, env)}"


def _view(a, env):
    t = env.file(a["file"].strip())
    return f"ls -lah {V.quote_path(t)}" if env.isdir(t) else f"cat {V.quote_path(t)}"


def _ls(a, env):
    if not a.get("path"):
        return "ls -lah"
    p = env.folder(V.path(a["path"], env.isdir))
    return f"cat {V.quote_path(p)}" if env.isfile(p) else f"ls -lah {V.quote_path(p)}"


def _tree(a, env):
    p = _dir(a["path"], env) if a.get("path") else ""
    d = V.count(a["depth"]) if a.get("depth") else None
    if env.has("tree"):
        return " ".join(x for x in ("tree", f"-L {d}" if d else "", p) if x)
    return " ".join(x for x in ("find", p or ".", f"-maxdepth {d}" if d else "", "-print | sort") if x)


def _assign(name, value, ident=True):
    """'FOO' + 'bar', or one word 'FOO=bar' given for both -> (name, quoted value)"""
    name, value = name.strip(), value.strip()
    if "=" in name:
        head, _, rest = name.partition("=")
        value = rest if value == name or not value else value
        name = head
    if "=" in value and value.split("=")[0] == name:
        value = value.partition("=")[2]
    name = (re.sub(r"[^A-Za-z0-9_]", "", name) or "VAR") if ident else q(name)
    return name, q(value)


def _last(env):
    if not env.last:
        raise Unsupported()
    return env.last


def _venv_on(a, env):
    if a.get("path"):
        p = V.path(a["path"], env.isdir)
        p = p[:-len("/bin/activate")] if p.endswith("/bin/activate") else p
    else:
        p = next((d for d in (".venv", "venv", "env") if env.isdir(d)), ".venv")
    return f"source {V.quote_path(p)}/bin/activate"


def _pkg(name):
    def build(a, env):
        tpl = _PKG[env.pm][_PKG_KEY[name]]
        if tpl is None:
            raise Unsupported()
        arg = ""
        if "names" in a:
            arg = _words(a["names"])
        elif "term" in a:
            arg = q(a["term"].strip())
        elif "name" in a:
            arg = q(a["name"].strip())
        elif "file" in a:
            t = a["file"].strip()
            arg = _file(t, env) if "/" in t or env.exists(t) else q(t)
        return tpl.format(arg)
    return build


FN = {
    "battery": lambda a, env: 'for b in /sys/class/power_supply/BAT*; do echo "$(cat "$b/capacity")% $(cat "$b/status")"; done',
    "dns": lambda a, env: (f"dig +short {q(a['host'].strip())}" if env.has("dig") else f"getent hosts {q(a['host'].strip())}"),
    "download": _download,
    "serve": lambda a, env: f"python3 -m http.server {V.port(a['port']) if a.get('port') else 8000}",
    "poweroff": lambda a, env: (f"shutdown -h +{V.minutes(a['in'])}" if a.get("in") else "systemctl poweroff"),
    "venv_make": lambda a, env: f"python3 -m venv {_dir(a['path'], env) if a.get('path') else '.venv'}",
    "venv_on": _venv_on,
    "proj_test": _project(0), "proj_build": _project(1), "proj_run": _project(2),
    "run_file": _run_file,
    "dk_build": lambda a, env: "docker build " + (f"-t {q(a['tag'].strip())} " if a.get("tag") else "") + ".",
    "git_log": lambda a, env: f"git log --oneline -n {V.count(a.get('n'), 10)}",
    "git_unstage": lambda a, env: f"git restore --staged {_file(a['file'], env) if a.get('file') else '.'}",
    "ls": _ls, "tree": _tree, "view": _view,
    "edit": lambda a, env: f"${{EDITOR:-nano}} {_file(a['file'], env)}",
    "open": _open,
    "backup": lambda a, env: f"cp -a {_file(a['file'], env)} {V.quote_path(env.file(a['file'].strip()) + '.bak')}",
    "extract": _extract, "archive_list": _archive_list,
    "var_set": lambda a, env: "export {}={}".format(*_assign(a["name"], a["value"])),
    "path_add": lambda a, env: f'export PATH={_dir(a["dir"], env)}:"$PATH"',
    "alias_make": lambda a, env: "alias {}={}".format(*_assign(a["name"], a["cmd"], ident=False)),
    "calc": lambda a, env: "awk 'BEGIN {print " + (V.expr(a["expr"]) or "0") + "}'",
    "timer": lambda a, env: f"sleep {V.seconds(a['in'])} && " + ("notify-send 'Time is up'" if env.has("notify-send") else "echo 'Time is up'"),
    "again": lambda a, env: _last(env),
    "watch_last": lambda a, env: f"watch -n {V.count(a.get('n'), 2)} {q(_last(env))}",
    "time_last": lambda a, env: f"time {_last(env)}",
}
FN.update({name: _pkg(name) for name in _PKG_KEY})


# arguments that name something the command creates: never matched against existing names
_CREATES = {"touch": ("file",), "write": ("file",), "append": ("file",), "link": ("name",), "rename": ("new",), "mkdir": ("path",),
            "venv_make": ("path",), "git_clone": ("into",), "extract": ("into",), "download": ("to",), "edit": ("file",),
            "git_new_branch": (), "alias_make": ()}


def _single(name, a, env):
    c = commands.BY_NAME[name]
    for slot in _CREATES.get(name, ()):
        if a.get(slot):
            env.exempt.add(a[slot].strip())
            env.exempt.add(V.path(a[slot]))
    if c["bash"] is None:
        return FN[name](a, env)
    return _fill(c, a, env)


# --------------------------------------------------------------------------- whole programs
def compile_pipeline(pipe, env=None):
    env = as_env(env)
    env.level = 2 if risk([pipe]) == 0 else 1          # a command that changes things takes names as typed
    sudo = pipe[0][0] == "sudo"
    if sudo:
        pipe = pipe[1:]
    head = pipe[0][0]
    if head == "unknown":
        raise Unsupported()
    if head == "pref":
        return ""
    if head in ("files", "paths"):
        return _files(pipe, env, sudo)
    if head in LINE_SOURCES:
        return _lines(pipe, env, sudo)
    if head in ("procs", "port"):
        return _procs(pipe, env, sudo)
    cmd = _single(head, pipe[0][1], env)
    return "sudo " + cmd if sudo and not cmd.startswith("sudo ") else cmd


def compile_program(program, env=None):
    env = as_env(env)
    return " && ".join(c for c in (compile_pipeline(p, env) for p in program) if c)


def risk(program):
    return max((RISK.get(name, 0) for pipe in program for name, _ in pipe), default=0)


_WORDS = {
    "flat": "top level only", "empty": "empty", "uniq": "no duplicates", "fshow": "list", "lshow": "show",
    "pshow": "show", "fcount": "count", "lcount": "count lines", "pcount": "count", "total": "total size",
    "sizes": "size of each", "delete": "DELETE", "kill": "KILL", "sudo": "as root", "not": "not", "procs": "processes",
    "hidden": "hidden", "isexec": "executable", "details": "list with details", "trash": "move to trash",
    "fopen": "open", "fedit": "edit", "rn_lower": "RENAME to lowercase", "rn_upper": "RENAME to uppercase",
    "rn_spaces": "RENAME spaces to underscores", "fhash": "checksum", "by_ext": "count by type", "dupes": "find duplicates",
    "fnames": "names", "fcat": "contents", "wcount": "count words", "ccount": "count characters",
    "inplace": "WRITE BACK to the file", "clip": "copy to clipboard", "lfreq": "most common first", "lnum": "number lines",
    "lrev": "reverse", "ldel": "drop line", "lshuf": "shuffle", "lupper": "upper case", "llower": "lower case", "lsum": "add up",
    "lblank": "drop blank lines", "history": "command history", "env": "environment", "pkgs": "installed packages",
    "services": "running services", "dmesg": "kernel log", "clipboard": "clipboard", "last": "output of the last command",
    "again": "run the last command again", "unknown": "?",
}


def describe(program):
    """One-line plain-English reading of the program, for the user to check before running."""
    parts = []
    for pipe in program:
        bits = []
        for name, a in pipe:
            if name == "files":
                bits.append(("folders" if a["what"] == "dirs" else "files") + (f" in {a['in']}" if a.get("in") else " here"))
            elif name == "paths":
                bits.append(a["names"])
            elif name == "lines":
                bits.append(f"lines of {a['of']}")
            elif name == "pref":
                bits.append(f"preference: {a['key']} {a['state']}")
            elif name in _WORDS:
                bits.append(_WORDS[name] + (" (force)" if a.get("force") == "hard" else ""))
            elif name in ("fsort", "lsort", "psort"):
                bits.append(f"sort by {a['key']} {'high to low' if a['order'] == 'desc' else 'low to high'}")
            elif name in ("ftake", "ltake", "ptake"):
                bits.append(f"first {V.count(a.get('n'), 10 if name == 'ltake' else 1)}")
            elif name == "llast":
                bits.append(f"last {V.count(a.get('n'), 10)}")
            elif name in ("larger", "smaller", "newer", "older") and not any(a.values()):
                bits.append({"larger": "large", "smaller": "small", "newer": "recent", "older": "old"}[name])
            else:
                vals = " ".join(str(v) for k, v in a.items() if v and k != "sep")
                label = name.replace("_", " ").upper() if RISK.get(name, 0) == 2 else name.replace("_", " ")
                bits.append(label + (f" {vals}" if vals else ""))
        text = " > ".join(bits).replace("not > ", "not ").replace("as root > ", "as root: ")
        parts.append(text)
    return "  then  ".join(parts)
