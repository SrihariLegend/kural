"""Turning the user's copied words into exact values: sizes, ages, counts, folders, file types.

All of this is ordinary deterministic code. The model's only job is to point at the
right words; what "2 gigs" or "last week" means is decided here.
"""
import os
import re
import shlex

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "couple": 2, "three": 3, "few": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "hundred": 100,
    "thousand": 1000, "half": 0.5, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10,
}


def number(text):
    """First number in the text, as a float, or None."""
    for w in re.findall(r"\d+(?:\.\d+)?|[a-z]+", text.lower()):
        if w[0].isdigit():
            return float(w)
        if w in NUMBER_WORDS:
            return NUMBER_WORDS[w]
    return None


def count(text, default=1):
    n = number(text or "")
    return max(1, int(n)) if n else default


SIZE_UNITS = [
    (r"t(?:b|erabytes?|era)?", 1024 ** 4), (r"g(?:b|igs?|igabytes?|i|ib)?", 1024 ** 3),
    (r"m(?:b|egs?|egabytes?|i|ib)?", 1024 ** 2), (r"k(?:b|ilobytes?|i|ib|ilo)?", 1024), (r"b(?:ytes?)?", 1),
]


def size_bytes(text):
    """'100MB', '2 gigs', '1.5 G', 'half a gig', '500k' -> bytes (int). Bare numbers are megabytes."""
    t = text.lower().replace(",", "")
    n = number(t)
    if n is None:
        n = 1
    m = re.search(r"(?:\d|\b)(tb|terabytes?|gb|gigs?|gigabytes?|gib|mb|megs?|megabytes?|mib|kb|kilobytes?|kib|bytes?|[tgmkb])\b", t)
    mult = 1024 ** 2
    if m:
        u = m.group(1)
        for pat, val in SIZE_UNITS:
            if re.fullmatch(pat, u):
                mult = val
                break
    return int(n * mult)


def find_size(text):
    """Argument for find -size, without the +/- sign: bytes rounded to the largest whole unit."""
    b = size_bytes(text)
    for unit, val in (("G", 1024 ** 3), ("M", 1024 ** 2), ("k", 1024)):
        if b >= val and b % val == 0:
            return f"{b // val}{unit}"
    for unit, val in (("M", 1024 ** 2), ("k", 1024)):
        if b >= val:
            return f"{round(b / val)}{unit}"
    return f"{b}c"


DUR_UNITS = [
    (r"s|secs?|seconds?", 1 / 60), (r"m|mins?|minutes?", 1), (r"h|hrs?|hours?", 60), (r"d|days?", 1440),
    (r"w|wks?|weeks?", 10080), (r"mo|months?", 43200), (r"y|yrs?|years?", 525600),
]
DUR_WORDS = {"today": 1440, "yesterday": 2880, "tonight": 720, "fortnight": 20160, "overnight": 720}


def minutes(text):
    """'30 days', 'a week', '2h', 'last month', 'today' -> minutes (int)."""
    t = text.lower()
    for w, v in DUR_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            return v
    n = number(re.sub(r"\b(this|last|past|the|previous)\b", "", t))
    unit = None
    for tok in re.findall(r"\d+(?:\.\d+)?\s*([a-z]+)|\b([a-z]+)\b", t):
        u = tok[0] or tok[1]
        for pat, val in DUR_UNITS:
            if re.fullmatch(pat, u):
                unit = val
                break
        if unit:
            break
    if unit is None:
        unit = 1440
    if n is None:
        n = 1
    return max(1, int(round(n * unit)))


def find_age(text, newer):
    """find test for a modification age: newer=True means changed within that long."""
    m = minutes(text)
    sign = "-" if newer else "+"
    if m % 1440 == 0:
        days = m // 1440
        # find -mtime +N means strictly more than N+1 days; shift so "older than 30 days" means what it says
        return f"-mtime {sign}{days if newer else max(0, days - 1)}"
    return f"-mmin {sign}{m}"


FOLDERS = {
    "downloads": "~/Downloads", "download": "~/Downloads", "documents": "~/Documents", "docs": "~/Documents",
    "desktop": "~/Desktop", "pictures": "~/Pictures", "photos": "~/Pictures", "music": "~/Music",
    "videos": "~/Videos", "movies": "~/Videos", "home": "~", "here": ".", "current": ".", "this": ".", "cwd": ".",
    "root": "/", "tmp": "/tmp", "temp": "/tmp", "trash": "~/.local/share/Trash/files", "parent": "..",
    "config": "~/.config", "projects": "~/Projects",
}
_FILLER = {"the", "my", "our", "a", "folder", "directory", "dir", "folders", "directories", "working", "system", "one"}


def path(text, isdir=None):
    """'~/src', 'the downloads folder', 'my desktop', 'here' -> a path string (not yet quoted).
    isdir: a function that says whether a folder of that name really exists where the command will run."""
    words = text.split()
    for w in words:
        if "/" in w or w.startswith(("~", ".")) and len(w) > 1:
            return w
    core = [w for w in words if w.lower() not in _FILLER]
    if len(core) == 1 and core[0].lower() in FOLDERS:
        # a real folder with that exact name in the current directory wins over the well-known one
        if len(words) == 1 and isdir is not None and isdir(core[0]):
            return core[0]
        return FOLDERS[core[0].lower()]
    if not core:
        return "."
    return " ".join(core)


def quote_path(p):
    """Shell-quote a path but keep a leading ~ working."""
    if p == "~":
        return "~"
    if p.startswith("~/"):
        return "~/" + shlex.quote(p[2:]) if p[2:] else "~/"
    return shlex.quote(p)


EXT_GROUPS = {
    "image": ["jpg", "jpeg", "png", "gif", "webp"], "picture": ["jpg", "jpeg", "png", "gif", "webp"],
    "photo": ["jpg", "jpeg", "png", "heic"], "video": ["mp4", "mkv", "mov", "avi", "webm"],
    "movie": ["mp4", "mkv", "mov", "avi", "webm"], "audio": ["mp3", "flac", "wav", "ogg", "m4a"],
    "music": ["mp3", "flac", "wav", "ogg", "m4a"], "song": ["mp3", "flac", "wav", "ogg", "m4a"],
    "archive": ["zip", "tar", "gz", "xz", "7z", "rar", "tgz"], "yaml": ["yaml", "yml"], "jpeg": ["jpg", "jpeg"],
    "jpg": ["jpg", "jpeg"],
}
EXT_NAMES = {
    "python": "py", "javascript": "js", "typescript": "ts", "rust": "rs", "golang": "go", "ruby": "rb",
    "markdown": "md", "text": "txt", "shell": "sh", "bash": "sh", "script": "sh", "c++": "cpp", "header": "h",
    "backup": "bak", "temp": "tmp", "temporary": "tmp", "word": "docx", "excel": "xlsx", "powerpoint": "pptx",
    "kotlin": "kt", "perl": "pl", "config": "conf", "notebook": "ipynb", "jupyter": "ipynb",
}


def extensions(text):
    """'python', 'pdfs', '.log', 'images', '*.txt' -> list of extensions without dots."""
    w = [x for x in text.lower().split() if x not in ("files", "file", "the", "all")]
    w = (w[0] if w else text.lower()).lstrip("*").lstrip(".")
    for cand in (w, w[:-1] if w.endswith("s") else w):
        if cand in EXT_GROUPS:
            return EXT_GROUPS[cand]
        if cand in EXT_NAMES:
            return [EXT_NAMES[cand]]
    if w.endswith("s") and len(w) > 3 and w[:-1].isalnum():
        w = w[:-1]
    return [w]


# ---- what a vague word means when no number was given
BIG, SMALL, OLD_DAYS, NEW_DAYS, SHOW_TOP = "100M", "100k", 90, 7, 20


def percent(text):
    """'50%', '50', 'half', 'full', 'max' -> whole number 0..100"""
    t = text.lower()
    for w, v in (("half", 50), ("full", 100), ("max", 100), ("maximum", 100), ("quarter", 25), ("zero", 0), ("min", 0), ("minimum", 0)):
        if re.search(rf"\b{w}\b", t):
            return v
    n = number(t.replace("%", " "))
    return max(0, min(100, int(n))) if n is not None else 50


def port(text):
    m = re.search(r"\d+", text)
    return m.group(0) if m else "80"


def seconds(text):
    """'10 minutes', '90s', 'half an hour' -> seconds"""
    t = text.lower()
    m = re.fullmatch(r"\s*(\d+)\s*(s|secs?|seconds?)\s*", t)
    if m:
        return int(m.group(1))
    return minutes(t) * 60


def date(text):
    """A date as the user wrote it, tidied for GNU date ('jan 5', 'last friday', '2024-03-01', 'yesterday')."""
    t = re.sub(r"\b(the|on|of)\b", " ", text.lower())
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)
    return " ".join(t.split())


MODES = [
    (r"exec|runnable|\+x", "+x"), (r"not exec|non.?exec|-x", "-x"), (r"read.?only|readonly|unwritable|write.?protect", "a-w"),
    (r"writable|writeable|editable", "u+w"), (r"private|only (me|i)|just me|mine only|owner.?only", "go-rwx"),
    (r"public|everyone|everybody|world.?readable|readable by all|all users", "a+rX"),
]


def mode(text):
    """'755', 'u+x', 'executable', 'read only', 'private' -> chmod argument"""
    t = text.strip().lower()
    m = re.search(r"\b[0-7]{3,4}\b", t)
    if m:
        return m.group(0)
    m = re.search(r"\b[ugoa]*[+=-][rwxXst]+(?:,[ugoa]*[+=-][rwxXst]+)*", t)
    if m:
        return m.group(0)
    for pat, val in sorted(MODES, key=lambda x: -len(x[0])):
        if re.search(pat, t):
            return val
    return "+x"


def owner(text):
    w = [x for x in text.split() if x.lower() not in ("the", "user", "owner", "account")]
    name = w[0] if w else text.strip()
    return '"$USER"' if name.lower() in ("me", "myself", "my", "mine", "i") else shlex.quote(name)


def names(text):
    """'a.txt and b.txt, c.txt' -> ['a.txt', 'b.txt', 'c.txt']"""
    return [w.strip(",") for w in text.split() if w.strip(",") and w.lower() not in ("and", "&", "plus", "also", "the", "file", "files")]


_EXPR_WORDS = [(r"\bto the power of\b", "^"), (r"\bpercent of\b|% of\b", "/100*"), (r"\bdivided by\b|\bover\b", "/"),
               (r"\bmultiplied by\b|\btimes\b|(?<=\d)\s*x\s*(?=\d)", "*"), (r"\bplus\b|\band\b", "+"), (r"\bminus\b", "-"),
               (r"\bsquared\b", "^2"), (r"\bmod\b", "%")]


def expr(text):
    """'12 times 8', '15 percent of 240', '(4+5)*3' -> an expression for bc"""
    t = text.lower().replace(",", "")
    for pat, rep in _EXPR_WORDS:
        t = re.sub(pat, rep, t)
    return "".join(re.findall(r"[0-9.+\-*/^%() ]+", t)).strip()


def extensions_any(text):
    """Like extensions(), but also accepts several: 'pdf or docx', 'jpg, png and gif'."""
    parts = [p for p in re.split(r"\s*(?:,|\bor\b|\band\b|/)\s*", text.strip()) if p.strip()]
    out = []
    for p in parts or [text]:
        for e in extensions(p):
            if e not in out:
                out.append(e)
    return out
