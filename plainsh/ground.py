"""Facts about the machine the command will run on.

The parser only understands language. Which files exist, which tools are installed and what kind of
project the current folder holds are looked up here, by ordinary code, when the command is built.
With live=False nothing is looked at (used for tests and training samples, so results are repeatable).
"""
import difflib
import os
import shutil
import subprocess

DEFAULT_TOOLS = {"curl", "dig", "lsof", "tree", "wl-copy", "wl-paste", "gio", "notify-send", "xdg-open", "rsync"}


def package_manager():
    for pm in ("pacman", "apt", "dnf", "zypper", "brew"):
        if shutil.which(pm):
            return pm
    return "pacman"


class Env:
    def __init__(self, cwd=None, live=True, pm=None, prefs=None, last=None, final=True):
        self.live = live
        self.cwd = cwd or (os.getcwd() if live else "/")
        self.pm = pm or (package_manager() if live else "pacman")
        self.prefs = dict(prefs or {})     # trash / confirm / verbose -> True
        self.last = last                   # the previous command line, for "again" and "that output"
        self.final = final                 # False while the user is still typing: skip anything slow
        self.notes = []                    # things worth telling the user about how the command was built
        # How far a name that does not exist may be bent towards one that does:
        #   2  read-only commands: ignore case, add a missing extension, fix a small typo
        #   1  commands that change things: ignore case only
        #   0  never
        self.level = 2
        self.exempt = set()                # names of things the command is about to create: never bent

    def fresh(self):
        e = Env(self.cwd, self.live, self.pm, self.prefs, self.last, self.final)
        return e

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    # ---- tools
    def has(self, tool):
        return shutil.which(tool) is not None if self.live else tool in DEFAULT_TOOLS

    def clipboard(self, paste):
        """The command that reads (paste) or fills the clipboard on this machine."""
        if self.has("wl-copy") or (not self.has("xclip") and os.environ.get("WAYLAND_DISPLAY")):
            if not self.has("wl-copy"):
                self.note("needs the wl-clipboard package")
            return "wl-paste" if paste else "wl-copy"
        return "xclip -o -selection clipboard" if paste else "xclip -selection clipboard"

    # ---- paths
    def resolve(self, p):
        return os.path.normpath(os.path.join(self.cwd, os.path.expanduser(p)))

    def exists(self, p):
        return self.live and os.path.lexists(self.resolve(p))

    def isdir(self, p):
        return self.live and os.path.isdir(self.resolve(p))

    def isfile(self, p):
        return self.live and os.path.isfile(self.resolve(p))

    def _closest(self, name, want_dir=None):
        """A name the user typed that does not exist -> the one entry it clearly means, or None."""
        d, base = os.path.split(name)
        folder = self.resolve(d) if d else self.cwd
        try:
            entries = os.listdir(folder)
        except OSError:
            return None
        if want_dir is not None:
            entries = [e for e in entries if os.path.isdir(os.path.join(folder, e)) == want_dir]
        low = base.lower()
        rules = [lambda e: e.lower() == low]
        if self.level >= 2:
            rules += [lambda e: os.path.splitext(e)[0].lower() == low, lambda e: len(low) >= 3 and e.lower().startswith(low)]
        for rule in rules:
            hits = [e for e in entries if rule(e)]
            if len(hits) == 1:
                return os.path.join(d, hits[0])
            if hits:
                return None            # several candidates: do not guess
        if self.level < 2:
            return None
        close = difflib.get_close_matches(base, entries, n=2, cutoff=0.8)
        if len(close) == 1:
            return os.path.join(d, close[0])
        return None

    def file(self, name):
        """The file the user means. Unchanged unless it does not exist and exactly one real name is clearly meant."""
        if not self.live or not name or self.level == 0 or name in self.exempt or any(c in name for c in "*?[") or self.exists(name):
            return name
        hit = self._closest(name)
        if hit is None:
            words = [w for w in name.split() if w.lower() not in ("the", "my", "a", "file", "folder")]
            if len(words) == 1 and words[0] != name:
                if self.exists(words[0]):
                    return words[0]
                hit = self._closest(words[0])
        if hit:
            self.note(f"{name} taken to mean {hit}")
            return hit
        return name

    def folder(self, path):
        if (not self.live or not path or self.level == 0 or path in self.exempt or path in (".", "..", "~", "/")
                or path.startswith(("~", "/")) or self.exists(path)):
            return path
        hit = self._closest(path, want_dir=True)
        if hit:
            self.note(f"{path} taken to mean {hit}")
            return hit
        return path

    # ---- services
    def is_service(self, name):
        if not (self.live and self.final) or not name or not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            return False
        try:
            return subprocess.run(["systemctl", "is-active", "--quiet", name], timeout=1).returncode == 0
        except Exception:
            return False

    # ---- projects
    def project(self):
        """What kind of project the current folder is, from the files in it."""
        if not self.live:
            return None
        for marker, kind in (("Cargo.toml", "cargo"), ("package.json", "npm"), ("go.mod", "go"), ("pyproject.toml", "python"),
                             ("setup.py", "python"), ("pytest.ini", "python"), ("Makefile", "make"), ("makefile", "make"),
                             ("CMakeLists.txt", "cmake"), ("build.gradle", "gradle"), ("build.gradle.kts", "gradle"),
                             ("pom.xml", "maven"), ("meson.build", "meson")):
            if os.path.exists(os.path.join(self.cwd, marker)):
                return kind
        return None


def as_env(env):
    """Accepts an Env, a package-manager name (tests) or None (look at the real machine)."""
    if isinstance(env, Env):
        return env
    if isinstance(env, str):
        return Env(live=False, pm=env)
    return Env()
