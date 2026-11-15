"""Drive the bash hook in a pseudo-terminal and look at the screen, the way a person would.

  .venv/bin/python tests/term.py            run the checks (needs the `pyte` terminal emulator)

Uses its own runtime folder, so it does not touch the answering process of your real terminals.
"""
import fcntl
import os
import pty
import select
import shutil
import struct
import sys
import tempfile
import termios
import time

import pyte

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Term:
    def __init__(self, rows=24, cols=100, cwd=None):
        self.tmp = tempfile.mkdtemp(prefix="plainsh-term-")
        os.chmod(self.tmp, 0o700)
        rc = os.path.join(self.tmp, "rc")
        with open(rc, "w") as f:
            f.write(f"PS1='[t \\W]\\$ '\nHISTFILE={self.tmp}/hist\nsource {ROOT}/plainsh/shell/plainsh.bash\n")
        os.mkdir(os.path.join(self.tmp, "bin"))
        launcher = os.path.join(self.tmp, "bin", "plainsh")
        with open(launcher, "w") as f:
            f.write(f"#!/bin/sh\nexec env PYTHONPATH={ROOT} python3 -m plainsh.cli \"$@\"\n")
        os.chmod(launcher, 0o755)
        env = dict(os.environ, XDG_RUNTIME_DIR=self.tmp, PATH=self.tmp + "/bin:" + os.environ["PATH"], TERM="xterm-256color",
                   LANG="en_US.UTF-8", PLAINSH_BIN=launcher)
        env.pop("PROMPT_COMMAND", None)
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.ByteStream(self.screen)
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(cwd or self.tmp)
            os.execvpe("bash", ["bash", "--noprofile", "--rcfile", rc, "-i"], env)
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.pump(0.5)

    def pump(self, quiet=0.15, limit=5.0):
        end = time.time() + limit
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], quiet)
            if not r:
                return
            try:
                data = os.read(self.fd, 65536)
            except OSError:
                return
            self.stream.feed(data)

    def send(self, text, wait=0.2):
        os.write(self.fd, text.encode())
        self.pump(wait)

    def type(self, text, delay=0.01, wait=0.6):
        for ch in text:
            os.write(self.fd, ch.encode())
            time.sleep(delay)
        self.pump(wait)

    def lines(self):
        return [ln.rstrip() for ln in self.screen.display if ln.strip()]

    def colour(self, row_text):
        """foreground colours of the characters on the first screen row containing row_text"""
        for y, ln in enumerate(self.screen.display):
            if row_text in ln:
                x = ln.index(row_text)
                return [self.screen.buffer[y][i].fg for i in range(x, x + len(row_text))]
        return []

    def session(self):
        d = os.path.join(self.tmp, "plainsh")
        out = []
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if name.startswith("s."):
                out += open(os.path.join(d, name)).read().splitlines()
        return out

    def close(self):
        try:
            os.write(self.fd, b"\x03exit\r")
            time.sleep(0.2)
            os.close(self.fd)
        except OSError:
            pass
        pidfile = os.path.join(self.tmp, "plainsh", "serve.pid")
        if os.path.exists(pidfile):
            try:
                os.kill(int(open(pidfile).read().strip() or 0), 15)
            except (OSError, ValueError):
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)


def show(t, title):
    print(f"--- {title}")
    for ln in t.lines():
        print("   |" + ln)


if __name__ == "__main__":
    work = tempfile.mkdtemp(prefix="plainsh-work-")
    for name in ("a.pdf", "b.pdf", "notes.txt", "src/x.py", "src/y.py"):
        os.makedirs(os.path.dirname(os.path.join(work, name)) or work, exist_ok=True)
        open(os.path.join(work, name), "w").write("hello\n")
    t = Term(cwd=work)
    fails = []

    def check(name, ok):
        print(("ok    " if ok else "FAIL  ") + name)
        if not ok:
            fails.append(name)

    t.send(":", 0.3)
    show(t, "after typing a colon")
    col = t.colour(":")
    check("the colon is drawn in a colour of its own", bool(col) and col[0] not in ("default",))
    t.type(" how many python files are in src", wait=1.2)
    show(t, "after typing a request")
    check("the request text is coloured", t.colour("python")[:1] == col[:1])
    check("a preview of the command is shown while typing", any("find src" in ln and "wc -l" in ln for ln in t.lines()))
    t.send("\r", 0.8)
    show(t, "after Enter")
    last = t.lines()[-1]
    check("Enter puts the command on the prompt", last.startswith("[t ") and "find src -type f -iname '*.py' | wc -l" in last)
    check("the command was not run", not any(ln.strip() == "2" for ln in t.lines()))
    t.send("\r", 0.5)
    check("a second Enter runs it", any(ln.strip() == "2" for ln in t.lines()))

    t.type(": find pdf files", wait=1.0)
    t.send("\r", 0.6); t.send("\r", 0.5)
    t.type(": delete them", wait=1.0)
    show(t, "a follow-up, while typing")
    t.send("\r", 0.8)
    show(t, "a follow-up, after Enter")
    last = t.lines()[-1]
    check("a follow-up builds on the previous request", "find . -type f -iname '*.pdf' -delete" in last)
    check("removing things shows the warning", any("removes or stops" in ln for ln in t.lines()))
    t.send("\x03", 0.3)

    t.send(":", 0.2); t.type(" list files", wait=0.3); t.send("\x1b", 0.4)
    check("Esc leaves an empty prompt", t.lines()[-1].rstrip() == "[t " + os.path.basename(work) + "]$")
    t.send(":", 0.2); t.send("\x7f", 0.3)
    check("Backspace over the colon leaves", t.lines()[-1].rstrip() == "[t " + os.path.basename(work) + "]$")
    t.type("echo a:b:c", wait=0.2); t.send("\r", 0.4)
    check("a colon inside a command is an ordinary colon", any(ln.strip() == "a:b:c" for ln in t.lines()))
    t.type(": > made.txt", wait=0.3); t.send("\r", 0.5)
    check("a real ':' command still runs as typed", os.path.exists(os.path.join(work, "made.txt")))
    t.send('echo "one\r', 0.3); t.send(':two"\r', 0.4)
    check("a colon on a continuation line is ordinary", any(ln.strip() == ":two" for ln in t.lines()))
    t.send(":", 0.2); t.send("\x1b[A", 0.5)
    check("Up recalls the previous request", any(": delete them" in ln for ln in t.lines()[-3:]))
    t.send("\x1b", 0.3)
    t.type(": show free disk space", wait=1.0); t.send("\r", 0.7)
    check("an offer for a simple request", t.lines()[-1].endswith("$ df -h"))
    t.type(" /tmp", wait=0.2); t.send("\r", 0.6)              # edit the offer, then run it
    t.type(": show free disk space", wait=1.2)
    show(t, "asking again after correcting the offer")
    check("the corrected command is offered the second time", any("df -h /tmp" in ln for ln in t.lines()[-3:]))
    t.send("\x1b", 0.3)
    t.type(": make it rain", wait=1.0); t.send("\r", 0.8)
    show(t, "something it cannot do")
    check("an impossible request stays in the editor with a message", any("outside what I can do" in ln for ln in t.lines()))
    t.send("\x1b", 0.3)
    print("--- session file")
    for ln in t.session():
        print("   |" + ln)
    check("the session file records commands and requests", any(ln.startswith("run\t") for ln in t.session()) and any(ln.startswith("ask\t") for ln in t.session()))
    t.close()
    shutil.rmtree(work, ignore_errors=True)
    print("\n%d failed" % len(fails) if fails else "\nall passed")
    sys.exit(1 if fails else 0)
