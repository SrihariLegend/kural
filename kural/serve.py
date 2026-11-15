"""Answers requests from shell hooks, so the model is loaded once instead of once per request.

One process per user, shared by every terminal. It reads requests from a named pipe in the user's
runtime folder, writes each answer to the asking shell's own pipe, and exits after ten idle minutes
(the next request starts it again). Nothing listens on the network.

Request line, fields separated by \\x1f:   seq  reply-pipe  mode  session-file  folder  text
  mode is "preview" (while typing: nothing is recorded) or "final:N" (the user accepted candidate N)
Reply line:                                seq  count  then for each candidate: command  risk  confidence  reading
  a newline inside a field is sent as \\x1e
"""
import errno
import fcntl
import os
import select
import sys
import time

IDLE_SECONDS = 600
US, RS = "\x1f", "\x1e"


def _sweep(d):
    """Remove pipes and session files left behind by shells that no longer exist."""
    for name in os.listdir(d):
        kind, _, pid = name.partition(".")
        if kind in ("r", "s") and pid.isdigit() and not os.path.exists(f"/proc/{pid}"):
            try:
                os.unlink(os.path.join(d, name))
            except OSError:
                pass


def _reply(path, line):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return                      # that shell is gone
    try:
        os.write(fd, line.encode("utf-8", "replace"))
    except OSError:
        pass
    finally:
        os.close(fd)


def main():
    from . import session
    d = session.runtime_dir()
    lock = open(os.path.join(d, "serve.pid"), "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0                    # already running
    if os.fork():
        return 0
    os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    req = os.path.join(d, "req")
    if not os.path.exists(req):
        os.mkfifo(req, 0o600)
    fd = os.open(req, os.O_RDWR | os.O_NONBLOCK)

    from . import cli, run
    run.weights()
    _sweep(d)
    buf, last = b"", time.time()
    while True:
        ready, _, _ = select.select([fd], [], [], 30)
        if not ready:
            if time.time() - last > IDLE_SECONDS:
                break
            _sweep(d)
            continue
        try:
            buf += os.read(fd, 65536)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EINTR):
                continue
            break
        lines = buf.split(b"\n")
        buf = lines.pop()
        requests = []
        for line in lines:
            f = line.decode("utf-8", "replace").split(US)
            if len(f) == 6 and os.path.dirname(f[1]) == d:
                requests.append(f)
        # while the user types, only the newest preview of each shell is worth answering
        newest = {f[1]: i for i, f in enumerate(requests) if f[2] == "preview"}
        for i, (seq, reply, mode, sess, cwd, text) in enumerate(requests):
            if mode == "preview" and newest[reply] != i:
                continue
            try:
                results = cli.answer(text, cwd=cwd or None, session_path=sess or None, final=mode.startswith("final"),
                                     choose=int(mode.partition(":")[2] or 0) if mode.startswith("final") else 0)
            except Exception:
                results = []
            fields = [seq, str(len(results))]
            for r in results:
                fields += [r["command"], str(r["risk"]), f"{r['confidence']:.3f}", r["reading"]]
            _reply(reply, US.join(x.replace("\n", RS).replace(US, " ") for x in fields) + "\n")
        last = time.time()
    return 0


if __name__ == "__main__":
    sys.exit(main())
