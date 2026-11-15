"""Reference documentation -> data/clean/docs_man, docs_tldr, docs_gnu, docs_archwiki.

docs_man      this machine's man pages (sections 1, 2, 4, 5, 7, 8 and the C/POSIX parts of
              section 3), rendered with `man -l` at 80 columns and stripped with `col -bx`.
docs_tldr     tldr-pages (github.com/tldr-pages/tldr), English pages for common, linux,
              osx, freebsd, openbsd, netbsd, sunos, android.
docs_gnu      GNU manuals installed as info files (bash, coreutils, findutils, grep, sed,
              gawk, tar, make, ...; glibc, gcc, gdb), rendered with `info --subnodes`,
              split into documents of up to ~24 KB at node boundaries.
docs_archwiki the offline Arch Wiki from the arch-wiki-docs package (downloaded from an
              Arch mirror and unpacked, not installed), English pages, HTML -> text.

These are small but high-value; they are meant to be upsampled at training time.
Language ID is skipped (all English by construction); exact and near dedup apply.

Usage: .venv/bin/python -I data_tools/docs.py [man|tldr|gnu|archwiki|all]
"""

from __future__ import annotations

import gzip
import html
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, Filters, finalize, log, run_tasks, tidy  # noqa: E402

MAN_ROOT = Path("/usr/share/man")
MAN_SECTIONS = ["man1", "man2", "man3", "man4", "man5", "man7", "man8"]
MAN3_OK = re.compile(r"\.(3|3p|3posix|3type|3const|3head)(\.gz|\.bz2|\.xz|\.zst)?$")
INFO_ROOT = Path("/usr/share/info")
INFO_MANUALS = ["bash", "coreutils", "diffutils", "find", "gawk", "grep", "sed", "tar", "make", "gzip", "cpio",
                "m4", "wget", "which", "readline", "history", "nano", "libc", "gdb", "gcc", "binutils",
                "inetutils", "parted", "autoconf", "automake", "libtool", "groff"]
TLDR_URL = "https://github.com/tldr-pages/tldr"
TLDR_PLATFORMS = ["common", "linux", "osx", "freebsd", "openbsd", "netbsd", "sunos", "android"]
ARCH_MIRROR = "https://fastly.mirror.pkgbuild.com/extra/os/x86_64"

DOC_FILTERS = Filters(langid=False, prose=False, spam=False, min_chars=80)


# ---------------------------------------------------------------- man pages

def man_files() -> list[Path]:
    out = []
    for sec in MAN_SECTIONS:
        d = MAN_ROOT / sec
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.is_symlink():
                continue
            if sec == "man3" and not MAN3_OK.search(p.name):
                continue
            out.append(p)
    return out


def render_man(path: Path) -> str | None:
    env = {**os.environ, "MANWIDTH": "80", "MAN_KEEP_FORMATTING": "0", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    try:
        r = subprocess.run(f"man -l -- {shlex_quote(str(path))} 2>/dev/null | col -bx", shell=True,
                           capture_output=True, env=env, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    t = r.stdout.decode("utf-8", "replace")
    return t if t.strip() else None


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def produce_man(paths: list[str], drops):
    for p in paths:
        p = Path(p)
        text = render_man(p)
        if not text:
            drops["render_failed"] += 1
            continue
        lines = text.split("\n")
        # .so stubs ("See the file ...") render to almost nothing
        if len(lines) < 8:
            drops["stub"] += 1
            continue
        name = re.sub(r"\.(gz|bz2|xz|zst)$", "", p.name)
        yield {"text": tidy(text), "source": "docs_man", "meta": {"page": name, "section": p.parent.name[3:]}}


# ---------------------------------------------------------------- tldr

def produce_tldr(platform: str, drops):
    d = RAW / "tldr" / "pages" / platform
    for p in sorted(d.glob("*.md")):
        text = p.read_text(errors="replace")
        yield {"text": tidy(text), "source": "docs_tldr", "meta": {"platform": platform, "page": p.stem}}


# ---------------------------------------------------------------- GNU info manuals

NODE_HDR = re.compile(r"^File: [^,]+,\s+Node: ([^,\n]+)", re.M)


def produce_info(manual: str, drops):
    f = next((p for p in (INFO_ROOT / f"{manual}.info.gz", INFO_ROOT / f"{manual}.info") if p.exists()), None)
    if f is None:
        drops["missing"] += 1
        return
    r = subprocess.run(["info", "--subnodes", "-f", str(f), "-o", "-"], capture_output=True, timeout=300,
                       env={**os.environ, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    text = r.stdout.decode("utf-8", "replace").replace("\x1f", "")
    pieces = NODE_HDR.split(text)  # [pre, node1, body1, node2, body2, ...]
    buf, size, first_node, part = [], 0, None, 0

    def flush():
        nonlocal buf, size, first_node, part
        if buf:
            body = tidy("\n\n".join(buf))
            yield {"text": f"GNU {manual} manual: {first_node}\n\n{body}", "source": "docs_gnu",
                   "meta": {"manual": manual, "node": first_node, "part": part}}
            part += 1
        buf, size, first_node = [], 0, None

    for i in range(1, len(pieces) - 1, 2):
        node, body = pieces[i].strip(), pieces[i + 1]
        body = "\n".join(body.split("\n")[1:])  # drop the rest of the header line
        if first_node is None:
            first_node = node
        buf.append(body)
        size += len(body)
        if size > 24_000:
            yield from flush()
    yield from flush()


# ---------------------------------------------------------------- Arch Wiki

def fetch_archwiki() -> Path | None:
    dest = RAW / "archwiki"
    html_dir = dest / "usr" / "share" / "doc" / "arch-wiki" / "html" / "en"
    if html_dir.is_dir():
        return html_dir
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["pacman", "-Si", "arch-wiki-docs"], capture_output=True, text=True)
    m = re.search(r"^Version\s*:\s*(\S+)", r.stdout, re.M)
    if not m:
        log("arch-wiki-docs: version not found (pacman -Si)")
        return None
    pkg = dest / f"arch-wiki-docs-{m.group(1)}-any.pkg.tar.zst"
    subprocess.run(["curl", "-sfL", "-o", str(pkg), f"{ARCH_MIRROR}/{pkg.name}"], check=True)
    subprocess.run(["bsdtar", "-xf", str(pkg), "-C", str(dest), "usr/share/doc/arch-wiki"], check=True)
    pkg.unlink()
    return html_dir if html_dir.is_dir() else None


def wiki_html_to_text(raw: str) -> tuple[str, str]:
    import lxml.html

    doc = lxml.html.fromstring(raw)
    title = (doc.findtext(".//title") or "").replace(" - ArchWiki", "").strip()
    body = doc.get_element_by_id("bodyContent", None) or doc.get_element_by_id("content", None) or doc
    for bad in body.xpath('.//*[@id="toc" or contains(@class,"mw-editsection") or contains(@class,"navbox") '
                          'or contains(@class,"catlinks") or contains(@class,"printfooter") or @id="siteSub" '
                          'or @id="jump-to-nav" or self::script or self::style]'):
        bad.drop_tree()
    for pre in body.xpath(".//pre"):
        pre.text = "\n```\n" + pre.text_content().strip("\n") + "\n```\n"
        for ch in list(pre):
            pre.remove(ch)
    for code in body.xpath(".//code"):
        if code.getparent() is not None and code.getparent().tag != "pre":
            code.text = "`" + code.text_content() + "`"
            for ch in list(code):
                code.remove(ch)
    for h in body.xpath(".//h1|.//h2|.//h3|.//h4"):
        h.text = "\n\n## " + h.text_content().strip() + "\n\n"
        for ch in list(h):
            h.remove(ch)
    for li in body.xpath(".//li"):
        li.text = "\n- " + (li.text or "")
    for el in body.xpath(".//p|.//div|.//table|.//tr|.//dd|.//dt|.//br"):
        el.tail = "\n" + (el.tail or "")
    text = html.unescape(body.text_content())
    text = re.sub(r"[ \t]+\n", "\n", text)
    return title, tidy(text)


def produce_archwiki(paths: list[str], drops):
    for p in paths:
        raw = Path(p).read_text(errors="replace")
        if "<title>" not in raw:
            drops["no_title"] += 1
            continue
        title, text = wiki_html_to_text(raw)
        if len(text) < 300:
            drops["stub"] += 1
            continue
        yield {"text": f"{title}\n\n{text}", "source": "docs_archwiki", "meta": {"title": title, "file": Path(p).name}}


# ---------------------------------------------------------------- main

def chunks(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("man", "all"):
        files = [str(p) for p in man_files()]
        tasks = [(f"{i:04d}", c) for i, c in enumerate(chunks(files, 200))]
        run_tasks("docs_man", tasks, produce_man, DOC_FILTERS)
        finalize("docs_man", [t for t, _ in tasks], extra_stats={
            "origin": "local man pages, /usr/share/man on the build machine (CachyOS/Arch packages)",
            "licence": "per page (man-pages project licence, GPL, BSD, MIT, ...); see each page's COPYRIGHT section"})
    if mode in ("tldr", "all"):
        dest = RAW / "tldr"
        if not (dest / ".git").exists():
            subprocess.run(["git", "clone", "-q", "--depth", "1", TLDR_URL, str(dest)], check=True)
        tasks = [(p, p) for p in TLDR_PLATFORMS if (dest / "pages" / p).is_dir()]
        run_tasks("docs_tldr", tasks, produce_tldr, Filters(langid=False, prose=False, spam=False, min_chars=40))
        finalize("docs_tldr", [t for t, _ in tasks], extra_stats={
            "origin": TLDR_URL, "licence": "CC BY 4.0, (c) the tldr-pages contributors"})
    if mode in ("gnu", "all"):
        tasks = [(m, m) for m in INFO_MANUALS]
        run_tasks("docs_gnu", tasks, produce_info, DOC_FILTERS)
        finalize("docs_gnu", [t for t, _ in tasks], extra_stats={
            "origin": "local GNU info manuals, /usr/share/info", "licence": "GNU Free Documentation License"})
    if mode in ("archwiki", "all"):
        d = fetch_archwiki()
        if d is None:
            log("Arch Wiki not available; skipped")
        else:
            files = sorted(str(p) for p in d.glob("*.html"))
            tasks = [(f"{i:04d}", c) for i, c in enumerate(chunks(files, 100))]
            run_tasks("docs_archwiki", tasks, produce_archwiki, DOC_FILTERS)
            finalize("docs_archwiki", [t for t, _ in tasks], extra_stats={
                "origin": "arch-wiki-docs package (https://archlinux.org/packages/extra/any/arch-wiki-docs/), "
                          "snapshot of https://wiki.archlinux.org",
                "licence": "GNU FDL 1.3 or later (ArchWiki content)"})


if __name__ == "__main__":
    main()
