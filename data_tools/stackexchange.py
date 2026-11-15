"""Stack Exchange -> pretraining text + the English->command pair dataset.

Input: HuggingFaceTB/stackexchange_2025_md (one row per question with its answers,
bodies as the original HTML, from the 2025 Stack Exchange data dump; CC BY-SA).
The small sites are fetched by fetch_hf.sh into data/raw/se_md/<site>/; the 177
Stack Overflow parquet files (90 GB) are downloaded one by one by the workers and
deleted as soon as each is processed.

Sources written (data/clean/<source>/):
  stackexchange          unix, askubuntu, superuser, serverfault and other technical
                         and English-language sites, plus Stack Overflow questions with a
                         shell/Linux/C/systems tag (TECH_TAGS)
  stackoverflow_general  the rest of Stack Overflow, stricter scores (question >= 5, answer >= 3)
  stackexchange_math     math.stackexchange, mathoverflow, stats.stackexchange

Document: the question title and body, then up to three answers (accepted first,
then by score). HTML is turned into markdown-ish text: code blocks are fenced with
```, inline code gets backticks. Author names are not put in the text; they are
kept in meta for attribution, together with the question URL and licence.

Held-out split: questions with xxh64("<site>:<id>") % 100 < 2 never enter any
training output. They go to data/pairs/heldout.jsonl (shell-relevant questions only).

Pairs: data/pairs/stackexchange_pairs.jsonl has, for every shell-relevant question
(unix/askubuntu/superuser/serverfault, SO with a TECH_TAG) whose accepted or top
answers contain a <pre> code block: title, body, site, id, score, tags and the raw
code blocks of those answers.

Usage: .venv/bin/python -I data_tools/stackexchange.py [sites|so|finalize|all]
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PAIRS, RAW, WORK, Decontaminator, Filters, finalize, log, run_tasks,  # noqa: E402
                    scrub_pii, tidy)

import xxhash  # noqa: E402

REPO = "HuggingFaceTB/stackexchange_2025_md"
SHELL_SITES = {"unix.stackexchange.com", "askubuntu.com", "superuser.com", "serverfault.com"}
OTHER_SITES = {"codereview.stackexchange.com", "softwareengineering.stackexchange.com", "cs.stackexchange.com",
               "english.stackexchange.com", "ell.stackexchange.com", "vi.stackexchange.com",
               "emacs.stackexchange.com", "raspberrypi.stackexchange.com", "retrocomputing.stackexchange.com",
               "networkengineering.stackexchange.com"}
MATH_SITES = {"math.stackexchange.com", "mathoverflow.net", "stats.stackexchange.com"}
TECH_TAGS = set("""
bash shell sh zsh ksh fish dash linux unix posix sed awk gawk grep find regex xargs c gcc clang make makefile
gnu-make cmake autotools systemd systemctl docker dockerfile docker-compose git ssh openssh rsync cron crontab tar
gzip zip unzip curl wget command-line terminal shell-script scripting ubuntu debian centos fedora archlinux
redhat rhel alpine linux-kernel kernel linux-device-driver embedded-linux busybox coreutils vim emacs tmux
screen nginx apache2 iptables networking permissions file-permissions chmod sudo environment-variables path
pipe stdout stderr stdin io-redirection heredoc process signals fork exec pthreads posix-api glibc gdb valgrind
strace ld linker shared-libraries elf pointers malloc segmentation-fault c-preprocessor printf scanf string.h
memory-management file-io filesystems mount fstab partitioning lvm raid disk zfs btrfs ext4 nfs samba
logging syslog journald users groups bashrc profile alias jq yaml awk-script perl-one-liner one-liner
text-processing csv sort uniq cut tr wc diff patch ln symlink inotify watch top ps kill htop
""".split())
SO_TECH_MIN = (1, 1)      # (question score, answer score) for TECH_TAGS questions
SO_GENERAL_MIN = (5, 3)   # for the rest of Stack Overflow
SHELL_SITE_MIN = (0, 1)   # unix, askubuntu, superuser, serverfault: any question with a good answer
SITE_MIN = (1, 1)         # for the other Stack Exchange sites
MAX_ANSWERS = 3
HELDOUT_PCT = 2


def is_heldout(site: str, qid: int) -> bool:
    return xxhash.xxh64_intdigest(f"{site}:{qid}".encode()) % 100 < HELDOUT_PCT


# ---------------------------------------------------------------- HTML -> text

_PRE = re.compile(r"<pre\b[^>]*>\s*(?:<code\b[^>]*>)?(.*?)(?:</code>\s*)?</pre>", re.S | re.I)
_CODE = re.compile(r"<code\b[^>]*>(.*?)</code>", re.S | re.I)
_BLOCKQ = re.compile(r"<blockquote\b[^>]*>((?:(?!<blockquote).)*?)</blockquote>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_PH = "\x00{}\x00"


def pre_blocks(body: str) -> list[str]:
    """Raw contents of <pre> code blocks (HTML entities decoded, tags removed)."""
    return [html.unescape(_TAG.sub("", m.group(1))).strip("\n") for m in _PRE.finditer(body or "")]


def html_to_text(body: str) -> str:
    if not body:
        return ""
    blocks: list[str] = []

    def pre(m):
        blocks.append(html.unescape(_TAG.sub("", m.group(1))).strip("\n"))
        return "\n\n" + _PH.format(len(blocks) - 1) + "\n\n"

    s = _PRE.sub(pre, body)
    s = _CODE.sub(lambda m: "`" + html.unescape(_TAG.sub("", m.group(1))) + "`", s)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<hr\s*/?>", "\n\n---\n\n", s, flags=re.I)
    s = re.sub(r"<h[1-6][^>]*>", "\n\n## ", s, flags=re.I)
    s = re.sub(r"</h[1-6]>", "\n\n", s, flags=re.I)
    s = re.sub(r"<li[^>]*>", "\n- ", s, flags=re.I)
    s = re.sub(r"</(?:p|ul|ol|div|table|tr)>", "\n\n", s, flags=re.I)
    s = re.sub(r"<img[^>]*>", "", s, flags=re.I)
    for _ in range(4):  # innermost quotes first
        s2 = _BLOCKQ.sub(lambda m: "\n\n" + "\n".join("> " + l for l in _TAG.sub("", m.group(1)).strip().split("\n")) + "\n\n", s)
        if s2 == s:
            break
        s = s2
    s = html.unescape(_TAG.sub("", s))
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub("\x00(\\d+)\x00", lambda m: "```\n" + blocks[int(m.group(1))] + "\n```", s)
    return tidy(s)


def parse_tags(tags: str | None) -> list[str]:
    return [t for t in re.split(r"[<>|]+", tags or "") if t]


# ---------------------------------------------------------------- per-question

def classify(site: str, tags: list[str]) -> str:
    if site in MATH_SITES:
        return "stackexchange_math"
    if site == "stackoverflow.com":
        return "stackexchange" if TECH_TAGS.intersection(tags) else "stackoverflow_general"
    return "stackexchange"


def thresholds(site: str, source: str) -> tuple[int, int]:
    if site == "stackoverflow.com":
        return SO_TECH_MIN if source == "stackexchange" else SO_GENERAL_MIN
    return SHELL_SITE_MIN if site in SHELL_SITES else SITE_MIN


def shell_relevant(site: str, tags: list[str]) -> bool:
    return site in SHELL_SITES or (site == "stackoverflow.com" and bool(TECH_TAGS.intersection(tags)))


def pick_answers(answers: list[dict], accepted_id, min_score: int) -> list[dict]:
    good = [a for a in answers or [] if a.get("Body") and ((a.get("Score") or 0) >= min_score or
                                                         a.get("Id") == accepted_id or a.get("IsAccepted"))]
    good.sort(key=lambda a: (not (a.get("Id") == accepted_id or a.get("IsAccepted")), -(a.get("Score") or 0)))
    return good[:MAX_ANSWERS]


def question_doc(site: str, q: dict, answers: list[dict], tags: list[str]) -> str:
    parts = [f"Question: {html.unescape(q['Title'] or '')}", html_to_text(q["Body"])]
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    for a in answers:
        acc = a.get("IsAccepted") or a.get("Id") == q.get("AcceptedAnswerId")
        parts.append(("Accepted answer:" if acc else "Answer:") + "\n" + html_to_text(a["Body"]))
    return scrub_pii("\n\n".join(p for p in parts if p))


COLUMNS = ["Id", "Title", "Body", "Tags", "Score", "AcceptedAnswerId", "ClosedDate", "ContentLicense",
           "OwnerDisplayName", "UserDisplayName", "Answers"]


def iter_questions(path: str):
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    cols = [c for c in COLUMNS if c in f.schema_arrow.names]
    for rg in range(f.metadata.num_row_groups):
        for q in f.read_row_group(rg, columns=cols).to_pylist():
            yield q


def process_file(site: str, path: str, task_id: str, drops) -> dict[str, list]:
    """Returns {source: [docs]} and writes pairs / held-out rows for this file."""
    out: dict[str, list] = {"stackexchange": [], "stackoverflow_general": [], "stackexchange_math": []}
    decon = Decontaminator()
    pairs, held = [], []
    for q in iter_questions(path):
        qid = q["Id"]
        tags = parse_tags(q.get("Tags"))
        source = classify(site, tags)
        relevant = shell_relevant(site, tags)
        heldout = is_heldout(site, qid)
        url = f"https://{site}/questions/{qid}"
        if relevant:
            code_ans = pick_answers(q.get("Answers"), q.get("AcceptedAnswerId"), 1)
            if not code_ans and q.get("Answers"):
                code_ans = sorted(q["Answers"], key=lambda a: -(a.get("Score") or 0))[:1]
            blocks = []
            for a in code_ans:
                for c in pre_blocks(a.get("Body")):
                    blocks.append({"answer_id": a.get("Id"), "answer_score": a.get("Score"),
                                   "accepted": bool(a.get("IsAccepted") or a.get("Id") == q.get("AcceptedAnswerId")),
                                   "code": c})
            if blocks:
                title = html.unescape(q.get("Title") or "")
                row = {"site": site, "id": qid, "url": url, "title": title, "body": html_to_text(q.get("Body")),
                       "score": q.get("Score"), "tags": tags, "license": q.get("ContentLicense"),
                       "code_blocks": blocks}
                if heldout:
                    held.append(row)
                elif decon.hit(title + "\n" + row["body"]):
                    drops["pairs_decontaminated"] += 1
                else:
                    pairs.append(row)
        if heldout:
            drops["heldout"] += 1
            continue
        qmin, amin = thresholds(site, source)
        if (q.get("Score") or 0) < qmin:
            drops["low_score_question"] += 1
            continue
        answers = pick_answers(q.get("Answers"), q.get("AcceptedAnswerId"), amin)
        if not answers:
            drops["no_good_answer"] += 1
            continue
        text = question_doc(site, q, answers, tags)
        authors = [q.get("OwnerDisplayName") or q.get("UserDisplayName")] + [a.get("UserDisplayName") for a in answers]
        out[source].append({"text": text, "source": source, "meta": {
            "site": site, "id": qid, "url": url, "score": q.get("Score"), "tags": tags,
            "answer_ids": [a.get("Id") for a in answers], "authors": authors,
            "license": q.get("ContentLicense")}})
    pdir = WORK / "se_pairs"
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / f"{task_id}.pairs.jsonl.tmp", "w") as fh:
        for r in pairs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(pdir / f"{task_id}.heldout.jsonl.tmp", "w") as fh:
        for r in held:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(pdir / f"{task_id}.pairs.jsonl.tmp", pdir / f"{task_id}.pairs.jsonl")
    os.replace(pdir / f"{task_id}.heldout.jsonl.tmp", pdir / f"{task_id}.heldout.jsonl")
    return out


def produce(arg, drops):
    """One parquet file -> documents for all three sources (routed by doc["source"])."""
    site, path, task_id = arg
    local, downloaded = path, False
    if not os.path.exists(local):  # a Stack Overflow file: fetch it now, delete it after
        from huggingface_hub import hf_hub_download

        for attempt in range(6):
            try:
                local = hf_hub_download(REPO, path, repo_type="dataset", local_dir=str(RAW / "se_so"))
                break
            except Exception as e:  # network hiccup: retry
                log(f"download {path} failed ({e}); retry {attempt + 1}")
                time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError(f"could not download {path}")
        downloaded = True
    out = process_file(site, local, task_id, drops)
    if downloaded:
        os.unlink(local)
    for docs in out.values():
        yield from docs


FILTERS = {
    "stackexchange": Filters(langid=True, lid_on_prose_only=True, en_threshold=0.5, min_chars=150,
                             prose_kwargs={"min_words": 15, "min_alpha_frac": 0.3, "max_symbol_frac": 1.0,
                                           "max_dup_line_frac": 0.5}),
    "stackexchange_math": Filters(langid=True, lid_on_prose_only=True, en_threshold=0.4, min_chars=150, prose=False),
}
FILTERS["stackoverflow_general"] = FILTERS["stackexchange"]


def site_tasks() -> list[tuple[str, tuple]]:
    tasks = []
    for d in sorted((RAW / "se_md").iterdir()):
        for i, f in enumerate(sorted(d.glob("*.parquet"))):
            tasks.append((f"{d.name}-{i:03d}", (d.name, str(f), f"{d.name}-{i:03d}")))
    return tasks


def so_tasks() -> list[tuple[str, tuple]]:
    from huggingface_hub import list_repo_files

    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("stackoverflow.com/") and f.endswith(".parquet"))
    return [(f"stackoverflow.com-{i:03d}", ("stackoverflow.com", f, f"stackoverflow.com-{i:03d}"))
            for i, f in enumerate(files)]


def write_pairs() -> None:
    pdir = WORK / "se_pairs"
    PAIRS.mkdir(parents=True, exist_ok=True)
    n = h = 0
    with open(PAIRS / "stackexchange_pairs.jsonl", "w") as fp, open(PAIRS / "heldout.jsonl", "w") as fh:
        for f in sorted(pdir.glob("*.pairs.jsonl")):
            for line in open(f):
                fp.write(line)
                n += 1
        for f in sorted(pdir.glob("*.heldout.jsonl")):
            for line in open(f):
                fh.write(line)
                h += 1
    log(f"pairs: {n} questions in stackexchange_pairs.jsonl, {h} held out")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.environ.setdefault("HF_HOME", str(RAW.parent / "hf_home"))
    tasks = []
    if mode in ("sites", "all"):
        tasks += site_tasks()
    if mode in ("so", "all"):
        tasks += so_tasks()
    if mode != "finalize":
        # SO downloads: fewer concurrent workers keeps transient disk use ~6 GB
        run_tasks("stackexchange", tasks, produce, FILTERS, workers=12 if mode in ("so", "all") else None)
    if mode in ("finalize", "all"):
        ids = [t for t, _ in site_tasks() + so_tasks()]
        # order matters for dedup and budgets: shell sites and SO first
        common = {"origin": f"https://huggingface.co/datasets/{REPO} (Stack Exchange data dump 2025)",
                  "licence": "CC BY-SA 2.5/3.0/4.0 per post (see meta.license); attribution: meta.url + meta.authors"}
        finalize("stackexchange", ids, extra_stats=common)
        finalize("stackoverflow_general", ids, extra_stats=common)
        finalize("stackexchange_math", ids, extra_stats=common)
        write_pairs()


if __name__ == "__main__":
    main()
