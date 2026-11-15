"""Reddit threads -> data/clean/reddit_tech/ and data/clean/reddit_general/.

Input: the monthly Pushshift / Arctic Shift dumps (RC_YYYY-MM.zst comments,
RS_YYYY-MM.zst submissions) mirrored at huggingface.co/datasets/Avni2110/pushshift-reddit
(original_dump/). They are *streamed*: downloaded, decompressed and filtered on the
fly, never stored, so disk use stays small.

Phase 1 (extract, one task per month): keep submissions and comments from the
subreddits in TECH_SUBS / GENERAL_SUBS. Dropped: [deleted]/[removed] bodies (posts of
deleted accounts whose text survives are kept), bots (AutoModerator, names ending in
"bot", "I am a bot" texts), NSFW submissions, comments under MIN_SCORE (2 for technical,
3 for general, 5 for AskReddit), comments under 20 characters. Rows are bucketed by thread id into data/work/reddit_extract/.

Phase 2 (threads, one task per bucket): each thread becomes one document:
"r/<sub>: <title>", the self text, then the top comments by score (30 for technical
subreddits, 8 for general ones), separated by "---". Comment trees are not available
in the dump (no parent ids are used), so comments are flat. Usernames, e-mail addresses
and phone numbers are removed from the text; no author names are kept.

Usage: .venv/bin/python -I data_tools/reddit.py [extract|threads|finalize|all]
"""

from __future__ import annotations

import html
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (WORK, Filters, finalize, log, read_jsonl_zst, run_tasks, scrub_pii,  # noqa: E402
                    tidy, write_jsonl_zst)

import xxhash  # noqa: E402

BASE = "https://huggingface.co/datasets/Avni2110/pushshift-reddit/resolve/main/original_dump"
EXTRACT = WORK / "reddit_extract"
BUCKETS = 256

TECH_SUBS = set("""
linux linuxquestions bash commandline zsh archlinux ubuntu debian fedora gentoo voidlinux nixos opensuse manjaro
linuxmint centos redhat freebsd openbsd netbsd unix sysadmin linuxadmin linux4noobs unixporn linuxmasterrace
linux_gaming linuxhardware c_programming cpp rust golang programming learnprogramming askprogramming
programminglanguages compsci computerscience askcomputerscience osdev embedded kernel compilers
homelab selfhosted datahoarder raspberry_pi docker kubernetes devops ansible aws networking netsec openwrt pihole
vim neovim emacs git techsupport talesfromtechsupport python learnpython webdev javascript cscareerquestions
math learnmath askmath matheducation
""".split())
GENERAL_SUBS = set("""
askreddit explainlikeimfive nostupidquestions askscience askhistorians changemyview trueaskreddit outoftheloop
tooafraidtoask casualconversation todayilearned lifeprotips youshouldknow personalfinance books writing
askculinary cooking diy fitness travel askengineers askphysics science history philosophy socialskills
iwanttolearn dataisbeautiful space futurology technology gardening woodworking homeimprovement legaladvice
askacademia gradschool languagelearning words etymology
""".split())
MIN_SCORE_TECH, MIN_SCORE_GENERAL, MIN_SCORE_ASKREDDIT = 2, 3, 5
MAX_COMMENTS_TECH, MAX_COMMENTS_GENERAL = 30, 8


def months() -> list[str]:
    """2008-01..2017-12 every month, then 2018-2023 every fourth month (Jan, May, Sep)."""
    out = [f"{y}-{m:02d}" for y in range(2008, 2018) for m in range(1, 13)]
    out += [f"{y}-{m:02d}" for y in range(2018, 2024) for m in (1, 5, 9)]
    return out


_SUB_RE = re.compile(rb'"subreddit":"([^"]+)"')
BOT_TEXT = re.compile(r"(?i)\bi am a bot\b|\bi'm a bot\b|beep boop|this action was performed automatically")
DEAD = {"[deleted]", "[removed]", "", None}


def is_bot(author: str | None) -> bool:
    if not author:
        return False
    a = author.lower()
    return a == "automoderator" or a.endswith("bot") or a.endswith("_bot") or "bot_" in a


def stream_lines(url: str):
    """Yield raw JSON lines of a remote .zst dump (streamed; retried from scratch on failure)."""
    import requests
    import zstandard as zstd

    r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "kural-data/1.0"})
    if r.status_code == 404:
        return
    r.raise_for_status()
    r.raw.decode_content = True
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    reader = io.BufferedReader(dctx.stream_reader(r.raw, read_size=1 << 22), 1 << 22)
    for line in reader:
        yield line


def cat_of(sub: str) -> str | None:
    s = sub.lower()
    if s in TECH_SUBS:
        return "tech"
    if s in GENERAL_SUBS:
        return "general"
    return None


def min_score(sub: str) -> int:
    s = sub.lower()
    if s == "askreddit":
        return MIN_SCORE_ASKREDDIT
    return MIN_SCORE_TECH if s in TECH_SUBS else MIN_SCORE_GENERAL


def extract_month(month: str) -> dict:
    import orjson

    done = EXTRACT / month / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    out_dir = EXTRACT / month
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"month": month}
    for kind, prefix in (("submissions", "RS"), ("comments", "RC")):
        for attempt in range(6):
            buckets = defaultdict(list)
            c = Counter()
            try:
                for line in stream_lines(f"{BASE}/{kind}/{prefix}_{month}.zst"):
                    c["lines"] += 1
                    m = _SUB_RE.search(line)
                    if not m:
                        continue
                    sub = m.group(1).decode("utf-8", "replace")
                    if cat_of(sub) is None:
                        continue
                    o = orjson.loads(line)
                    if is_bot(o.get("author")):
                        c["bot"] += 1
                        continue
                    if kind == "submissions":
                        if o.get("over_18"):
                            c["nsfw"] += 1
                            continue
                        title = o.get("title") or ""
                        selftext = o.get("selftext") or ""
                        if selftext in ("[deleted]", "[removed]"):
                            selftext = ""
                        if (o.get("removed_by_category") or "") in ("moderator", "deleted", "automod_filtered"):
                            c["removed"] += 1
                            continue
                        row = {"id": o.get("id"), "sub": sub, "title": title, "selftext": selftext,
                               "score": o.get("score") or 0, "created": int(float(o.get("created_utc") or 0)),
                               "url": o.get("url") if not o.get("is_self") else None}
                        key = row["id"]
                    else:
                        body = o.get("body")
                        if body in DEAD:
                            c["deleted"] += 1
                            continue
                        if (o.get("score") or 0) < min_score(sub):
                            c["low_score"] += 1
                            continue
                        if len(body) < 20:
                            c["short"] += 1
                            continue
                        if BOT_TEXT.search(body):
                            c["bot"] += 1
                            continue
                        lid = (o.get("link_id") or "")[3:]
                        row = {"link": lid, "sub": sub, "body": body, "score": o.get("score") or 0,
                               "created": int(float(o.get("created_utc") or 0))}
                        key = lid
                    if not key:
                        continue
                    buckets[xxhash.xxh64_intdigest(key.encode()) % BUCKETS].append(row)
                    c["kept"] += 1
                break
            except Exception as e:  # dropped connection etc.: start this file again
                log(f"{prefix}_{month}: {type(e).__name__}: {e}; retry {attempt + 1}")
                time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {prefix}_{month}")
        for b, rows in buckets.items():
            write_jsonl_zst(out_dir / f"{kind}-b{b:02d}.jsonl.zst", rows)
        stats[kind] = dict(c)
    done.write_text(json.dumps(stats))
    return stats


def run_extract(workers: int = 8) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    todo = [m for m in months() if not (EXTRACT / m / "done.json").exists()]
    log(f"reddit extract: {len(todo)} months to stream, {workers} at a time")
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(extract_month, m): m for m in todo}
        for f in as_completed(futs):
            st = f.result()
            log(f"reddit extract {st['month']}: subs {st['submissions'].get('kept', 0)} "
                f"comments {st['comments'].get('kept', 0)} of {st['comments'].get('lines', 0)} lines")


# ---------------------------------------------------------------- threads

def clean_body(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"(?m)^&gt;", ">", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", s)  # [text](url) -> text
    return tidy(scrub_pii(s, reddit=True))


def produce_threads(bucket: int, drops):
    subs: dict[str, dict] = {}
    comments: dict[str, list] = defaultdict(list)
    for mdir in sorted(EXTRACT.iterdir()):
        p = mdir / f"submissions-b{bucket:02d}.jsonl.zst"
        if p.exists():
            for r in read_jsonl_zst(p):
                subs[r["id"]] = r
        p = mdir / f"comments-b{bucket:02d}.jsonl.zst"
        if p.exists():
            for r in read_jsonl_zst(p):
                comments[r["link"]].append(r)
    ids = set(subs) | set(comments)
    for tid in sorted(ids):
        s = subs.get(tid)
        cs = comments.get(tid, [])
        sub = (s or cs[0])["sub"]
        cat = cat_of(sub)
        cap = MAX_COMMENTS_TECH if cat == "tech" else MAX_COMMENTS_GENERAL
        seen, top = set(), []
        for c in sorted(cs, key=lambda c: -c["score"]):
            b = clean_body(c["body"])
            k = b.lower()
            if len(b) >= 20 and k not in seen:
                seen.add(k)
                top.append(b)
            if len(top) >= cap:
                break
        parts = []
        if s:
            if s["score"] < (1 if cat == "tech" else 2) and not top:
                drops["low_score_thread"] += 1
                continue
            parts.append(f"r/{sub}: {clean_body(s['title'])}")
            if s["selftext"]:
                parts.append(clean_body(s["selftext"]))
            if not top and len(s["selftext"]) < 200:
                drops["no_discussion"] += 1
                continue
        else:
            if len(top) < 2:
                drops["orphan_comment"] += 1
                continue
            parts.append(f"r/{sub}: (comments from a thread)")
        parts.extend(top)
        text = "\n\n---\n\n".join(p for p in parts if p)
        created = (s or cs[0])["created"]
        yield {"text": text, "source": "reddit_tech" if cat == "tech" else "reddit_general",
               "meta": {"subreddit": sub, "thread": tid, "created_utc": created, "n_comments": len(top)}}


FILTERS = {
    "reddit_tech": Filters(langid=True, lid_on_prose_only=True, en_threshold=0.6, min_chars=200,
                           prose_kwargs={"min_words": 30, "min_alpha_frac": 0.4, "max_symbol_frac": 1.0,
                                         "max_dup_line_frac": 0.4}),
    "reddit_general": Filters(langid=True, en_threshold=0.7, min_chars=200,
                              prose_kwargs={"min_words": 30, "max_dup_line_frac": 0.4}),
}
GENERAL_BUDGET = 7_500_000_000  # bytes (~1.9B tokens): AskReddit alone would be several times this
ORIGIN = ("https://huggingface.co/datasets/Avni2110/pushshift-reddit (Pushshift / Arctic Shift monthly dumps "
          "RC_*/RS_*.zst)")
LICENCE = ("Reddit user content; no open licence. Redistribution is governed by Reddit's terms; "
           "use for local model training only, do not republish the text.")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("extract", "all"):
        run_extract(int(os.environ.get("REDDIT_STREAMS", "8")))
    if mode in ("threads", "all"):
        tasks = [(f"b{b:02d}", b) for b in range(BUCKETS)]
        run_tasks("reddit_tech", tasks, produce_threads, FILTERS, workers=12)
    if mode in ("finalize", "all"):
        ids = [f"b{b:02d}" for b in range(BUCKETS)]
        extra = {"origin": ORIGIN, "licence": LICENCE, "months": months()}
        finalize("reddit_tech", ids, extra_stats=extra)
        finalize("reddit_general", ids, budget_bytes=GENERAL_BUDGET, extra_stats=extra)


if __name__ == "__main__":
    main()
