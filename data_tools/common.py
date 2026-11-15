"""Shared pieces of the kural pretraining-data pipeline.

Every source script (wikipedia.py, reddit.py, ...) does the same three things:

1. split its input into *tasks* (one parquet file, one dump month, ...);
2. for each task, in a worker process, turn raw records into documents
   ``{"text", "source", "meta"}``, run the shared quality filters on them and
   write the survivors to ``data/work/<source>/part-<task>.jsonl.zst`` together
   with their dedup signatures (``.npz``) and per-reason drop counts (``.json``);
3. ``finalize()``: drop exact and near duplicates across all parts (first
   occurrence wins, in task order), then write ``data/clean/<source>/shard-NNNNN.jsonl.zst``
   and ``data/clean/<source>/stats.json``.

Step 2 is resumable: a task whose ``.json`` exists is skipped, so an interrupted
run continues where it stopped. Delete ``data/work/<source>`` to start over.

Token counts everywhere are the quick estimate ``utf8_bytes / 4``.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import xxhash
import zstandard as zstd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
WORK = DATA / "work"
CLEAN = DATA / "clean"
EVAL = DATA / "eval"
PAIRS = DATA / "pairs"
LID_MODEL = DATA / "models" / "lid.176.bin"

SHARD_BYTES = 256 << 20  # uncompressed text per output shard
ZSTD_LEVEL = 6


def log(*a) -> None:
    print(time.strftime("[%H:%M:%S]"), *a, file=sys.stderr, flush=True)


def est_tokens(nbytes: int) -> int:
    return nbytes // 4


# ---------------------------------------------------------------- jsonl.zst io

def write_jsonl_zst(path: Path, rows: Iterable[dict]) -> int:
    """Write rows to path atomically (via a .tmp file). Returns the row count."""
    import orjson

    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "wb") as fh:
        with zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=0).stream_writer(fh) as w:
            for r in rows:
                w.write(orjson.dumps(r) + b"\n")
                n += 1
    os.replace(tmp, path)
    return n


def read_jsonl_zst(path: Path) -> Iterator[dict]:
    import orjson

    with open(path, "rb") as fh:
        reader = io.BufferedReader(zstd.ZstdDecompressor().stream_reader(fh), 1 << 20)
        for line in reader:
            if line.strip():
                yield orjson.loads(line)


class ShardWriter:
    """Writes data/clean/<source>/shard-NNNNN.jsonl.zst, rotating every SHARD_BYTES."""

    def __init__(self, outdir: Path, shard_bytes: int = SHARD_BYTES):
        import orjson

        self._dumps = orjson.dumps
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        for old in outdir.glob("shard-*.jsonl.zst"):
            old.unlink()
        self.shard_bytes = shard_bytes
        self.idx = -1
        self.cur = 0
        self.fh = self.w = None
        self.docs = 0
        self.text_bytes = 0

    def _open(self) -> None:
        self.close()
        self.idx += 1
        self.fh = open(self.outdir / f"shard-{self.idx:05d}.jsonl.zst", "wb")
        self.w = zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=-1).stream_writer(self.fh)
        self.cur = 0

    def write(self, row: dict) -> None:
        if self.w is None or self.cur >= self.shard_bytes:
            self._open()
        b = len(row["text"].encode("utf-8"))
        self.w.write(self._dumps(row) + b"\n")
        self.cur += b
        self.docs += 1
        self.text_bytes += b

    def close(self) -> None:
        if self.w is not None:
            self.w.close()
            self.fh.close()
            self.w = self.fh = None


# ---------------------------------------------------------------- text cleanup

_WS_RUN = re.compile(r"[ \t  -​　]+")
_MANY_NL = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def tidy(text: str) -> str:
    """Normalise Unicode (NFC), drop control chars, trim trailing spaces, cap blank lines.

    Leaves spelling, casing and punctuation alone: typos are kept on purpose.
    """
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MANY_NL.sub("\n\n", text)
    return text.strip()


EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?![\w.])")
REDDIT_USER = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]{3,20}")


def scrub_pii(text: str, reddit: bool = False) -> str:
    """Replace e-mail addresses and phone numbers (and Reddit u/names) with placeholders."""
    text = EMAIL.sub("<email>", text)
    text = PHONE.sub("<phone>", text)
    if reddit:
        text = REDDIT_USER.sub("u/<user>", text)
    return text


# ---------------------------------------------------------------- quality filters

_WORD = re.compile(r"[A-Za-z']+")


def repetition_stats(text: str) -> tuple[float, float]:
    """(fraction of characters in duplicated lines, fraction of chars in the most common line)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return 0.0, 0.0
    c = Counter(lines)
    total = sum(len(l) for l in lines) or 1
    dup = sum(len(l) * (n - 1) for l, n in c.items() if n > 1)
    top_line, top_n = c.most_common(1)[0]
    return dup / total, len(top_line) * top_n / total


def prose_checks(text: str, *, min_words: int = 20, max_dup_line_frac: float = 0.3,
                 max_symbol_frac: float = 0.25, min_alpha_frac: float = 0.55) -> str | None:
    """Gopher-style checks for natural-language documents. Returns a drop reason or None.

    Deliberately lenient: casual writing, slang and misspellings pass; walls of
    symbols, URL dumps, repeated lines and keyboard mash do not.
    """
    words = text.split()
    if len(words) < min_words:
        return "too_short"
    n = len(text)
    alpha = sum(ch.isalpha() for ch in text)
    if alpha / n < min_alpha_frac:
        return "low_alpha"
    sym = sum(text.count(s) for s in ("#", "...", "|", "{", "}", "\\"))
    if sym / max(len(words), 1) > max_symbol_frac:
        return "symbols"
    real = _WORD.findall(text)
    if real:
        mean_len = sum(map(len, real)) / len(real)
        if mean_len < 2.5 or mean_len > 12:
            return "word_length"
    dup, top = repetition_stats(text)
    if dup > max_dup_line_frac or top > 0.2:
        return "repetitive"
    # one word or character repeated many times ("lol lol lol ...", "aaaaaaaa")
    if len(words) >= 10 and Counter(words).most_common(1)[0][1] / len(words) > 0.3 and len(set(words)) < 8:
        return "repetitive"
    if re.search(r"(.)\1{40,}", text):
        return "repetitive"
    return None


SPAM_PATTERNS = re.compile(
    r"(?i)\b(?:buy now|click here to|limited time offer|free shipping|viagra|cialis|casino bonus|"
    r"onlyfans|crypto giveaway|earn \$\d+ (?:a|per) day|work from home and earn|"
    r"subscribe to my channel|follow me on (?:instagram|twitter|tiktok)|use my referral)\b"
)


def spam_check(text: str) -> str | None:
    hits = len(SPAM_PATTERNS.findall(text))
    if hits >= 2 or (hits == 1 and len(text) < 600):
        return "spam"
    urls = len(re.findall(r"https?://", text))
    words = len(text.split())
    if urls >= 5 and urls / max(words, 1) > 0.1:
        return "link_dump"
    return None


# ---------------------------------------------------------------- language id

class LangID:
    """fastText lid.176 wrapper. ``english(text)`` -> probability of __label__en."""

    def __init__(self):
        import fasttext

        fasttext.FastText.eprint = lambda *a, **k: None
        self.m = fasttext.load_model(str(LID_MODEL))

    def english(self, text: str, max_chars: int = 3000) -> float:
        t = text[:max_chars].replace("\n", " ")
        labels, probs = self.m.predict(t, k=3)
        for l, p in zip(labels, probs):
            if l == "__label__en":
                return float(p)
        return 0.0


_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def prose_only(text: str) -> str:
    """Text with fenced and inline code removed, for language ID of technical posts."""
    t = _CODE_FENCE.sub(" ", text)
    t = _INLINE_CODE.sub(" ", t)
    t = "\n".join(l for l in t.split("\n") if not l.startswith("    "))
    return t


# ---------------------------------------------------------------- dedup signatures

NUM_BANDS = 14
ROWS_PER_BAND = 8
NUM_PERM = NUM_BANDS * ROWS_PER_BAND
_rng = np.random.default_rng(0x6B7572616C)  # fixed: signatures must match across runs
_A = (_rng.integers(1, 2**63, NUM_PERM, dtype=np.uint64) * np.uint64(2) + np.uint64(1))
_B = _rng.integers(0, 2**63, NUM_PERM, dtype=np.uint64)
_SHINGLE_MULT = np.uint64(0x9E3779B97F4A7C15)
_norm_re = re.compile(r"[^\w]+", re.U)


def exact_hash(text: str) -> int:
    """Hash of the text with case, punctuation and whitespace ignored."""
    return xxhash.xxh64_intdigest(" ".join(_norm_re.sub(" ", text.lower()).split()).encode())


def minhash_bands(text: str, n: int = 5, max_words: int = 20000) -> np.ndarray:
    """MinHash over word n-gram shingles (112 permutations), folded into 14 band hashes.

    Two documents whose shingle sets have Jaccard similarity >= ~0.75 share a band with
    high probability (FineWeb uses the same 14x8 setting).
    """
    words = _norm_re.sub(" ", text.lower()).split()[:max_words]
    if not words:
        return np.zeros(NUM_BANDS, dtype=np.uint64)
    wh = np.fromiter((xxhash.xxh64_intdigest(w.encode()) for w in words), dtype=np.uint64, count=len(words))
    if len(wh) >= n:
        sh = wh[: len(wh) - n + 1].copy()
        for k in range(1, n):
            sh = sh * _SHINGLE_MULT + wh[k: len(wh) - n + 1 + k]
    else:
        sh = wh
    sh = np.unique(sh)
    with np.errstate(over="ignore"):
        # multiply-shift hashing, one row per permutation, chunked to bound memory
        mins = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
        for i in range(0, len(sh), 4096):
            blk = sh[i:i + 4096]
            hv = (_A[:, None] * blk[None, :] + _B[:, None]) >> np.uint64(16)
            np.minimum(mins, hv.min(axis=1), out=mins)
    bands = mins.reshape(NUM_BANDS, ROWS_PER_BAND)
    out = np.empty(NUM_BANDS, dtype=np.uint64)
    for b in range(NUM_BANDS):
        out[b] = xxhash.xxh64_intdigest(bands[b].tobytes(), seed=b)
    return out


# ---------------------------------------------------------------- decontamination

class Decontaminator:
    """Finds documents that contain an evaluation request verbatim.

    Patterns come from data/eval/decontam_patterns.txt (built by eval_sets.py):
    kural's hand-written test requests and the NL side of NL2Bash / InterCode-Bash /
    NL2SH-ALFA test sets. Matching ignores case and whitespace runs and requires word
    boundaries at both ends.
    """

    def __init__(self, path: Path | None = None):
        import ahocorasick

        path = path or (EVAL / "decontam_patterns.txt")
        self.n = 0
        self.A = ahocorasick.Automaton()
        if path.exists():
            for line in path.read_text().splitlines():
                p = self.norm(line)
                if p:
                    self.A.add_word(p, p)
                    self.n += 1
        if self.n:
            self.A.make_automaton()

    @staticmethod
    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    def hit(self, text: str) -> str | None:
        if not self.n:
            return None
        t = self.norm(text)
        for end, p in self.A.iter(t):
            start = end - len(p) + 1
            if (start == 0 or not t[start - 1].isalnum()) and (end + 1 == len(t) or not t[end + 1].isalnum()):
                return p
        return None


# ---------------------------------------------------------------- task runner

class Filters:
    """Per-source filter settings applied in the workers after source-specific parsing."""

    def __init__(self, *, langid: bool = True, en_threshold: float = 0.65, prose: bool = True,
                 min_chars: int = 200, max_chars: int = 1_000_000, lid_on_prose_only: bool = False,
                 prose_kwargs: dict | None = None, spam: bool = True, near_dedup: bool = True):
        self.langid = langid
        self.en_threshold = en_threshold
        self.prose = prose
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.lid_on_prose_only = lid_on_prose_only
        self.prose_kwargs = prose_kwargs or {}
        self.spam = spam
        self.near_dedup = near_dedup


_worker_state: dict = {}


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _worker_state["lid"] = None
    _worker_state["decon"] = Decontaminator()


def _run_task(source: str, task_id: str, task, produce: Callable, filters) -> dict:
    """Run one task. ``filters`` is a Filters, or {source: Filters} when ``produce`` emits
    documents for several sources (routed by doc["source"]); each source gets its own
    part files under data/work/<source>/. The group marker is data/work/<source>/part-<task>.json."""
    fmap = filters if isinstance(filters, dict) else {source: filters}
    marker = WORK / source / f"part-{task_id}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    if any(f.langid for f in fmap.values()) and _worker_state.get("lid") is None:
        _worker_state["lid"] = LangID()
    lid = _worker_state.get("lid")
    decon: Decontaminator = _worker_state["decon"]
    drops = {s: Counter() for s in fmap}
    kept = {s: [] for s in fmap}
    ex = {s: [] for s in fmap}
    bands = {s: [] for s in fmap}
    seen = Counter()
    t0 = time.time()
    for doc in produce(task, drops[source]):
        src = doc.setdefault("source", source)
        filt, d = fmap[src], drops[src]
        seen[src] += 1
        text = doc["text"]
        if len(text) < filt.min_chars:
            d["too_short"] += 1
            continue
        if len(text) > filt.max_chars:
            d["too_long"] += 1
            continue
        if filt.langid:
            sample = prose_only(text) if filt.lid_on_prose_only else text
            if len(sample.split()) < 5 or lid.english(sample) < filt.en_threshold:
                d["not_english"] += 1
                continue
        if filt.prose:
            r = prose_checks(prose_only(text) if filt.lid_on_prose_only else text, **filt.prose_kwargs)
            if r:
                d[r] += 1
                continue
        if filt.spam:
            r = spam_check(text)
            if r:
                d[r] += 1
                continue
        if decon.hit(text):
            d["decontaminated"] += 1
            continue
        doc.setdefault("meta", {})
        kept[src].append(doc)
        ex[src].append(exact_hash(text))
        bands[src].append(minhash_bands(text) if filt.near_dedup else np.zeros(NUM_BANDS, np.uint64))
    result = {}
    # write the group's own source last: its .json is the completion marker
    for src in sorted(fmap, key=lambda s: s == source):
        out_dir = WORK / src
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl_zst(out_dir / f"part-{task_id}.jsonl.zst", kept[src])
        np.savez(out_dir / f"part-{task_id}.npz",
                 exact=np.array(ex[src], dtype=np.uint64),
                 bands=np.array(bands[src], dtype=np.uint64).reshape(-1, NUM_BANDS))
        st = {"task": task_id, "seen": seen[src], "kept": len(kept[src]),
              "bytes": sum(len(x["text"].encode()) for x in kept[src]), "drops": dict(drops[src]),
              "secs": round(time.time() - t0, 1)}
        (out_dir / f"part-{task_id}.json").write_text(json.dumps(st))
        result[src] = st
    return result[source] if len(result) == 1 else {**result[source], "by_source": result}


def run_tasks(source: str, tasks: list[tuple[str, object]], produce: Callable, filters,
              workers: int | None = None) -> None:
    """Run ``produce`` over tasks in a process pool (resumable). tasks: [(task_id, arg)].

    ``produce(arg, drops)`` yields documents; ``drops`` is a Counter for parse-time drops.
    """
    out_dir = WORK / source
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    todo = [(tid, a) for tid, a in tasks if not (out_dir / f"part-{tid}.json").exists()]
    log(f"{source}: {len(tasks)} tasks, {len(todo)} to run, {workers} workers")
    if not todo:
        return
    with ProcessPoolExecutor(workers, initializer=_worker_init) as ex:
        futs = {ex.submit(_run_task, source, tid, a, produce, filters): tid for tid, a in todo}
        done = 0
        for f in as_completed(futs):
            st = f.result()
            done += 1
            if "by_source" in st:
                msg = " ".join(f"{s}:{v['kept']}/{v['seen']}={v['bytes'] / 1e6:.0f}MB" for s, v in st["by_source"].items())
            else:
                msg = f"kept {st['kept']}/{st['seen']} {st['bytes'] / 1e6:.0f} MB"
            log(f"{source}: {done}/{len(todo)} {st['task']} {msg} in {st['secs']}s")


def finalize(source: str, task_ids: list[str] | None = None, *, near_dedup: bool = True,
             budget_bytes: int | None = None, keep_work: bool = False, extra_stats: dict | None = None) -> dict:
    """Dedup across all parts of a source (in task order) and write the clean shards.

    budget_bytes: stop after this much text (parts are in task order, so pick a
    task order that puts the preferred data first).
    """
    out_dir = WORK / source
    if task_ids is None:
        task_ids = sorted(p.name[5:-5] for p in out_dir.glob("part-*.json"))
    drops: Counter = Counter()
    seen = 0
    for tid in task_ids:
        st = json.loads((out_dir / f"part-{tid}.json").read_text())
        drops.update(st["drops"])
        seen += st["seen"]
    # ---- duplicate mask (first occurrence wins)
    exacts, bands, sizes = [], [], []
    for tid in task_ids:
        z = np.load(out_dir / f"part-{tid}.npz")
        exacts.append(z["exact"])
        bands.append(z["bands"])
        sizes.append(len(z["exact"]))
    exact = np.concatenate(exacts) if exacts else np.zeros(0, np.uint64)
    band = np.concatenate(bands) if bands else np.zeros((0, NUM_BANDS), np.uint64)
    n = len(exact)
    keep = np.zeros(n, dtype=bool)
    _, first = np.unique(exact, return_index=True)
    keep[first] = True
    drops["exact_dup"] += int(n - keep.sum())
    if near_dedup and n:
        near = np.zeros(n, dtype=bool)
        for b in range(NUM_BANDS):
            col = band[:, b]
            order = np.argsort(col, kind="stable")
            sc = col[order]
            same_as_prev = np.empty(n, dtype=bool)
            same_as_prev[0] = False
            same_as_prev[1:] = sc[1:] == sc[:-1]
            # rows that share a band hash with an earlier row (by position)
            near[order[same_as_prev]] = True
        # only count near dups among rows that survived exact dedup
        before = int(keep.sum())
        keep &= ~near
        drops["near_dup"] += before - int(keep.sum())
    # ---- write
    w = ShardWriter(CLEAN / source)
    pos = 0
    over_budget = 0
    for tid, sz in zip(task_ids, sizes):
        m = keep[pos:pos + sz]
        pos += sz
        if not m.any():
            continue
        for i, doc in enumerate(read_jsonl_zst(out_dir / f"part-{tid}.jsonl.zst")):
            if not m[i]:
                continue
            if budget_bytes is not None and w.text_bytes >= budget_bytes:
                over_budget += 1
                continue
            w.write(doc)
    w.close()
    if over_budget:
        drops["over_budget"] += over_budget
    stats = {"source": source, "documents": w.docs, "text_bytes": w.text_bytes,
             "est_tokens": est_tokens(w.text_bytes), "input_docs": seen, "drops": dict(drops),
             "shards": w.idx + 1, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra_stats:
        stats.update(extra_stats)
    (CLEAN / source / "stats.json").write_text(json.dumps(stats, indent=1))
    log(f"{source}: wrote {w.docs} docs, {w.text_bytes / 1e9:.2f} GB, ~{est_tokens(w.text_bytes) / 1e9:.2f}B tokens")
    if not keep_work:
        import shutil
        shutil.rmtree(out_dir)
    return stats
