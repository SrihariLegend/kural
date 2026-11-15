"""English Wikipedia (wikimedia/wikipedia, 20231101.en) -> data/clean/wikipedia/.

Each article becomes "Title\\n\\ntext" with the trailing reference sections
("See also", "References", "External links", ...) and category lines cut off.
Stubs under 300 characters and disambiguation pages are dropped.

Input: data/raw/wikipedia/20231101.en/*.parquet (data_tools/fetch_hf.sh)
Usage: .venv/bin/python -I data_tools/wikipedia.py [--keep-raw]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, Filters, finalize, run_tasks, tidy  # noqa: E402

SOURCE = "wikipedia"
TAIL_HEADINGS = {"see also", "references", "notes", "further reading", "external links", "sources",
                 "bibliography", "citations", "footnotes", "notes and references", "references and notes",
                 "works cited", "general references", "explanatory notes", "selected bibliography"}


def strip_tail(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i > 0 and line.strip().lower() in TAIL_HEADINGS:
            return "\n".join(lines[:i])
    return text


def produce(path: str, drops):
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    for rg in range(f.metadata.num_row_groups):
        t = f.read_row_group(rg, columns=["id", "url", "title", "text"]).to_pydict()
        for pid, url, title, text in zip(t["id"], t["url"], t["title"], t["text"]):
            if "(disambiguation)" in title or "\nmay refer to:" in text[:400] or " may refer to:\n" in text[:400]:
                drops["disambiguation"] += 1
                continue
            body = tidy(strip_tail(text))
            if len(body) < 300:
                drops["stub"] += 1
                continue
            yield {"text": f"{title}\n\n{body}", "source": SOURCE,
                   "meta": {"id": pid, "title": title, "url": url}}


FILTERS = Filters(langid=False, min_chars=300,
                  prose_kwargs={"min_words": 40, "min_alpha_frac": 0.45, "max_symbol_frac": 0.5,
                                "max_dup_line_frac": 0.5})


def main() -> None:
    files = sorted((RAW / "wikipedia" / "20231101.en").glob("*.parquet"))
    assert files, "run data_tools/fetch_hf.sh first"
    tasks = [(f.stem.split("-")[1], str(f)) for f in files]
    run_tasks(SOURCE, tasks, produce, FILTERS)
    finalize(SOURCE, [t for t, _ in tasks], extra_stats={
        "origin": "https://huggingface.co/datasets/wikimedia/wikipedia (20231101.en)",
        "licence": "CC BY-SA 4.0 (text) / GFDL"})
    if "--keep-raw" not in sys.argv:
        for f in files:
            f.unlink()


if __name__ == "__main__":
    main()
