"""Curated web text: FineWeb-Edu and FineMath -> data/clean/fineweb_edu/, data/clean/finemath/.

fineweb_edu: HuggingFaceFW/fineweb-edu, sample/10BT files 000-003 (educational-quality
             classifier score >= 3, already English-filtered and PII-anonymised upstream).
finemath:    HuggingFaceTB/finemath, finemath-4plus files 0-7 (math-quality score >= 4).

Both get the shared prose, spam, decontamination and dedup filters plus a check for
"spun" machine-rewritten text (quotes glued to words). The output is capped by a
byte budget; tasks are row-group slices in file order (the files are already shuffled).

Usage: .venv/bin/python -I data_tools/web.py [fineweb_edu|finemath] [--keep-raw]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, Filters, finalize, run_tasks, tidy  # noqa: E402

CONFIG = {
    "fineweb_edu": {
        "glob": "fineweb_edu/sample/10BT/*.parquet",
        "budget": 8_200_000_000,  # bytes, ~2.05B tokens
        "origin": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu (sample/10BT, files 000-003)",
        "licence": "ODC-By 1.0; also subject to the Common Crawl terms of use",
        "filters": Filters(langid=True, en_threshold=0.65, min_chars=300),
    },
    "finemath": {
        "glob": "finemath/finemath-4plus/*.parquet",
        "budget": 2_600_000_000,  # bytes, ~0.65B tokens; math Stack Exchange adds the rest
        "origin": "https://huggingface.co/datasets/HuggingFaceTB/finemath (finemath-4plus, files 0-7)",
        "licence": "ODC-By 1.0; also subject to the Common Crawl terms of use",
        # LaTeX is symbol-heavy: loosen the symbol and alpha checks
        "filters": Filters(langid=True, en_threshold=0.5, min_chars=300,
                           prose_kwargs={"min_alpha_frac": 0.35, "max_symbol_frac": 2.0, "max_dup_line_frac": 0.4}),
    },
}
ROWS_PER_TASK = 40_000
SPUN = re.compile(r"[”″][A-Za-z]|[A-Za-z][“]")  # closing quote glued to the next word, or vice versa


def produce(arg, drops):
    import pyarrow.parquet as pq

    source, path, rg0, rg1 = arg
    f = pq.ParquetFile(path)
    for rg in range(rg0, rg1):
        t = f.read_row_group(rg, columns=["text", "url"]).to_pydict()
        for text, url in zip(t["text"], t["url"]):
            if len(SPUN.findall(text)) >= 4:
                drops["spun_text"] += 1
                continue
            yield {"text": tidy(text), "source": source, "meta": {"url": url}}


def tasks_for(source: str) -> list:
    import pyarrow.parquet as pq

    tasks = []
    for fi, path in enumerate(sorted(RAW.glob(CONFIG[source]["glob"]))):
        md = pq.ParquetFile(path).metadata
        start, rows = 0, 0
        for rg in range(md.num_row_groups):
            rows += md.row_group(rg).num_rows
            if rows >= ROWS_PER_TASK or rg == md.num_row_groups - 1:
                tasks.append((f"{fi:02d}-{start:04d}", (source, str(path), start, rg + 1)))
                start, rows = rg + 1, 0
    return tasks


def main() -> None:
    source = sys.argv[1]
    cfg = CONFIG[source]
    tasks = tasks_for(source)
    assert tasks, "run data_tools/fetch_hf.sh first"
    run_tasks(source, tasks, produce, cfg["filters"])
    finalize(source, [t for t, _ in tasks], budget_bytes=cfg["budget"],
             extra_stats={"origin": cfg["origin"], "licence": cfg["licence"]})
    if "--keep-raw" not in sys.argv:
        for p in RAW.glob(cfg["glob"]):
            p.unlink()


if __name__ == "__main__":
    main()
