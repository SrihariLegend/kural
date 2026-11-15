"""Write data/MANIFEST.md from data/clean/*/stats.json and the descriptions below.

Usage: .venv/bin/python -I data_tools/manifest.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN, DATA, EVAL, PAIRS  # noqa: E402

GROUPS = [  # (mixture slot, sources)
    ("1. Wikipedia", ["wikipedia"]),
    ("2. Reddit", ["reddit_tech", "reddit_general"]),
    ("3. Stack Exchange", ["stackexchange", "stackoverflow_general"]),
    ("4. Web (educational)", ["fineweb_edu"]),
    ("5. Math", ["finemath", "stackexchange_math"]),
    ("6. Code", ["code", "code_c_more"]),
    ("7. Docs", ["docs_man", "docs_tldr", "docs_gnu", "docs_archwiki"]),
]

NOTES = {
    "wikipedia": "Script: `wikipedia.py`. Articles as `Title\\n\\ntext`; reference sections and category lines "
                 "cut; disambiguation pages and stubs (< 300 chars) dropped. Attribution: Wikipedia contributors; "
                 "the article URL is in `meta.url`. Share-alike: derived text must stay CC BY-SA.",
    "reddit_tech": "Script: `reddit.py`. Technical and math subreddits (linux, linuxquestions, bash, commandline, "
                   "archlinux, ubuntu, sysadmin, linux4noobs, unixporn, C_Programming, programming, "
                   "learnprogramming, cpp, rust, homelab, selfhosted, ... see `TECH_SUBS`). Months 2008-01..2017-12 "
                   "and every fourth month 2018-2023. One document per thread: title, self text, top 30 comments "
                   "(comment score >= 2). Usernames (u/...), e-mail addresses and phone numbers removed; no author "
                   "names kept. Bots, [deleted]/[removed], NSFW threads, comments < 20 chars dropped.",
    "reddit_general": "Script: `reddit.py`. General conversation subreddits (AskReddit with comment score >= 5; "
                      "explainlikeimfive, NoStupidQuestions, askscience, AskHistorians, changemyview, "
                      "CasualConversation, ... with score >= 3; see `GENERAL_SUBS`). Top 8 comments per thread. "
                      "Same cleaning as reddit_tech. Capped at 7.5 GB of text (random by thread-id bucket).",
    "stackexchange": "Script: `stackexchange.py`. unix.stackexchange, askubuntu, superuser, serverfault (question "
                     "score >= 0, answers score >= 1 or accepted), codereview, softwareengineering, cs, english, "
                     "ell, vi, emacs, raspberrypi, retrocomputing, networkengineering (score >= 1), and Stack "
                     "Overflow questions with a shell/Linux/C/systems tag (`TECH_TAGS`: bash, shell, sh, zsh, linux, "
                     "unix, sed, awk, grep, find, regex, c, gcc, make, systemd, docker, git, ssh, rsync, cron, tar, "
                     "...; score >= 1). Question + up to 3 answers (accepted first). 2% of questions held out.",
    "stackoverflow_general": "Script: `stackexchange.py`. All other Stack Overflow questions with question score "
                             ">= 5 and answer score >= 3 (user approved Stack Overflow beyond the listed tags).",
    "stackexchange_math": "Script: `stackexchange.py`. math.stackexchange, mathoverflow, stats.stackexchange "
                          "(score >= 1, LaTeX kept as written).",
    "fineweb_edu": "Script: `web.py`. FineWeb-Edu sample/10BT, first 4 of 14 files, capped at 8.2 GB of text.",
    "finemath": "Script: `web.py`. FineMath finemath-4plus files 0-7, capped at 2.6 GB of text.",
    "code": "Script: `code.py`. Curated repositories first (BSD userlands, musl, toybox, suckless sbase/ubase/st/"
            "dwm/dmenu, sqlite, nginx, curl, tmux, openssh, jq, lua, libuv, zlib, lz4, libevent, cJSON, jansson, "
            "oh-my-zsh, nvm, neofetch, rbenv, pyenv, bats, bash-it, n, fzf shell scripts, docker-install), then "
            "codeparrot/github-code-clean: permissive licences only (mit, apache-2.0, bsd-2/3-clause, isc, "
            "unlicense, cc0-1.0); Shell/Makefile/Dockerfile from all 880 shards, C from the first 110. Vendored, "
            "generated, minified files and data tables dropped. Licence notices inside files are kept; "
            "`meta.repo`, `meta.path`, `meta.license` give provenance.",
    "code_c_more": "Script: `code.py c_more`. C files (permissive licences) from github-code-clean shards "
                   "110-199, same file filters as `code`; deduplicated within itself only.",
    "docs_man": "Script: `docs.py`. Man pages installed on the build machine, sections 1, 2, 3 (C/POSIX only), "
                "4, 5, 7, 8, rendered at 80 columns.",
    "docs_tldr": "Script: `docs.py`. tldr-pages English pages (common, linux, osx, freebsd, openbsd, netbsd, sunos, "
                 "android). **Attribution required (CC BY 4.0): \"tldr-pages, (c) the tldr-pages contributors, "
                 "https://github.com/tldr-pages/tldr, CC BY 4.0\".**",
    "docs_gnu": "Script: `docs.py`. GNU manuals from the installed info files (bash, coreutils, findutils, grep, "
                "sed, gawk, tar, make, gzip, cpio, m4, wget, readline, nano, glibc, gcc, gdb, binutils, autoconf, "
                "automake, libtool, groff, ...), split at node boundaries into ~24 KB documents.",
    "docs_archwiki": "Script: `docs.py`. ArchWiki snapshot shipped in the `arch-wiki-docs` package "
                     "(downloaded from an Arch mirror and unpacked, not installed). Attribution: ArchWiki "
                     "contributors, https://wiki.archlinux.org.",
}


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with open(p, "rb") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    rows, total_docs, total_bytes, total_disk = [], 0, 0, 0
    lines = ["# kural pretraining corpus", "",
             f"Built {time.strftime('%Y-%m-%d')} by `data_tools/` (see data_tools/README.md to rebuild).",
             "Token estimates are `UTF-8 bytes / 4` (no tokenizer). Each shard line is "
             "`{\"text\", \"source\", \"meta\"}`; shards are `data/clean/<source>/shard-NNNNN.jsonl.zst`.", "",
             "## Summary", "",
             "| slot | source | documents | text GB | est. tokens (B) | disk GB (zst) |",
             "|---|---|---:|---:|---:|---:|"]
    details = []
    for slot, sources in GROUPS:
        for s in sources:
            sp = CLEAN / s / "stats.json"
            if not sp.exists():
                lines.append(f"| {slot} | {s} | not built | | | |")
                continue
            st = json.loads(sp.read_text())
            disk = dir_size(CLEAN / s)
            total_docs += st["documents"]
            total_bytes += st["text_bytes"]
            total_disk += disk
            lines.append(f"| {slot} | {s} | {st['documents']:,} | {st['text_bytes'] / 1e9:.2f} | "
                         f"{st['est_tokens'] / 1e9:.2f} | {disk / 1e9:.2f} |")
            drops = ", ".join(f"{k} {v:,}" for k, v in sorted(st["drops"].items(), key=lambda kv: -kv[1]))
            details += [f"### {s}", "",
                        f"- Origin: {st.get('origin', '?')}",
                        f"- Licence: {st.get('licence', '?')}",
                        f"- {NOTES.get(s, '')}",
                        f"- Documents {st['documents']:,}; text {st['text_bytes'] / 1e9:.3f} GB; "
                        f"~{st['est_tokens'] / 1e9:.3f}B tokens; {st['shards']} shards; built {st['finished']}.",
                        f"- Input documents (after source-specific parsing) {st['input_docs']:,}. Dropped: {drops}.",
                        ""]
    lines.append(f"| | **total** | **{total_docs:,}** | **{total_bytes / 1e9:.2f}** | "
                 f"**{total_bytes / 4e9:.2f}** | **{total_disk / 1e9:.2f}** |")
    lines += ["", "## Filters applied to every source", "",
              "- Unicode NFC, control characters removed, trailing spaces trimmed, runs of blank lines capped. "
              "Spelling, casing, slang and typos are left alone on purpose.",
              "- English language ID with fastText `lid.176.bin` (prose sources; on technical posts the code "
              "blocks are removed before ID). Thresholds: Reddit general 0.7, Reddit tech 0.6, web 0.65, "
              "Stack Exchange 0.5, math 0.4-0.5. Wikipedia, docs and code are not language-filtered.",
              "- Prose quality (Gopher-style, lenient): minimum words, alphabetic fraction, symbol ratio, mean "
              "word length 2.5-12, duplicated-line fraction, most-common-line fraction, runs of 40+ identical "
              "characters, one-word spam. Spam phrases and link dumps. FineWeb/FineMath: spun-text check.",
              "- Exact dedup (xxh64 of the lowercased, punctuation-free text) and near dedup (MinHash, 5-word "
              "shingles, 112 permutations in 14 bands of 8, roughly Jaccard >= 0.75) within each source; the "
              "first occurrence (in task order) is kept.",
              "- Decontamination: a document is dropped if it contains, case- and whitespace-insensitively and "
              "on word boundaries, any line of `data/eval/decontam_patterns.txt` (all 12,607 NL2Bash requests, "
              "InterCode-Bash queries, the 300 NL2SH-ALFA test requests, and kural's tests/*.txt requests). "
              "Request lines shorter than 4 words and 20 characters are not used as patterns "
              "(listed in `data/eval/decontam_skipped_short.txt`): phrases like \"git status\" are everywhere.",
              "- PII: e-mail addresses and phone numbers replaced in Reddit and Stack Exchange text; Reddit "
              "u/usernames replaced. (FineWeb-Edu/FineMath were anonymised upstream.)", "",
              "## Sources", ""] + details
    pairs = PAIRS / "stackexchange_pairs.jsonl"
    lines += ["## Task data and evaluation sets (not in the pretraining text)", "",
              f"- `data/pairs/stackexchange_pairs.jsonl`: {count_lines(pairs):,} shell-relevant questions "
              "(unix, askubuntu, superuser, serverfault, Stack Overflow with a TECH_TAG) with title, body, site, "
              "id, url, score, tags, licence and the raw `<pre>` code blocks of the accepted/top answers. "
              "CC BY-SA: keep `url` for attribution.",
              f"- `data/pairs/heldout.jsonl`: {count_lines(PAIRS / 'heldout.jsonl'):,} questions "
              "(xxh64(\"site:id\") % 100 < 2, over *all* questions of every site). Never train on these; they "
              "are also excluded from the pretraining text.",
              f"- `data/pairs/nl2sh_alfa_train.jsonl`: {count_lines(PAIRS / 'nl2sh_alfa_train.jsonl'):,} "
              "NL2SH-ALFA training pairs (MIT, Westenfelder et al., NAACL 2025), **unverified** "
              "(`verified: false`); 5 whose request equals a test request were removed. **7,153 rows are NL2Bash "
              "requests (`in_nl2bash: true`, 492 from the NL2Bash test split): drop them before training if "
              "NL2Bash is used as a test set.**",
              "- `data/eval/nl2sh_alfa/test.jsonl`: NL2SH-ALFA test, 300 requests x 2 reference commands "
              "(600 verified pairs), MIT.",
              "- `data/eval/nl2bash/`: NL2Bash (Lin et al., LREC 2018), all.nl/all.cm and the standard "
              "train/dev/test split; GPL-3.0 (see LICENSE there).",
              "- `data/eval/intercode_bash/`: InterCode-Bash queries (Yang et al., 2023), MIT.", "",
              "## Licence summary", "",
              "- CC BY-SA (Wikipedia, Stack Exchange, ArchWiki is GFDL): attribution + share-alike apply to "
              "redistributed text; a trained model is generally not considered a derivative copy, but do not "
              "publish the corpus or verbatim regurgitations without attribution.",
              "- CC BY 4.0 (tldr-pages): credit \"tldr-pages contributors\" wherever the data or a dataset "
              "derived from it is shared.",
              "- ODC-By (FineWeb-Edu, FineMath): attribution to Hugging Face / the dataset; Common Crawl terms.",
              "- Reddit: no open licence; Reddit's user agreement and API terms govern the content. Fine for "
              "local training; do not redistribute the text.",
              "- Code: permissive licences only (per-file `meta.license`); keep licence headers when "
              "redistributing.",
              "- GNU manuals: GFDL. Man pages: per page. NL2Bash is GPL-3.0 (eval only, not in training).", ""]
    du = subprocess.run(["du", "-sb", str(DATA)], capture_output=True, text=True).stdout.split()[0]
    lines.append(f"Total size of data/ when this file was written: {int(du) / 1e9:.1f} GB.")
    (DATA / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {DATA / 'MANIFEST.md'}: {total_docs:,} docs, {total_bytes / 1e9:.1f} GB text, "
          f"~{total_bytes / 4e9:.1f}B tokens")


if __name__ == "__main__":
    main()
