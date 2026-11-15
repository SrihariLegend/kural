"""Fetch evaluation sets into data/eval/ and build the decontamination pattern list.

Outputs
  data/eval/nl2bash/{all.nl,all.cm,LICENSE}, data/eval/nl2bash/split/{train,dev,test}.json
  data/eval/intercode_bash/*.json
  data/eval/nl2sh_alfa/test.jsonl              (300 requests x 2 reference commands = 600 pairs)
  data/eval/decontam_patterns.txt              (one request per line, used by common.Decontaminator)
  data/pairs/nl2sh_alfa_train.jsonl            (40,639 *unverified* pairs; training use is allowed)

Patterns: every NL2Bash request (all 12,607, not only the test split, since the
whole set is used as an untuned test in docs/ARCHITECTURE.md section 6), every
InterCode-Bash query, every NL2SH-ALFA test request, and every request line in
kural's tests/*.txt. Lines shorter than 4 words and 20 characters ("git status",
"tell me") are skipped: they are everyday phrases, and matching them would remove a
large share of all text without protecting anything.

Usage: .venv/bin/python -I data_tools/eval_sets.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL, PAIRS, RAW, ROOT, log  # noqa: E402

NL2BASH_RAW = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master"
NL2BASH_SPLIT = "https://www.dropbox.com/s/wy7uahzbir7lrq1/nl2bash.zip?dl=1"  # from jiacheng-ye/nl2bash
INTERCODE_RAW = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/nl2bash"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sfL", "-m", "300", "-o", str(dest), url], check=True)


def keep_pattern(s: str) -> bool:
    s = " ".join(s.split())
    return len(s.split()) >= 4 or len(s) >= 20


def kural_test_requests() -> list[str]:
    out = []
    for p in sorted((ROOT / "tests").glob("*.txt")):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("! "):
                continue
            if line.startswith("+ "):
                line = line[2:]
            req = line.split("=>")[0].strip()
            if req:
                out.append(req)
    return out


def main() -> None:
    nb = EVAL / "nl2bash"
    for f in ("data/bash/all.nl", "data/bash/all.cm", "LICENSE"):
        fetch(f"{NL2BASH_RAW}/{f}", nb / Path(f).name)
    split = nb / "split"
    if not (split / "test.json").exists():
        z = nb / "nl2bash_split.zip"
        fetch(NL2BASH_SPLIT, z)
        split.mkdir(exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(z), "-C", str(split), "--strip-components", "1"], check=True)
        z.unlink()
    ic = EVAL / "intercode_bash"
    for f in ("nl2bash_fs_1.json", "nl2bash_fs_2.json", "nl2bash_fs_3.json", "nl2bash_fs_4.json",
              "test_queries.json", "README.md"):
        fetch(f"{INTERCODE_RAW}/{f}", ic / f)

    alfa_raw = RAW / "nl2sh_alfa"  # downloaded by fetch_hf.sh (westenfelder/NL2SH-ALFA, MIT)
    alfa = EVAL / "nl2sh_alfa"
    alfa.mkdir(parents=True, exist_ok=True)
    test_rows = list(csv.DictReader(open(alfa_raw / "test.csv", newline="")))
    with open(alfa / "test.jsonl", "w") as fh:
        for i, r in enumerate(test_rows):
            fh.write(json.dumps({"id": i, "nl": r["nl"], "bash": r["bash"], "bash2": r.get("bash2"),
                                 "difficulty": r.get("difficulty")}) + "\n")
    (alfa / "README.md").write_text((alfa_raw / "README.md").read_text())
    PAIRS.mkdir(parents=True, exist_ok=True)
    train_rows = list(csv.DictReader(open(alfa_raw / "train.csv", newline="")))
    test_nl = {" ".join(r["nl"].lower().split()) for r in test_rows}
    n_overlap = 0
    with open(PAIRS / "nl2sh_alfa_train.jsonl", "w") as fh:
        for i, r in enumerate(train_rows):
            if " ".join(r["nl"].lower().split()) in test_nl:
                n_overlap += 1
                continue
            fh.write(json.dumps({"id": i, "nl": r["nl"], "bash": r["bash"], "verified": False,
                                 "source": "westenfelder/NL2SH-ALFA train (MIT)"}) + "\n")
    log(f"NL2SH-ALFA: {len(test_rows)} test requests, {len(train_rows) - n_overlap} train pairs "
        f"({n_overlap} dropped: same request as a test item)")

    pats: dict[str, str] = {}

    def add(s: str, origin: str) -> None:
        s = " ".join(s.split())
        if s and keep_pattern(s):
            pats.setdefault(s.lower(), origin)

    for line in (nb / "all.nl").read_text(errors="replace").splitlines():
        add(line, "nl2bash")
    for f in sorted(ic.glob("*.json")):
        for item in json.loads(f.read_text()):
            add(item.get("query", ""), "intercode")
    for r in test_rows:
        add(r["nl"], "nl2sh_alfa_test")
    kt = kural_test_requests()
    for r in kt:
        add(r, "kural_tests")
    skipped = sorted({" ".join(r.split()) for r in kt if not keep_pattern(r)})
    (EVAL / "decontam_patterns.txt").write_text("\n".join(sorted(pats)) + "\n")
    (EVAL / "decontam_skipped_short.txt").write_text("\n".join(skipped) + "\n")
    by = {}
    for o in pats.values():
        by[o] = by.get(o, 0) + 1
    log(f"decontam patterns: {len(pats)} {by}; {len(skipped)} short kural lines skipped")


if __name__ == "__main__":
    main()
