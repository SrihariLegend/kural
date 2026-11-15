# kural-gen

The training-data generator for kural, in Rust: random valid programs paired with varied English requests,
including follow-ups that build on a previous request. It is a port of `kural/gen.py` and `kural/mix.py`
(which stay as the reference) and reads the language from the same files, `kural/lang/stages.txt` and
`kural/lang/commands.txt`, so the two can never disagree about stages, slots or token numbers.

```
cargo build --release -p kural-gen

kural-gen data --n 1600000 --seed 1 --out data/train --shard-size 200000   # training shards, all cores
kural-gen data --n 4000 --seed 9999 --out data/held                        # a held-out set (different seed)
kural-gen show --n 20 [--follow]                                           # eyeball examples and their programs
kural-gen lexicon --n 400000 --out kural/lexicon.json                      # the vocabulary-check table
kural-gen check --n 1000000                                                # count generator errors, write nothing

python -m kural.train --data data/train --held data/held --epochs 12       # train by streaming the shards
```

Each shard is generated from its own seed (derived from `--seed` and the shard number), so the output does not
depend on how many threads ran.

## Shard format

One self-contained file per shard (`shard-000000.kds`, ...), little-endian, described in full at the top of
`src/shard.rs`. In short: a 128-byte header, then a per-shard table of the distinct words with their features
already computed (hashed trigrams, hashed word, shape bits, and the word text), then for every example its word
indices, how many of them are the request (the rest is session context), and its output tokens.

Storing each distinct word once makes an example about 80 bytes, so a billion examples is roughly 80 GB.
Training (`kural/train.py --data`) keeps one shard on the GPU at a time and gathers each batch there, so memory
does not grow with the number of examples. `kural/shards.py` reads shards from Python.

## Checked against the Python generator (2026-11-15)

- 0 generator errors in 1,000,000 examples.
- Speed: about 270,000 examples/s on one core (the Python generator: about 25,000/s), 1.7 to 4 million/s on all cores.
- 50,000 examples: every program is valid under the grammar and compiles to bash with `kural/compile.py`
  (3% are deliberate "unknown" declines); word features equal `kural/feat.py` on 20,000 distinct words
  (`tests/check_rustgen.py`).
- Distribution, 200,000 examples each: the share of examples using each stage differs from Python by at most
  0.3 points, and lengths, context share and multi-step share match to the second decimal.
- A streamed training batch is identical to the in-memory batch for the same examples.
