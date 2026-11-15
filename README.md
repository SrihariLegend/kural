# kural 🪸

Plain English to bash, fully local. A 7 MB parser that runs on the CPU with NumPy only.

```
$ : delete log files older than 30 days in ~/Downloads        the line turns violet as soon as you type ":"
  → find ~/Downloads -type f -iname '*.log' -mtime +29 -delete     shown live underneath while you type
    files in ~/Downloads > ext log > older 30 days > DELETE   99%

(Enter)
  files in ~/Downloads > ext log > older 30 days > DELETE  (99% sure)
  this removes or stops things: read it before pressing Enter
$ find ~/Downloads -type f -iname '*.log' -mtime +29 -delete   now on your prompt: edit it, run it, or Ctrl-C
```
## Use

Add one line to `~/.bashrc`, then open a new terminal:

```
source /path/to/kural/kural/shell/kural.bash
```

| Key | What it does |
|---|---|
| `:` on an empty prompt | starts a request; type plain English after it |
| Enter | puts the command on your prompt. Nothing is run until you press Enter again |
| Tab | next reading, when there is more than one |
| Up / Down | earlier requests of this terminal |
| Esc, Ctrl-C, Backspace over the `:` | leave |

A colon anywhere else is an ordinary colon, and `: > file` still works as bash.

Requests can build on the one before:

```
: find pdf files bigger than 10mb in ~/docs        : is the nginx service running      $ make test
: only the ones from this week                      : restart it                        : again with sudo
: how much space do they take                       : show its logs                     : save that to out.txt
: move them to /mnt/usb
```

Without the hook: `kural "show the 5 biggest files in /var/log"` prints the command, `kural -v ...` also prints the
reading, confidence and alternatives.

Settings: `KURAL_COLOR` (SGR numbers, default `38;5;141`), `KURAL_PREVIEW=0` (no live preview). Say
`: use the trash instead of deleting`, `: ask me before deleting` or `: be verbose` to change how commands are built
for the session; put `trash=on` etc. in `~/.config/kural/config` to keep it.

## What it covers

290 building blocks (`kural/ir.py`, `kural/commands.py`):

- **Files**: find by type, name, size, age, date, owner, content, hidden, executable, and "not" any of these; then
  list, count, total size, delete, trash, move, copy, zip, search inside, replace inside, open, edit, permissions,
  owner, rename in bulk (case, spaces, prefix, suffix, extension, text), checksum, duplicates, upload.
- **Text**: lines of a file, of your history, environment, installed packages, running services, the journal, the
  kernel log, the clipboard, or the previous command's output; filter, sort, dedupe, columns, line ranges, replace,
  frequency, sums, counts; save, append, or write back in place.
- **Processes and ports**: by name, user, memory, CPU; what is on a port; kill.
- **Stand-alone commands** (192): packages, services, power, sound, screen, network, docker, git, python and node
  projects, archives, environment variables, single-file operations.

Still outside: converting media, users and groups, mounting drives, firewalls, writing scripts, explaining errors,
anything that needs a loop or an "if". It does not speak zsh or fish yet.

## How it works

The model never writes bash. It does three small things, and ordinary code does the rest.

1. **Parse to a typed program.** A request becomes one to three pipelines of stages
   (`files > ext > older > delete`) from a small typed language.
2. **Copy, don't generate.** Every path, name, size, number and search text in the program is a pointer to words
   the user typed (or to the previous request). The model cannot invent a file name or a flag.
3. **Grammar-constrained decoding.** At each step only tokens that keep the program valid are allowed, so every
   output is a well-formed program.

Around the model, all deterministic:

- **Compiler** (`compile.py`): program to bash, with NUL-separated file pipelines and proper quoting.
  `tests/run_commands.py` runs 51 generated commands for real, on file names with spaces, quotes and leading dashes.
- **Values** (`values.py`): what "2 gigs", "last week", "read only" or "15 percent of 240" means.
- **Machine facts** (`ground.py`): `readme` becomes `README.md` if that is the only match, "run the tests" becomes
  `cargo test` in a Rust project, "stop nginx" becomes `systemctl stop` when nginx is a running service. Commands
  that change things take names exactly as typed.
- **Session context** (`session.py`): one throwaway text file per terminal, in the runtime folder (RAM), holding
  the commands you ran, your requests, and what you ran instead of an offer. The previous request is fed to the
  parser as a program, so a follow-up is an edit of that program. If you correct an offer and ask the same thing
  again, you get your corrected command.
- **Three guards**:
  - Nothing is run for you.
  - A delete, kill, overwrite or power-off is only accepted if the request (or the request it follows) contains a
    word asking for it. A question cannot become a delete.
  - A reading is declined if none of the request's own words has ever been seen with it in training
    (`lexicon.py`), and shown as "not sure" if most of them have not.
- **Answering process** (`serve.py`): one per user, started on demand, exits after ten idle minutes. It keeps the
  model loaded so the live preview costs about 10 ms per pause in typing instead of 110 ms of Python start-up.
  It listens on a pipe in your runtime folder, not on the network.

The model is trained only on synthetic data: `gen.py` and `mix.py` sample valid programs and write varied English
for each, including follow-ups. No larger model is involved at run time.

## Numbers (2026-11-15, this laptop)

| | |
|---|---|
| Parameters / file size | 3.58 M / 6.7 MB (float16), plus an 83 KB word table |
| Run-time dependencies | Python, NumPy |
| Live preview / Enter, with the answering process running | about 13 ms per request |
| One-off call from the shell (`kural "..."`) | about 100 ms, mostly Python start-up |
| Peak memory | about 48 MB |
| Training | 1.6 M synthetic requests, 12 epochs, about 20 minutes on one laptop GPU |

Accuracy (right = the top answer compiles to exactly the expected command):

| Set | Requests | Right | Note |
|---|---|---|---|
| `tests/v2_final.txt` | 188 | 98.4% (follow-ups 42/42) | **scored once, never used for tuning**; right answer in the top 3 for all 188 |
| `tests/v2_dev.txt` | 261 | 98.9% | used to find gaps in the training data |
| `tests/once_outside.txt` | 39 | 92.3% | written for 0.1 as things it could not do; 3 phrasings were later added to training |
| `tests/natural_*.txt` (0.1 abilities) | 323 | 97.9-98.6% | no regression from 0.1 |
| `tests/outside_v2.txt` | 40 | 100% declined | **not a fair number**: some of its phrasings were added to the decline examples after a first run declined 36/40 |
| held-out synthetic | 4,000 | 99.7% | same distribution as training |
| `tests/run_commands.py` | 51 | 51 | generated commands executed on awkward file names |
| `tests/term.py` | 18 checks | all pass | the bash hook driven in a pseudo-terminal |

Caveats that matter:

- Every test set was written by the same author as the training phrasings. A stranger's wording will do worse
  until their misses are fed back into the generator.
- At 95% confidence or more, no wrong answer got through on the final set; below 75% the hook says "not sure".
  Confidence is still the model's own probability, not a calibrated one.
- Out-of-scope requests that resemble something it knows can still get a wrong reading. The reading line under the
  command is there to catch that.

## Develop

```
python -m kural.mix 0 20                              # 20 sample requests with their commands
python -m kural.mix 0 20 follow                       # follow-ups only
python -m kural.train --n 1600000 --epochs 12         # needs torch; writes kural/weights.npz
python -m kural.lexicon                               # rebuild the word table after changing the generator
python tests/evaluate.py tests/v2_dev.txt -v            # score, listing the misses
python tests/run_commands.py                            # execute generated commands in a scratch folder
python tests/term.py                                    # drive the bash hook in a pseudo-terminal (needs pyte)
```

To teach it something new: a stand-alone command is a few lines in `commands.py` (arguments, bash, phrasings).
A new pipeline stage goes in `ir.py`, `compile.py` and the generator. Then retrain and rebuild the word table.
