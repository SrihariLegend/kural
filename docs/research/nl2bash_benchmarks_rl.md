# NL-to-Bash benchmarks, execution-based evaluation, execution-guided decoding and execution RL (for kural / Reef)

Scope note: about 18 tool calls. Primary sources read in full or in part: the NLC2CMD report (arXiv 2103.02523 HTML), the InterCode paper (arXiv 2306.14898 HTML), Westenfelder et al. NAACL 2025 (the full PDF text), Vo et al. 2024 (arXiv 2405.06807, the full PDF text), the NL2Bash GitHub README, and the abstracts of RLEF, unix-ctf and execution-guided decoding. Anything marked **[unverified]** comes from model memory or from secondary snippets and was not checked against a primary source in this session.

## Q1. Which datasets can be used for training (licences), and which should be kept for evaluation?

### Takeaway
NL2Bash (9,305 pairs, data under MIT) and tldr-pages (CC BY 4.0) can legally be used for training. But NL2Bash is noisy and is the parent of almost every test set (NLC2CMD, InterCode, NL2SH-ALFA test), so a model trained on it cannot be scored fairly on any of them. The cleanest held-out execution benchmark is NL2SH-ALFA / InterCode-ALFA: a 600-pair test set, manually verified, under MIT. Train on the NL2SH-ALFA *training* split (40,939 pairs, MIT, unverified) or on NL2Bash only after deduplicating against every test set, and keep a test set of your own, written by other people.

### Cited findings
**NL2Bash (Lin et al., LREC 2018)**
- About 10k bash one-liners scraped from StackOverflow and similar sites, each paired with an English description. They cover 102 unique utilities and 206 unique flags. The splits are train 8,090, dev 609, test 606 pairs (9,305 in total), and "neither a natural language description nor a Bash command appears in more than one split" — [NL2Bash repo](https://github.com/TellinaTool/nl2bash)
- Licence: the code is GPL-3.0. The data (`data/bash`) has been separately licensed under MIT since 24 April 2020 — [NL2Bash repo](https://github.com/TellinaTool/nl2bash). (The InterCode paper still calls NL2Bash "GPLv3" — [InterCode](https://arxiv.org/html/2306.14898v3). The repo's data licence is the current one.)
- The 9,305-pair count is confirmed independently — [Westenfelder et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)

**NLC2CMD (NeurIPS 2020 competition)**
- Evaluation data: 1,000 samples drawn from a filtered NL2Bash in phase 1, growing to more than 1,800 in the test phase. The additions were about 700 filtered Tellina user-query-log utterances (from nearly 1,000, annotated by three programmers) and about 100 crowd-submitted examples (from more than 120). Utilities went from 101 to 128 — [NLC2CMD report](https://arxiv.org/html/2103.02523v2)
- Filtering: a command was kept only if parsing it to an AST and reconstructing it gave back the original exactly, and only if all its utilities were in the Ubuntu 18.04 set — [NLC2CMD report](https://arxiv.org/html/2103.02523v2)
- The report itself flags the overlap: the phase-1 test set was sampled from the public NL2Bash, "so it may overlap with public training data", which could inflate scores — [NLC2CMD report](https://arxiv.org/html/2103.02523v2)
- The report states no licence for the competition data — [NLC2CMD report](https://arxiv.org/html/2103.02523v2)

**InterCode-Bash (Yang et al., NeurIPS 2023 D&B)**
- 200 tasks in 4 disjoint sets: 3 grounded to custom Docker file systems and 1 file-system agnostic. They were chosen from 1,000 NL2Bash commands with at least 4 utilities each, removing ssh, sudo, time and GUI tools, adding concrete paths, and updating deprecated flags — [InterCode](https://arxiv.org/html/2306.14898v3)
- Conflict: Westenfelder et al. count 224 commands in the InterCode Bash dataset — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf). The likely cause is the released repo against the paper's count **[unverified]**.

**NL2SH-ALFA / InterCode-ALFA (Westenfelder et al., NAACL 2025)**
- Test set: 600 instruction-command pairs, manually verified, with Docker environments. It was built from the 111 valid InterCode pairs, 82 corrected ones and 117+ new verified ones. Training set: 40,939 pairs from NL2Bash, tldr-pages and other sources, deduplicated, filtered with the bashlex parser, and de-conflicted against the test set. Both are released under **MIT** (GitHub westenfelder/NL2SH and HuggingFace) — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)
- The authors say the training set "is too large for manual verification, and we are unable to guarantee its correctness". The data is English only and one-line commands only. Building and verifying the test set took more than 100 hours — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)

**tldr-pages**
- The page content is CC BY 4.0 and the scripts are MIT, per the project's README and licence badge. Wikipedia's infobox says CC BY-SA 4.0, which conflicts with the repo — [search summary of tldr-pages repo / tldr.sh](https://tldr.sh/); [Wikipedia](https://en.wikipedia.org/wiki/TLDR_Pages)

**Vo et al. 2024 (IBM)**: 125 hand-written test cases in three suites: single-line Bash, multi-line Bash (50 tasks, of which 9 are sysadmin and 41 are math) and PowerShell. They were written with SREs and run in rootless podman — [arXiv 2405.06807](https://arxiv.org/pdf/2405.06807). Licence and release status not found.

**Other datasets named in the brief**
- NL2CMD (Fu et al. 2023, synthetic data generated from command grammars plus NL paraphrases) **[unverified, not fetched]**
- AnishJoshi/nl2bash-custom on HuggingFace (it exists; licence not checked) — [HF](https://huggingface.co/datasets/AnishJoshi/nl2bash-custom/commit/2f8d1956aa6516579032474e533b0f42969ae571)
- Jacobs & Lapon (LAST-X 2026): local open models on NL2Bash with 10 prompting techniques; Qwen3 strong; role-play prompting helps — [NDSS LAST-X 2026](https://www.ndss-symposium.org/wp-content/uploads/lastx2026-49.pdf) (search snippet only)
- No benchmark named "ShellBench" was found. Terminal-Bench (agentic, multi-step) exists **[unverified, not fetched]**.

### Inferences (verdicts)
- **NL2Bash train split: TEST, for training only.** The licence is fine (MIT). The data has heavy noise (see Q-quality below), and it is the parent of nearly every public test set. Before training, dedupe against NL2SH-ALFA test, InterCode, and NLC2CMD by normalised command *and* by fuzzy NL match.
- **NL2Bash test split as an evaluation set: SKIP as the headline metric.** It is not execution-ready (no environments), it is noisy, and exact-match or BLEU style scoring is meaningless for Reef. Use it at most as a secondary "template accuracy" sanity number for comparison with older papers.
- **NL2SH-ALFA test (600) / InterCode-ALFA: ADOPT as the public held-out execution benchmark.** It is under MIT and Docker-based, it is the most recent verified set, and it has published 0.5B to 8B and GPT-4o numbers to compare against. Never train on it. Make sure the NL2SH-ALFA training set (which includes NL2Bash and tldr) is de-conflicted from it (the authors did this by exact match only).
- **NL2SH-ALFA train (40,939): TEST for training.** It is the largest permissive set. Spot-check about 200 pairs by hand before trusting it.
- **tldr-pages: ADOPT for the local index and for training pairs** (CC BY 4.0 needs attribution; ship a notice). Because tldr is inside NL2SH-ALFA train, tldr-derived phrasing must not leak into the people-written test set.
- **InterCode-Bash original (200/224): SKIP as-is.** About half of it is erroneous (Q-quality). Use the ALFA-corrected version instead.
- **NLC2CMD data: SKIP for evaluation.** It overlaps with NL2Bash, has no licence stated, and uses a metric that ignores arguments. The Tellina query-log portion is interesting as "real user phrasing", but its availability and licence are unclear.
- **Vo et al. 125 cases: TEST as an extra held-out set if it has been released.** It covers sysadmin and side-effect tasks well.
- **kural's own people-written test set: ADOPT.** This is still required. Every public set inherits NL2Bash's StackOverflow distribution and English phrasing, which is not what kural users type.

### Gaps
- NLC2CMD's licence, and whether its final test set (with the Tellina log) was ever released publicly.
- The exact licence and release of the Vo et al. suites and of the NL2CMD (Fu et al. 2023) synthetic data.
- The source of the InterCode 200 vs 224 task-count discrepancy.

## Q2. What metrics measure execution correctness for shell commands with side effects?

### Takeaway
Use state-based scoring: the output (stdout) plus a diff of the file system (which paths changed, plus content hashes) in a fresh container, compared with what the reference command does in an identical container. InterCode defines the standard continuous form. The NAACL 2025 work shows that comparing stdout by TF-IDF or embeddings, or using an LLM without execution, both misjudge equivalence; an LLM judging *execution outputs* reached 95% agreement with humans. For a model that must be deterministic and local, a hand-written per-task checker (like Vo et al.'s epilogue scripts) plus a file-system diff is the most trustworthy design.

### Cited findings
- **NLC2CMD metric (not execution-based).** Per utility, it scores flag overlap and penalises extra flags (normalised by the larger flag set). A wrong utility costs a penalty of equal size. Each prediction is weighted by its confidence δ and divided by the larger utility count. The score is the maximum over references. Over the top 5 predictions it takes the maximum if any is above 0, otherwise the mean. **Arguments are ignored**, and the order of utilities in a pipe matters. Energy (Joules, measured with experiment-impact-tracker) was a separate track — [NLC2CMD report](https://arxiv.org/html/2103.02523v2)
- The NLC2CMD report notes that equivalence is undecidable (Bash is Turing-complete) and that BLEU correlates weakly with semantic correctness — [NLC2CMD report](https://arxiv.org/html/2103.02523v2)
- **InterCode-Bash reward:** 0.34 × TF-IDF cosine of the output against the gold output, plus 0.33 × (1 − erf(|symmetric difference of the changed-file lists|)), plus 0.33 × the fraction of commonly changed paths whose md5sums match. Changes are tracked with `git status -s`, and the environment is reset with `git reset --hard` and `git clean -fd`. Success means a reward of 1 — [InterCode](https://arxiv.org/html/2306.14898v3)
- **The InterCode reset is buggy:** Westenfelder et al. "identify and fix an error in the benchmark's Docker reset script that causes the filesystem structure of the two execution environments to diverge" — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)
- **Functional-equivalence heuristics compared** on a 600-pair labelled set. Precision / recall / F1 / accuracy:

  | Heuristic | Precision | Recall | F1 | Accuracy |
  |---|---|---|---|---|
  | llama-3.1-8b judge, no execution | 1.00 | 0.05 | 0.10 | 0.53 |
  | exec + llama-3.1-8b | 0.88 | 0.74 | 0.80 | 0.82 |
  | gpt-4-0613, no execution | 1.00 | 0.51 | 0.68 | 0.76 |
  | exec + gpt-4-0613 | 0.99 | 0.91 | 0.95 | 0.95 |

  The BLEU and nl2cmd heuristics had low recall: they "cannot identify cases where syntactically different commands are functionally equivalent" (awk vs sed). Exec + mxbai-embed (cosine ≥ 0.75 on stdout) was used to score the model tables — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)
- Tasks that are hard to compare by execution: time and date output in different formats, line-numbered text, and non-deterministic or network utilities — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)
- **Vo et al. design:** a per-test `prologue.sh` that sets up state, the model's command run in rootless podman (`--userns`, a mounted home, `--timeout`), and a per-test `epilogue.sh` that checks side effects. A generic epilogue diffs system files and fails a test that "created spurious system files". Documented false verdicts: a correct `date -d "+90 days"` answer marked FAIL because of output-format differences; a command with a superfluous `-exec echo x` that should still pass — [Vo et al.](https://arxiv.org/pdf/2405.06807)

### Inferences
- For Reef's held-out score, define per task: (a) the initial state (a seeded file tree, mtimes and sizes for "older than a week" or "big"); (b) the expected state delta, as an exact set of created, deleted and modified paths with hashes; (c) an expected-stdout predicate (a set of lines, ignoring order where the request is unordered). Score **strict execution accuracy** = (state delta exactly equal) AND (stdout predicate holds). Also report InterCode's continuous reward, for comparability.
- Penalise *extra* changes asymmetrically (deleting more than asked is worse than deleting less). InterCode's symmetric-difference term treats both the same. The ARCHITECTURE.md "penalty for doing more than asked" is the right instinct.
- Run each task under at least two seeded file-system variants, so that a command which happens to work on one tree (hard-coded names, `head -n 3` that matches by luck) is caught. InterCode's robustness check across file systems 2–4 shows scores move by about 4 points.
- Use an LLM judge on outputs only as an offline audit of the checker, never in the RL reward (it is stochastic and slow, and it can be gamed).

### Gaps
- No primary source in this session gave a standard for scoring commands with network, service or sudo effects. All benchmarks simply exclude them (InterCode removed ssh and sudo).

## Q3. Strongest small-model results, and how 7B–14B LLMs score

### Takeaway
In the competition setting, small specialised seq2seq models were competitive: NLC2CMD was won by an ensemble of 5 six-layer transformers (likely tens of millions of parameters in total, [unverified]) at 0.532 on a metric that ignores arguments, and TF-IDF retrieval alone reached 0.43–0.47. On execution-verified sets, general LLMs dominate the published tables. GPT-4o reaches 0.74 on NL2SH-ALFA, Qwen2.5-Coder-7B 0.61–0.62 and Llama-3.1-8B 0.46–0.56. Sub-1B general models reach only 0.10–0.37, mostly because of output-format failures. There is **no published execution-based result for a model under 100M parameters**; the "small beats 8B" claim is untested in the literature.

### Cited findings
**NL2Bash original (manual evaluation, test set)**
- Sub-token CopyNet: full-command accuracy top-1 0.36, top-3 0.45; template accuracy top-1 0.49.
- Tellina: full-command top-1 0.27, template top-1 0.53. BLEU top-1 was 50.9 and 48.6 respectively — [NL2Bash repo](https://github.com/TellinaTool/nl2bash)

**NLC2CMD final leaderboard (NLC2CMD metric)**

| Team | Score | Energy | Latency | Model |
|---|---|---|---|---|
| Magnum | 0.532 | 0.484 J | 0.709 s | ensemble of 5 six-layer transformers with beam search |
| Hubris | 0.513 | 12.0 J | – | fine-tuned GPT-2, two models ensembled |
| Jb | 0.499 | – | – | utility classifier + seq2seq transformer |
| AICore | 0.489 | 0.252 J | – | not described |
| AINixCLAISimple | 0.429 | – | – | TF-IDF retrieval, no neural model (0.472 with tuning) |
| Tellina baseline | 0.138 | – | – | RNN |

Source: [NLC2CMD report](https://arxiv.org/html/2103.02523v2). The report gives no parameter counts.

**InterCode-Bash success rate, all sets**

| Model | Single turn | Try again (n=10) |
|---|---|---|
| gpt-4 | 34.0 | 48.5 |
| gpt-3.5-turbo | 34.5 | 46.5 |
| text-davinci-003 | 24.6 | 38.7 |
| Vicuna-13B | 16.0 | 24.5 |
| StarChat-16B | 17.7 | 23.7 |
| PaLM-2 bison | 17–18 | 19–22.5 |

Source: [InterCode](https://arxiv.org/html/2306.14898v3). These scores are on the original, about 50%-erroneous data.

**NL2SH-ALFA test, accuracy with the exec + mxbai-embed heuristic.** Columns: Base / constrained decoding (CD) / markdown parse / in-context learning (ICL) / LoRA fine-tune on the 40,939-pair set (IWL).

| Model | Base | CD | Parse | ICL | IWL |
|---|---|---|---|---|---|
| llama-3.2-1b | 0.12 | 0.19 | 0.32 | 0.34 | 0.37 |
| llama-3.2-3b | 0.17 | 0.39 | 0.49 | 0.47 | 0.47 |
| llama-3.1-8b | 0.46 | 0.51 | 0.53 | 0.56 | 0.40 |
| qwen2.5-coder-0.5b | 0.10 | 0.05 | 0.35 | 0.36 | 0.27 |
| qwen2.5-coder-1.5b | 0.21 | 0.06 | 0.50 | 0.44 | 0.19 |
| qwen2.5-coder-3b | 0.26 | 0.06 | 0.58 | 0.50 | 0.51 |
| qwen2.5-coder-7b | 0.61 | 0.08 | 0.62 | 0.62 | 0.51 |
| gpt-3.5-turbo | 0.58 | – | 0.67 | 0.69 | – |
| gpt-4o-mini | 0.71 | – | 0.72 | 0.71 | – |
| gpt-4o | 0.74 | – | 0.73 | 0.73 | – |
| gpt-4-0613 | 0.68 | – | 0.68 | 0.73 | – |

Source: [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf). The qwen-3b, qwen-7b, gpt-4o and gpt-4 rows had to be reconstructed from PDF columns that extracted out of order, so check them against the paper before quoting.

- Authors' reading of the table: "model size remains the dominant factor". Incorrect output format (markdown and the like) is "the dominant failure mode for small (less than 7b parameter) models". With fine-tuning, llama-3.2-3b matches the base llama-3.1-8b, and qwen-0.5b matches the base qwen-3b. Fine-tuning *hurt* the 7B/8B models, which the authors attribute to LoRA capacity limits. Constraining the first token to a Bash utility helps Llama but badly hurts Qwen — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)
- Why first-token constraining helps: the utility token in the first position has about 10⁻⁴ of the relative probability of the following tokens, and once the utility is right, the flags and arguments are usually right — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)
- A dangerous output was observed from Llama-3.1-8B: `find ... -delete -print; rm -f /dev/null` — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)

**Vo et al. (execution accuracy, zero-shot).** Single-line Bash: GPT-4o 84%, Granite-34b-code-base 76%, CodeLlama-34b 70%, Mixtral-8x7B 70%, Mistral-7B 62%. Multi-line Bash: GPT-4o 68%; the open models scored 12–32%. Few-shot prompting did not reliably help — [Vo et al.](https://arxiv.org/pdf/2405.06807)

### Inferences
- Evidence that a tiny specialised model can match an 8B–14B general model is **indirect**:
  - in NLC2CMD, small transformers and TF-IDF retrieval beat fine-tuned GPT-2 on cost and efficiency;
  - in NL2SH-ALFA, task fine-tuning plus output constraints let 0.5–3B models reach the base scores of models 2.7–6× larger;
  - the main failure of small general models is format or interface, which Reef removes by construction (typed program + grammar).
- None of this shows a model under 100M parameters reaching about 0.55–0.62 execution accuracy on NL2SH-ALFA. That is the number Reef must beat to support its claim, and the people-written kural set is where it should win (grounding to installed tools and the local file tree is something the LLM baselines do not get).
- Fair comparison protocol for local 8B/14B baselines (Qwen2.5-Coder-7B/14B, Qwen3-8B/14B, Llama-3.1-8B):
  - give them the *best* published wrapper: markdown parse plus about 10 retrieved few-shot examples (the paper found the optimum near 10) and, for Llama, first-token utility constraining;
  - score with the identical sandbox checker;
  - report latency, RAM, and energy per request in Joules, NLC2CMD-style;
  - also report a "baseline + kural grounding context" variant, so the comparison isolates the model rather than the grounding.
- The typed-program path will score well only on requests the program language covers. Report coverage separately, and score declines as failures (or as a separate "abstain" column), so declining cannot inflate accuracy.

### Gaps
- Parameter counts for the NLC2CMD winners (Magnum, AICore) were not in the report.
- No execution-based results were found for Qwen 14B-class models on NL2SH-ALFA. Jacobs & Lapon (LAST-X 2026) evaluate Qwen3 locally, but only the snippet was seen — [NDSS LAST-X 2026](https://www.ndss-symposium.org/wp-content/uploads/lastx2026-49.pdf)
- No published execution result for any model under 100M parameters on NL2Bash-derived sets.

## Q4. Execution-guided decoding and execution RL: reported gains and pitfalls

### Takeaway
Execution-guided decoding (prune beam candidates whose partial or complete program fails) is cheap, proven in semantic parsing (WikiSQL execution accuracy 83.8%, new state of the art in 2018), and fits Reef's typed programs well. Use it for read-only candidates; for mutating ones, run only complete programs on an overlay. Execution RL (RLEF, GRPO with test rewards, sandboxed shell RL) gives large gains for 8B+ models, but the published bash-RL work is all agentic and 8B-sized. For a few-million-parameter model, the reward-design and sandbox lessons transfer; the size of the gains does not.

### Cited findings
- **Execution-guided decoding (Wang et al. 2018):** it conditions on the execution of the partially generated program to "detect and exclude faulty programs". It works with any autoregressive decoder and was applied to 4 recurrent or template models on WikiSQL, ATIS and GeoQuery, improving all three; WikiSQL execution accuracy reached 83.8% (state of the art) — [arXiv 1807.03100](https://arxiv.org/abs/1807.03100). Per-dataset gains were not in the abstract.
- **RLEF (Gehring et al., ICML 2025):** multi-turn code generation where public-test feedback is fed back to the model and the reward is passing held-out tests. It reached state of the art on CodeContests with 8B and 70B Llama 3.1 while needing about 10× fewer samples, and the gains carried over to HumanEval+ and MBPP+. Ablations with random feedback show the model really uses the feedback — [PMLR](https://proceedings.mlr.press/v267/gehring25a.html); [arXiv](https://arxiv.org/pdf/2410.02089). The figure "70B 54.5% pass@1 vs AlphaCodium 29%" comes from a secondary summary **[unverified]**.
- **unix-ctf (2026):** procedurally generated Unix capture-the-flag tasks in fresh containers, with a bidirectional filter (656 of 750 generated variants are valid; 155 distinct techniques). GRPO + LoRA on Qwen3-8B raised held-out solve rate from 11.6% to 43.6%; InterCode-CTF reached 32/100. The abstract reports no reward-hacking observations — [arXiv 2605.29115](https://arxiv.org/abs/2605.29115)
- **Other sandboxed-shell RL work** (abstracts or snippets only):
  - LLM-in-Sandbox-RL (bash tool, final-output-correctness reward) — [arXiv 2601.16206](https://arxiv.org/pdf/2601.16206)
  - PIPer: Qwen3-8B, online RL with rule-based rewards for bash environment-setup scripts — [arXiv 2509.25455](https://arxiv.org/pdf/2509.25455)
  - SWE-MiniSandbox: container-free isolation with mount namespaces and chroot, for cheap RL rollouts — [arXiv 2602.11210](https://arxiv.org/pdf/2602.11210)
  - a transactional, fault-tolerant sandbox reporting 100% interception of high-risk commands and 100% rollback success, which treats a sandbox refusal as a negative reward — [arXiv 2512.12806](https://arxiv.org/html/2512.12806)
  - a 2026 reward-hacking benchmark that measures agents exploiting graders in bash and Python sandboxes — [arXiv 2605.02964](https://arxiv.org/html/2605.02964v1)
- **CodeRL** (Le et al., NeurIPS 2022): actor-critic over CodeT5 (about 770M) with unit-test rewards plus a critic, giving gains on APPS and MBPP. **GRPO with verifiable rewards** (DeepSeekMath / R1) is the current standard recipe **[both unverified; not fetched]**.
- **Sandbox-level pitfalls documented:**
  - InterCode's reset script let the two environments drift apart — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)
  - in Vo et al., files created by root inside the container caused uid/gid trouble outside it, and the authors gave the test user no sudo outside the container — [Vo et al.](https://arxiv.org/pdf/2405.06807)
  - the dangerous `rm -f /dev/null` output above "could have corrupted a user's system" if not containerised — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)

### Inferences (verdicts)
- **Execution-guided decoding in beam search: ADOPT, with a limited scope.**
  - For the typed-program path, prune at the *program-stage* level: a stage whose file glob matches nothing, a `find` predicate that yields an empty set when the user asked to find something, a tool or flag missing from the local index. These checks are pure and cost about milliseconds.
  - For the open-bash path, execute only complete candidates (partial bash is rarely executable), read-only in bwrap.
  - Do not prune on "empty output" when emptiness is a valid answer ("are there any core dumps?"). Make it a ranking feature, not a hard filter.
  - Cap at about 4–8 executed candidates per request to stay inside the latency budget.
- **Execution-based RL: TEST, after SFT has saturated.** Use GRPO-style group-relative advantages (no critic, so it suits a tiny model) over about 8 sampled programs per request. Recommended reward:
  - +1 only if the state delta equals the reference and the stdout predicate holds;
  - a partial reward from the InterCode-style terms for shaping in early training only;
  - an asymmetric large penalty for any extra deletion, kill, or write outside the target set;
  - a small penalty for program length, so the policy does not learn to pad.
- Reward-hacking risks to design out:
  - (1) commands that print the expected string instead of computing it. Use ≥2 randomised file trees per prompt, and keep stdout reference values hidden and different per seed;
  - (2) "do nothing" policies on tasks where the reference also leaves no state change. Include stdout checks and positive-delta tasks;
  - (3) exploiting the checker (writing to the checker's paths, touching mtimes). Run the checker outside the sandbox, on a snapshot;
  - (4) `exit 0` / `|| true` masking failures. Never use the exit code alone;
  - (5) timeouts and fork bombs. Apply rlimits and pids cgroups.
  Because Reef's output space is a typed program under a grammar, many of these exploits are impossible by construction; this is a real advantage over free-text RL.
- **Sandboxing guidance for bwrap:**
  - `--unshare-all` (network included), `--die-with-parent`, `--new-session`;
  - a read-only bind of the user tree plus an overlay or tmpfs upper layer for writes;
  - no `/dev` beyond a minimal `--dev`, `--proc` in a new pid namespace, and a seccomp filter;
  - rlimits on CPU time, file size and processes; a wall-clock kill;
  - never pass sudo, D-Bus, or the real `$XDG_RUNTIME_DIR`.
  For RL rollouts, use synthetic file trees, never the user's home. Use one fresh namespace per candidate rather than reset scripts: InterCode's reset bug shows reset-based reuse drifts. A ready-made reference for overlay-based "dry run then inspect" is `binpash/try`, which the NAACL paper cites; it was not examined here.
- The confidence head should be trained on sandbox outcomes from the *held-in* distribution only, and its calibration measured (ECE) on the people-written test set.

### Gaps
- No primary source on execution RL for models under 100M parameters, or for single-step NL-to-command (not agentic) tasks.
- Primary numbers for CodeRL and GRPO were not fetched.
- No published sandbox escape *in an RL training run* was found; the risks above are standard container-hardening concerns, not documented incidents.

## Known data-quality problems of NL2Bash (supporting Q1)

### Takeaway
More than half of an NL2Bash-derived, hand-curated subset turned out to be wrong, so NL2Bash itself is likely to be substantially noisy.

### Cited findings
- **InterCode audit:** of 224 pairs drawn from NL2Bash, 113 (50.4%) were invalid:
  - invalid prompt: 7.6%;
  - invalid command: 10.7%;
  - invalid environment: 8.0%;
  - prompt and command both invalid: 12.9%;
  - all three invalid: 4.9%;
  - plus 11 duplicates (4.9%).
  The authors state "there is a risk the NL2Bash dataset contains a significant percentage of erroneous data", which is "concerning because the NL2Bash dataset is commonly used to train NL2SH models" — [NAACL 2025](https://aclanthology.org/2025.naacl-long.555.pdf)
- **Other problems:**
  - commands use deprecated utilities and flags, and descriptions are under-specified (no concrete paths), which InterCode had to fix — [InterCode](https://arxiv.org/html/2306.14898v3)
  - some commands do not round-trip through a parser, and some utilities fall outside the Ubuntu set, both of which NLC2CMD filtered out — [NLC2CMD report](https://arxiv.org/html/2103.02523v2)
  - NL2Bash's own authors say string matching cannot judge correctness and recommend manual evaluation of 50–100 dev examples — [NL2Bash repo](https://github.com/TellinaTool/nl2bash)

### Inferences
- If NL2Bash is used for training, filter it: bashlex or tree-sitter round-trip, utility in the local index, and the command executes without error on a synthetic tree. Down-weight it relative to kural's generator data.

### Gaps
- No error-rate audit of the full NL2Bash (only the 224-pair subset has been audited).
