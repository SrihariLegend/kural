# Recursive / latent-reasoning architectures for Reef (kural): BDH, BDH-CQ, TRM, HRM, recurrent depth, UT/ACT/PonderNet, early exit

Scope: for each idea, what it is, verified results, evidence below 50M (ideally below 10M) parameters, CPU inference cost, fit with Reef (a few-million-parameter, CPU-only, shared-block-iterated NL-to-bash parser; see /home/srihari/Projects/kural/docs/ARCHITECTURE.md), and a verdict. Research date November 2026. "Unverified" marks anything not confirmed in a primary source during this pass.

## Q1. BDH (Dragon Hatchling): what the modus-ponens gate and Hebbian linear attention compute, and what the paper showed; BDH-CQ claims

### Takeaway
BDH is a weight-shared (Universal-Transformer-style) stack in which a low-dimensional state d=256 is lifted by ReLU into a large positive neuron space n (for example 32,768). Plain causal linear attention runs in that neuron space with Q=K=the sparse code and V=the dense state. The "modus-ponens gate" is the elementwise product of the current sparse code x with a ReLU code y read back from attention. The authors claim parity with GPT-2 from 10M to 1B parameters on language and translation, with about 5% active neurons. BDH-CQ's 29.5% pass@2 on ARC-AGI-1 at 150M is verified as an author claim from its abstract and HTML, but it is a single proprietary configuration on GPU. **Verdict: BDH core is TEST (it is already the Reef core, but the parts Reef changes from BDH are untested). BDH-CQ is SKIP as a method source.**

### Cited Findings
**What BDH computes**
- BDH is "a new Large Language Model architecture based on a scale-free biologically inspired network" of locally interacting neurons. The abstract claims it "rivals GPT2 performance on language and translation tasks, at the same number of parameters (10M to 1B), for the same training data", with sparse, positive activation vectors and monosemanticity — [arXiv 2509.26507 abstract](https://arxiv.org/abs/2509.26507)
- State-space form (Eq. 8, BDH-GPU):
  - ρ_{t,l} = (ρ_{t-1,l} + LN(E·y_{t,l-1})·x_{t,l}ᵀ)·U
  - x_{t,l} = x_{t,l-1} + (D_x·LN(E·y_{t,l-1}))⁺
  - y_{t,l} = (D_y·LN(ρ_{t-1,l}·x_{t,l}))⁺ ⊙ x_{t,l}

  Here (·)⁺ is ReLU, LN is parameter-free LayerNorm, U is a diagonal or block-diagonal rotation or damping (RoPE or ALiBi) and ⊙ is the elementwise product — [BDH paper HTML](https://arxiv.org/html/2509.26507)
- The intuition in Section 1.2 is a weighted belief X(i) that contributes X(i)σ(i,j) to A(j), with a Hebbian update that strengthens σ(i,j) by Y(i)X(j). That is the "if i then j" (modus-ponens) reading. The paper gives no separate equation for a "gate" beyond the ⊙ above — [BDH paper HTML](https://arxiv.org/html/2509.26507)
- Official code `bdh.py`, per layer iteration:
  - `x_sparse = relu(x @ encoder)`
  - `yKV = ln(attn(Q=x_sparse, K=x_sparse, V=x))`
  - `y_sparse = relu(yKV @ encoder_v)`
  - `xy_sparse = x_sparse * y_sparse`
  - `y = ln(xy_sparse @ decoder)`
  - `x = ln(x + y)`

  Attention is unnormalised linear: `scores = (QR @ KR.mT).tril(diagonal=-1)`, then `scores @ V`, with RoPE-style phases and Q and K forced to be the same tensor. One set of `encoder`, `encoder_v`, `decoder`, `attn` and `ln` is reused in every layer. Defaults: n_layer=6, n_embd=256, n_head=4, mlp_internal_dim_multiplier=128, dropout 0.1, byte vocabulary of 256 — [pathwaycom/bdh bdh.py](https://raw.githubusercontent.com/pathwaycom/bdh/main/bdh.py)
- "As in the Universal Transformer, all layers use the same set of weights." L=8 layers in most experiments, d=256, parameters ≈ 3nd. The example 25M model uses d=256 and n=32,768 — [BDH paper HTML](https://arxiv.org/html/2509.26507)

**Sparsity, monosemanticity and scaling**
- Sparsity: "only ρ≈5% of the n entries of vector x_t are non-zero" in typical runs, arising from ReLU thresholding. L1 regularisation was disabled, so the sparsity is architectural — [BDH paper HTML](https://arxiv.org/html/2509.26507)
- Monosemanticity: specific synapses strengthen when a specific concept is processed, reported "even below 100M parameters" — [BDH paper HTML](https://arxiv.org/html/2509.26507). The concrete concept examples (I recall currency and country synapses on translation data) are **unverified**: Section 6 was not read.
- Scaling: on a translation task (Fig. 7), BDH grows only n while GPT grows width and depth (with tuned dropout). Training uses truncated BPTT on 2,048-character sequences. The text says BDH-GPU′ matches GPT at all tested sizes up to about 1B. Numeric loss values were not extracted in this pass — [BDH paper HTML](https://arxiv.org/html/2509.26507)
- Limitations the authors note: negative-weight edges are hard to express with local dynamics, and moving fast-weight state into long-term memory is not addressed — [BDH paper HTML](https://arxiv.org/html/2509.26507)

**BDH-CQ (arXiv 2608.09888, 10 Aug 2026, Pathway authors including Kosowski and Chorowski)**
- Abstract: "A 150M-parameter configuration reaches 29.5% pass@2 at a computed inference cost of $0.0007 per task" on the public ARC-AGI-1 evaluation set. The authors claim this breaks the cost-accuracy Pareto frontier — [arXiv 2608.09888](https://arxiv.org/abs/2608.09888)
- Full text: only the 150M configuration is reported, at 29.5% pass@2 (118/400) and 24.25% pass@1 (97/400). "Early experiments" at 1B-600B are mentioned with no numbers — [BDH-CQ HTML](https://arxiv.org/html/2608.09888)
- What the paper discloses about the design:
  - Demonstrations update a recurrent memory S_t = U_θ(S_{t-1}, D_t), and a latent workspace is iterated R steps. R is not given.
  - "Dimensions, exact update rules, and implementation details remain proprietary." No halting rule is given.
  - Training data: private examples plus ARC-AGI-1 training, RE-ARC, ConceptARC, ARC-Heavy and ARC-GEN100K. Compute is not reported.

  — [BDH-CQ HTML](https://arxiv.org/html/2608.09888)
- Cost and effort levels:
  - The cost basis is about 0.85 H200 GPU-seconds per task at $3 per hour, which is GPU inference, not CPU.
  - Reasoning-effort table: HIGH 29.5%, MEDIUM 27%, LOW 21% pass@2. The cost figures in the paper do not reconcile with each other, for example $0.00265 per task "standard effort" against the $0.0007 headline.
  - No step-count ablation. ARC-AGI-2 and language and math tasks are untested.

  — [BDH-CQ HTML](https://arxiv.org/html/2608.09888)

### Inferences
- Reef's planned core departs from BDH as published in three ways that BDH's results do not cover:
  1. **k-WTA at 1-2%** instead of ReLU giving about 5%.
  2. **A delta-rule write with decay and pruning** instead of BDH's plain additive Hebbian accumulation, where RoPE or ALiBi is the only damping.
  3. **A structured Hadamard lift** instead of dense E, D_x and D_y.

  Each needs its own ablation. The BDH paper supports only "ReLU lift + additive linear attention + x⊙y + shared weights ≈ GPT-2 from 10M up".
- The smallest BDH data point is 10M (and the 25M example). There is no published BDH evidence below 10M, which is Reef's regime.
- In BDH most parameters sit in the n×d lift and readout matrices (≈3nd). Reef's plan to shrink these, through a structured lift and a top-k readout, is where it most changes BDH's capacity balance.
- CPU cost (inference from the equations):
  - Per token per pass, BDH costs about 3·n·d multiply-adds for the matrices plus an n×d state read and update. At n=32k and d=256 that is about 25M MACs per token per layer, much more than Reef's 0.7M-op budget.
  - Reef's budget only works with the sparse top-k readout and the structured lift. It must be measured, because sparse gathers on CPU are often memory-bound rather than FLOP-bound.
- BDH-CQ shows that Pathway believes the recurrent-latent and in-context-memory pattern is strong for few-shot rule induction. But it is unreproducible, 150M (about 20-50× Reef), GPU and ARC-only, so it is no evidence for Reef.

### Gaps
- No independent replication of BDH's GPT-2 parity was found in this pass. The scaling numbers were not extracted.
- The concrete monosemanticity examples (Section 6) were not read.
- No published BDH run below 10M, with k-WTA, or with a delta-rule write.
- No public BDH-CQ code or weights. Parameter counts other than 150M are not reported.

## Q2. TRM and HRM: parameters, results, deep supervision and recursion, critiques; transfer beyond grid puzzles

### Takeaway
TRM (7M, 2 layers, one network) beats HRM (27M) on Sudoku-Extreme, Maze-Hard and ARC. The robust lessons are deep supervision with carried and detached state, full backprop through one recursion block, EMA, and small nets on tiny data. Independent analyses show that much of the ARC score comes from:
- the refinement loop
- test-time augmentation and voting
- puzzle-ID embeddings that require the test tasks in training

The architecture itself contributes less. The only language-ish follow-up (Autoregressive TRM) found no reliable gain over standard transformers at matched compute. **Verdict: TEST deep supervision and refinement-loop training for Reef's passes. SKIP the HRM H/L hierarchy and the TRM recipe as evidence that recursion buys language reasoning.**

### Cited Findings
**TRM results and setup**
- TRM: 7M parameters, "a single tiny network with only 2 layers", 45% ARC-AGI-1 and 8% ARC-AGI-2. HRM uses two networks and 27M parameters, trained on about 1,000 examples — [TRM arXiv 2510.04871](https://arxiv.org/abs/2510.04871)
- Test accuracy (%):

  | Benchmark | HRM 27M | TRM-Att 7M | TRM-MLP |
  |---|---|---|---|
  | Sudoku-Extreme | 55.0 | 74.7 | 87.4 (5M) |
  | Maze-Hard | 74.5 | 85.3 | 0.0 |
  | ARC-AGI-1 | 40.3 | 44.6 | 29.6 (19M) |
  | ARC-AGI-2 | 5.0 | 7.8 | 2.4 (19M) |

  — [TRM HTML](https://arxiv.org/html/2510.04871)
- Recursion: TRM uses n=6, T=3, one 2-layer network and N_sup up to 16, for an effective depth of 42 per supervision step. HRM uses n=2, T=2 and two 4-layer networks. Deep supervision carries (y, z) across steps, detached — [TRM HTML](https://arxiv.org/html/2510.04871)
- Gradient: TRM backprops through all n+1=7 recursion evaluations of the last cycle. HRM's 1-step (implicit-function-style) approximation scored 56.5% against 87.4% on Sudoku — [TRM HTML](https://arxiv.org/html/2510.04871)
- Halting: HRM uses Q-learning ACT with halt and continue losses, which needs 2 forward passes. TRM uses only a halting BCE (1 pass). Dropping the continue loss gave 86.1% → 87.4%, which the paper calls no significant difference — [TRM HTML](https://arxiv.org/html/2510.04871)
- Ablations on Sudoku:

  | Variant | Sudoku accuracy |
  |---|---|
  | TRM default (2 layers, single network, MLP, EMA) | 87.4% |
  | 4 layers, n=3 (10M) | 79.5% |
  | Separate f_H/f_L networks (10M) | 82.4% |
  | Self-attention instead of MLP | 74.7% |
  | No EMA | 79.9% |

  The authors attribute the advantage of the small network to overfitting on scarce data and offer no theory — [TRM HTML](https://arxiv.org/html/2510.04871)
- Data and compute:
  - Sudoku: 1K training examples with 1,000 augmentations each, on 1 L40S for under 36 hours.
  - ARC: 4 H100s for about 3 days, 100K epochs.
  - The ARC answer is a majority vote over 1,000 augmentations. No inference latency or FLOPs are reported, and n=6, T=6 ran out of memory.

  — [TRM HTML](https://arxiv.org/html/2510.04871)

**Critiques and replications**
- ARC Prize verification of HRM, Semi-Private set:
  - ARC-AGI-1: 32%, against the paper's 41% on Public Eval, at $1.48 per task.
  - ARC-AGI-2: 2%.

  — [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis)
- ARC Prize ablations of HRM:
  - **Refinement loop:** the outer refinement loop is the largest factor. Going from 1 to 2 loops gave +13 points, and 1 to 8 loops roughly doubled the score. Training with 16 loops raised single-loop inference by more than 15 points.
  - **Hierarchy:** a plain transformer of about 27M came within about 5 points untuned and matched HRM at 1 loop. Changing the H and L iteration counts from 2/2 made results worse.
  - **Transfer and conditioning:** training only on the 400 eval tasks still gave 31% (against 41%). Puzzle-ID embeddings mean HRM can only solve puzzles it saw in training, so it is "purely transductive".

  — [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis)
- Re-analysis of the TRM ARC-AGI-1 checkpoint (arXiv 2512.11847, v2 Jan 2026):
  - 1,000-sample voting adds about 11 points of pass@1.
  - A blank or random puzzle ID gives 0% accuracy.
  - "Most final accuracy appears at the first recursion step", so effective recursion is shallow.
  - The authors conclude that performance comes from efficiency, task conditioning and test-time compute, "not deep internal reasoning".

  — [arXiv 2512.11847](https://arxiv.org/abs/2512.11847)
- Tiny Autoregressive Recursive Models (Rauba, Fanconi, van der Schaar; ICLR 2026 workshop): in compute-matched experiments on character-level algorithmic tasks, the full Autoregressive TRM gave "no reliable gains" over standard-Transformer variants, while some two-level refinement baselines did well — [arXiv 2603.08082](https://arxiv.org/abs/2603.08082)

### Inferences
- What transfers to Reef is the **training recipe**, not the claimed reasoning:
  - a loss at every pass (deep supervision)
  - carried state with detached history
  - full backprop through the last pass group
  - EMA
  - small width with many reuses when labelled data is scarce

  The refinement-loop finding (training with many loops improves even 1-loop inference) suggests Reef's per-pass supervision may help shallow-exit quality too. This needs testing.
- Grid puzzles are fixed-size, non-autoregressive and have one exact answer per input. Reef's decoder is autoregressive under grammar constraints, so recursion would apply to the encoder (request understanding), which is a different setting. The only autoregressive test (2603.08082) is negative.
- TRM's "MLP beats attention on small fixed grids, fails on large ones" result is a warning that small-data wins are task-specific.

### Gaps
- No TRM or HRM result on any natural-language, semantic-parsing or code task was found. Only an informal TinyStories forum post exists ([Google Dev forum](https://discuss.google.dev/t/tiny-recursive-models-for-text-generation/318588)), with no rigorous numbers.
- The exact numbers in 2603.08082 were not extracted (abstract only).
- No ARC Prize semi-private verification of TRM was retrieved in this pass. A TRM cost of $1.76 per task is cited only second-hand, inside BDH-CQ.

## Q3. Recurrent-depth latent reasoning (Geiping et al. 2025), Universal Transformer / ACT / PonderNet, early exit: results, iterations, cost, halting stability

### Takeaway
Geiping et al. show large gains from iterating a 4-layer core at 3.5B parameters and 800B tokens. On GSM8K CoT, r=1 scores 0% and r=32 scores 42% flexible, and easy tasks saturate at about 8 iterations. A simple KL-threshold per-token exit costs almost nothing in quality. They also document how fragile training is: tokens collapse to the same state, or the model learns to ignore the recurrence. Learned halting (ACT) is reported to be unstable by the PonderNet authors. **Verdict:**
- **ADOPT** the training tricks: randomly sampled iteration count, truncated backprop, sandwich norm and a low learning rate.
- **ADOPT** a simple convergence-threshold exit with a hard cap, rather than a learned halting unit.
- **TEST** whether extra passes help at all at a few million parameters, since there is no small-scale evidence.

### Cited Findings
**Geiping et al. architecture and training**
- 3.5B parameters, 800B tokens. Test-time unrolling reaches a computation load "equivalent to 50 billion parameters" — [arXiv 2502.05171](https://arxiv.org/abs/2502.05171)
- Layout: prelude 2 layers, core 4, coda 2. Width 5,280 with 55 heads. r=32 gives an effective 132 layers — [Geiping HTML](https://arxiv.org/html/2502.05171)
- Training draws r from a log-normal Poisson (σ=½, mean about 32) and backprops only through the last k=8 iterations, so memory stays constant — [Geiping HTML](https://arxiv.org/html/2502.05171)
- Benchmarks:

  | Benchmark | r=1 | r=32 |
  |---|---|---|
  | GSM8K CoT (strict / flexible) | 0 / 0 | 34.8 / 42.1 |
  | ARC-C | 24.1 | 38.2 |
  | HellaSwag | 29.3 | 65.2 |
  | SciQ | 47.1 | 93.5 |

  HumanEval 23.2 and MBPP 24.8 are reported at r=32 only — [Geiping HTML](https://arxiv.org/html/2502.05171)
- Saturation depends on the task: HellaSwag at about 8 iterations, ARC-C at 8-12 with no few-shot examples and about 32 with 25-50 examples — [Geiping HTML](https://arxiv.org/html/2502.05171)

**Exit rule and latent trajectories**
- Exit when the KL between successive steps is below 5×10⁻⁴. MT-Bench scores 5.63 with standard decoding and 5.56 with early exit. Warm-starting from the previous token's state saves about 1-2 steps. No end-to-end compute-saving figure is given — [Geiping HTML](https://arxiv.org/html/2502.05171)
- Latent trajectories: many tokens converge to fixed points, harder tokens fall into multi-dimensional **orbits**, and some act as "sliders" that drift steadily — [Geiping HTML](https://arxiv.org/html/2502.05171)

**Training failures**
- Bad Run 1 had token hidden-state correlation rising toward 1, collapsing every token to the same state.
- Bad Run 2 learned to ignore the incoming state: perplexity was identical at r=1 and r=32.
- The fix was sandwich norm and a peak learning rate of 4×10⁻⁵ instead of 4×10⁻⁴.
- Small (1,4,1)-shaped, 1,024-wide ablation models were used, with no numbers reported.

— [Geiping HTML](https://arxiv.org/html/2502.05171)

**ACT, PonderNet, UT**
- PonderNet (Banino et al. 2021) describes ACT as unstable. It reformulates halting as a probabilistic (geometric) model with a prior λp — [PonderNet arXiv 2107.05407](https://arxiv.org/pdf/2107.05407)
- The claims that PonderNet is robust to λp while ACT is sensitive to its time penalty, and that sampled halting adds variance, come only from an aggregator. They are **unverified** in the primary paper here — [Emergent Mind summary](https://www.emergentmind.com/topics/pondernet)
- PALBERT proposes a deterministic "Q-exit" criterion to stabilise PonderNet-style exits in BERT — [arXiv 2204.03276](https://www.arxiv.org/abs/2204.03276v4)
- Csordás et al. 2021 ("The Devil is in the Detail") test Universal Transformers (shared weights, no ACT) on SCAN, COGS and others. They show that tricks such as relative position, scaling and early stopping raise COGS from 35% to 81% — [arXiv 2108.12284](https://arxiv.org/pdf/2108.12284). The search result does not say whether that figure is for the UT or the standard Transformer, so the UT-specific numbers are **unverified** here.

### Inferences
- **Orbits in Geiping's latent trajectories** are direct evidence for the Reef risk "stop when stable can oscillate". A Reef exit rule should combine a hard cap (for example 6-8 passes), a tolerance threshold on the change in output distribution (KL) rather than exact equality of the k-WTA set, and maybe hysteresis.
  - Under k-WTA a single neuron swapping in or out at the k boundary can flip forever, so "the pattern stops changing" is the fragile criterion.
- **Reef's self-modifying delta-rule memory adds a feedback loop** that Geiping's model lacks: the state S changes every pass. Ways to cut that loop:
  - write to S once per token, not per pass
  - freeze writes during the inner passes
  - damp writes with β

- **CPU cost scales linearly with passes**, and early exit only helps if most requests exit early. Reef's budget (about 0.7M ops per pass × 6 passes × 20 tokens ≈ 80M) is small enough for milliseconds in principle. The risk is memory traffic in sparse gathers and scatters, not FLOPs. This is not measured anywhere in the literature found.
- Train with a random pass count, as Geiping does, and with per-pass loss, as in TRM and Needle-style ladder exits. Otherwise the model can learn to ignore recurrence, which is Geiping's Bad Run 2. Check this by comparing the loss at r=1 against r=max.

### Gaps
- No recurrent-depth result below about 1B parameters with reported numbers was found. Geiping's small ablations are unreported.
- PonderNet's own quantitative stability comparisons were not extracted. Only secondary claims were found.
- No per-request CPU latency measurements for any iterated-block model were found.

## Q4. Evidence on semantic parsing, text-to-SQL, NL-to-code or NL2Bash; overall verdicts and risks

### Takeaway
I found no published evidence that BDH, TRM, HRM, BDH-CQ or recurrent-depth halting help on NL2Bash, text-to-SQL or NL-to-code parsing. The nearest evidence is Universal-Transformer (shared-weight) work on compositional semantic-parsing benchmarks (SCAN and COGS), where training details mattered more than the architecture. Every gain claimed for these models comes from grid puzzles (TRM, HRM, BDH-CQ) or from large-scale LM benchmarks (Geiping, BDH at 10M-1B). So recursion is an experiment for Reef's ladder, not a foundation to assume.

### Cited Findings
- UT and Transformer studies on SCAN and COGS (Csordás et al.) find that "simple tricks" such as positional encoding, scaling and early-stopping choices drive compositional-generalisation results. The UT there is weight-shared without ACT — [arXiv 2108.12284](https://arxiv.org/pdf/2108.12284)
- Ontanon et al. (ACL 2022) explicitly ask whether weight sharing across layers helps compositional generalisation — [ACL 2022.acl-long.251](https://preview.aclanthology.org/author-url/2022.acl-long.251/). The answer could not be extracted (the PDF was unreadable), so it is **unverified**.
- The only code results for a recurrent-depth model are HumanEval 23.2 and MBPP 24.8 at r=32, for a 3.5B model, with no r=1 comparison — [Geiping HTML](https://arxiv.org/html/2502.05171)
- TRM is non-generative and gives one answer per input, which the authors list as a limitation — [TRM HTML](https://arxiv.org/html/2510.04871). BDH-CQ explicitly leaves language and math untested — [BDH-CQ HTML](https://arxiv.org/html/2608.09888)

### Inferences
**Summary verdicts for Reef**

| Idea | Verdict | Reason |
|---|---|---|
| BDH core (ReLU lift, x⊙y, linear attention, shared weights) | **TEST** | GPT-2 parity claimed from 10M; nothing below 10M. Reef's k-WTA, delta rule and structured lift are untested departures, so ablate each against BDH-as-published at equal parameters. |
| BDH-CQ | **SKIP** (as method) | 150M, proprietary, GPU-costed, ARC-only, internally inconsistent cost figures. It is only a signal that Pathway's recurrent-latent and in-context-memory direction works on few-shot rule induction. |
| TRM/HRM deep supervision, full backprop through the last pass, EMA | **TEST / likely adopt** | These are the robust, ablated parts (87.4 vs 56.5 for full against 1-step gradient; 87.4 vs 79.9 with and without EMA). Training-time refinement loops raised 1-loop inference by more than 15 points in ARC Prize's HRM ablations. |
| HRM two-level H/L hierarchy | **SKIP** | ARC Prize found it had minimal effect, and TRM's single network beat separate networks. |
| TRM or HRM as evidence that recursion "buys reasoning" for language | **SKIP** | ARC gains depend on puzzle IDs, voting and the refinement loop; recursion is shallow; and the autoregressive test showed no reliable gain. |
| Geiping recurrent-depth training recipe | **ADOPT** | Random r, truncated backprop, sandwich norm and a low learning rate are cheap, and they prevent the documented failures: token collapse and the model ignoring the recurrence. |
| Geiping benefit of many passes | **TEST** | Proven only at 3.5B / 800B tokens. Reef should measure accuracy against pass count (1, 2, 4, 6, 8). |
| Learned halting (ACT, PonderNet) | **SKIP for now** | ACT is reported unstable. A KL or stability threshold plus a hard cap worked nearly losslessly at scale (MT-Bench 5.63 → 5.56) and needs no extra training objective. |
| Early exit when stable | **ADOPT with changes** | Use a KL tolerance on the output head, not identity of the k-WTA set, plus a cap and hysteresis, given the observed orbits. |

**Key risks and suggested measurements**
1. **Oscillation.** Geiping observed orbits, and k-WTA boundary flips are an added risk. Log how many passes each request uses and what fraction hits the cap.
2. **Recurrence ignored or token collapse.** Compare the loss at r=1 against r=max, and the token-state correlation, during training.
3. **Delta-rule memory changing inside the pass loop** creates a feedback loop no cited model has tested.
4. **CPU latency** is linear in passes, and sparse access is bandwidth-bound. Benchmark the real kernels before committing to more than 6 passes.
5. **Small-data overfitting.** TRM's lesson is that smaller and deeper-by-reuse beats wider. That supports Reef's approach, but it was shown only on grids.

### Gaps
- No study of recurrence or adaptive depth on NL2Bash, Spider/text-to-SQL or other NL-to-code parsing was found in this pass. That is a clear literature gap.
- UT-specific numbers on SCAN and COGS (Csordás; Ontanon) were not extracted.
- No independent BDH replication, and no BDH result below 10M parameters.
