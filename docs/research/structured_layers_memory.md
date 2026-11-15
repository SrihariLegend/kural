# Structured layers, lookup memories, sparsity and confidence heads for Reef (kural)

Scope: Cactus Needle 3 components, DeepSeek Engram, product-key memory and memory layers, hashed n-gram embeddings, structured matrices (Monarch, Kronecker, Hadamard, low-rank), k-winners-take-all sparsity, and confidence heads. Each is judged for Reef: a CPU-only model of a few million parameters (D about 128) that parses requests into a typed program under a grammar (design: /home/srihari/Projects/kural/docs/ARCHITECTURE.md).

Primary source for Needle 3: the cloned source at `/tmp/claude-1000/-home-srihari-Projects/f59bc3d6-c14f-49cb-9e4f-a2b62c465ca7/scratchpad/cactus2` (`cactus/model/architecture.py`, `cactus/model/config.py`, `llms.txt`), which may be cleared on reboot. Repo: https://github.com/cactus-compute/cactus. Claims marked "(source code)" were read directly from those files. Claims marked "(prior knowledge, not re-fetched)" are from my training knowledge, were not checked this session, and should be verified before anyone relies on them.

---

## Q1. What Needle 3 is and what it reports (size, accuracy, ablations of its tricks)

### Takeaway
Needle 3 is a 121M-parameter model, of which 70.8M sits in Engram tables. Its "Hadamard MLP" is a nearly parameter-free square mixer of about 25K parameters per layer, not a lift to a wide sparse layer. In effect Needle is a model built from attention plus lookup tables. Cactus publishes headline claims ("beats models 10x its size on mobile tool calls"), but its pages contain no per-benchmark numbers and no component ablations, apart from an FFN-removal study, a QK-norm divergence note, a neutral result for scalar residual gates, and a ladder ablation.

### Cited Findings

**Size and layout**
- Needle 3 has 29-121M parameters over 2-20 layers and ships as an 8-29 MB binary at 2 bits (CQ2). The full 20-layer model has 121M parameters, 70.8M of them in Engram tables. It was trained on 360B tokens of a proprietary structured dataset. — [Cactus Needle page](https://cactuscompute.com/needle)
- Components listed:
  - an 8,192 × 768 embedding with tied unembedding and SentencePiece BPE (8,192 pieces);
  - mHC read and write across 4 residual lanes;
  - Engram (hashed 2- and 3-grams) at layers 3, 7, 11, 15 and 19, with 18,432 slots;
  - GQA attention with RoPE, QK-norm and a 1024-token window, plus global attention at layers 4, 9, 14 and 19;
  - a 3-tap causal convolution per channel;
  - a "Monarch Hadamard FFN";
  - ZCRMSNorm, a byte-level grammar, and an MTP t+2 head used only in training.
  The architecture is said to use "over 2x fewer MFLOPs per token" than a transformer of the same configuration. — [Cactus Needle page](https://cactuscompute.com/needle)
- The defaults in `config.py` are d_model 768, 12 heads, 2 KV heads, 20 layers, qk_head_dim 48, v_head_dim 64, vocab 16384, `engram_orders=(2,3)`, `engram_slots=18432`, `engram_layers=(3,7,11,15,19)`, `qkv_conv_taps=3` and `mhc_lanes=4`. The vocab default (16384) differs from the shipped 8,192 on the product page. — (source code, config.py)

**Benchmarks**
- Datasets: Mobile Actions (961 rows), DroidCall (200), BFCL v4 (3,641), DSTC8 (1,813 turns) and two SNIPS sets (700 each). Baselines are f16 models under vLLM and DeepSeek V4 Flash via API. The page text gives no per-benchmark accuracy; the numbers exist only inside figures. — [Cactus Needle page](https://cactuscompute.com/needle)
- Fine-tuning on DroidCall raises every subnetwork by 18-36 points. From 4 layers (29M) up, tuned subnetworks pass DeepSeek V4 Flash. The 2-layer base model scores 0 when gated and 20.5 on DroidCall with calls forced, and reaches 56.5 after fine-tuning. — [Cactus: Intelligence ladders](https://cactuscompute.com/blog/intelligence-ladders); [Needle page](https://cactuscompute.com/needle)
- Cactus' own docs admit weak spots: "The shipped base model does not pass all six suites on engine 2.0.4 at the default gate: five fall short, missing calls the query did state and inventing values it did not." A suite passes at 29 of 32 cases. — (source code, llms.txt)

**Speed**
- 400-4k tokens/s decode and 1-10k tokens/s prefill on a Raspberry Pi 5. No other CPU benchmarks are given. — [Needle page](https://cactuscompute.com/needle)

**Hadamard MLP (source code, architecture.py `HadamardMLP`)**
- The width is n = the next power of two of d_model (1024 for 768); the output is cut back to d_model. It is square, not an expansion.
- It applies three Kronecker-factored maps `kron(A_s, B_s)`. Each factor is a learned 32×32 matrix initialised to a normalised Walsh-Hadamard matrix (`_hada_blocks(1024)` gives 32×32).
- The rest of the block:
  - learned per-channel diagonal scales d1, d2, d3 and an output scale d4 initialised to 0.02;
  - one SiLU, with bias b2;
  - two fixed seeded permutations between stages;
  - a rank-8 input-conditioned multiplicative modulation, `cond = 1 + softmax(x·V)·U`, with U initialised to zero.
- Cactus states "25K parameters per layer" for it. A dense 768→3072→768 FFN would be about 4.7M. — [Simple Attention Networks blog](https://cactuscompute.com/blog/simple-attention-networks)

**Ablations Cactus reports**
- FFN-removal study (20 layers, d=512, 87M-parameter transformer, SYNTH data up to 105B tokens). Removing the FFN costs:
  - 0.47 nats at matched depth (87M vs 24M);
  - 0.26 nats at matched FLOPs (43M vs 24M);
  - 0.0055 nats at matched parameters (24M vs 24M).
  — [Simple Attention Networks](https://cactuscompute.com/blog/simple-attention-networks)
- QK-norm is "load-bearing": "Removing it diverges training at the tuned learning rate", the only divergence in the study. Scalar residual gates are "performance-neutral at every depth from 20 to 48 layers". Post-attention sandwich normalisation is the only variant that beats the baseline, by 0.009 nats. — [Simple Attention Networks](https://cactuscompute.com/blog/simple-attention-networks)
- The blog gives no loss or speed comparison of the Hadamard MLP against a dense FFN, and no numbers for Engram, short convolutions or mHC. — [Simple Attention Networks](https://cactuscompute.com/blog/simple-attention-networks)

**Attention details (source code, `MultiHeadAttention`)**
- q, k and v each pass a per-channel causal depthwise convolution (3 taps, identity-initialised), then RMS QK-norm, then RoPE.
- The attention output is multiplied by `sigmoid(W_gate·x)`: an elementwise, input-dependent gate, separate from the scalar residual gates.

**Engram in Needle (source code, `engram_indices`, `Engram`, `Block`)**
- Hashing: multiplicative-XOR over the last n token ids, with a seeded hash per head.
- Tables: per site, 2 orders × 3 heads = 6 tables of 18,432 × 128. That is 14.2M per site and 70.8M over 5 sites, matching the page.
- Fetched vectors go through key and value projections to d_model. Values pass a 4-tap dilated causal convolution.
- Injection is `x += sigmoid(<rms(x), rms(k)>/√d) · v`, a context-aware scalar gate per position and site.
- 18,432 = 2^11·9 is not prime. DeepSeek uses prime-sized tables (Q2).

**Ladder (early-exit depths)**
- Training: each step runs the full model with probability 0.8. With probability 0.2 it runs a random depth d in {2..L-1}, scaled by d/(2L), with cross-entropy plus KL (λ=1) toward the stop-gradient full model.
- Results (18 layers, only 3,000 steps):
  - full-depth validation loss 1.958 with the ladder vs 2.037 without, a 3.9% improvement;
  - bits per byte by depth: 18L 0.660, 12L 0.769, 4L 0.991, 2L 1.538;
  - switching the ladder on late was worse (2.059).
  — [Intelligence ladders](https://cactuscompute.com/blog/intelligence-ladders)
- Rung sizes: 2L 25M; 4L 29M (30 MFLOPs/token); 8L 52M; 16L 98M; 20L 121M (100 MFLOPs/token). No comparison against separately trained models of each size. — [Intelligence ladders](https://cactuscompute.com/blog/intelligence-ladders)
- Rungs are chosen by bisection order: blocks 0 and L-1 always, then midpoints. Rungs are nested and slice the Engram sites and head rows. — (source code, config.py `_ladder_layer_order`, `ladder_slice`)

**Heads**
- The confidence head (1 output) and router head (3 outputs) are "probe heads". Learned probes attend over every layer's hidden states (4 probes per layer). A second softmax over layers and probes, driven by 4 learned queries, pools them, followed by a linear projection. The router export carries a 3-value calibration vector, initialised at (0.90, 0.00, 0.60). — (source code, `probe_pool`, `ConfidenceHead`, `RouterHead`)

### Inferences
- **The design document misreads Needle on one point.** ARCHITECTURE.md takes "Walsh-Hadamard factors in Kronecker form ... (Needle's Hadamard MLP)" as the basis for a D=128 → N=8k-32k lift. Needle never lifts. Its Hadamard MLP is a square, width-preserving, nearly parameter-free mixer.
- **Cactus' own evidence is that the FFN slot hardly matters at matched parameters.** Their 0.006-nat result points the other way from "a wide sparse lift carries knowledge": they moved parameters out of the FFN and into attention and Engram tables. So Needle gives no evidence that a structured lift can stand in for a dense one.
- **Needle has no result that maps to Reef's size.** Its smallest rung (25-29M) is still 3-4× Reef's whole budget, and about 70% of Needle's parameters are lookup tables.
- **Verdicts for Reef:**
  - QK-norm: **adopt**. Cactus says removing it diverged training, and Memory Layers at Scale also needed it for small models.
  - Sigmoid output gate on attention: **adopt**. Near-free; Qwen's gated-attention work reports gains (prior knowledge, not re-fetched: https://arxiv.org/abs/2505.06708).
  - 3-tap q/k/v convolutions: **adopt**. Negligible cost on a 44-token request.
  - Scalar residual gates: **skip** as a claimed gain (neutral per Cactus).
  - Ladder exits / deep supervision at every pass: **adopt** in the recurrent-depth form, i.e. supervise every pass, with self-distillation toward the final pass. Note that Reef's "iterate one block" is not Needle's nested slicing of distinct blocks.
  - mHC lanes: **skip** for now (no ablation published; adds complexity).
  - Probe-pooled confidence and router heads: **test**. See Q5.

### Gaps
- Per-benchmark accuracies for Needle 3 vs larger models exist only in chart images. I could not extract them from the page text.
- No published ablation of Engram, the Hadamard MLP vs a dense FFN, short convolutions or mHC inside Needle 3.
- How the confidence head is trained (labels, calibration method, ECE or AUROC) is not disclosed. See Q5.
- Whether "Monarch Hadamard FFN" on the product page means something beyond the Kronecker-of-Walsh code in `architecture.py` (for example a different shipped variant) is unknown.

---

## Q2. Lookup memories: DeepSeek Engram, product-key memory, memory layers at scale, hashed n-gram embeddings. Do they help small models on language understanding?

### Takeaway
Lookup memories reliably help large language models, mostly on **factual recall**. Engram even shows that reading-comprehension skill barely depends on it. At small scale and small table size the gains are small: in PKM's own ablation, a 16k-slot memory improved perplexity by under 1%. Input-side hashed n-grams are the most robust variant across sizes, including a 2.4M-parameter model in a synthetic task. For Reef, Engram tables are a **test**, not an adopt. Expect little help on understanding paraphrases, and a possible risk to robustness from matching on surface n-grams.

### Cited Findings

**DeepSeek Engram** (arXiv 2601.07372, Jan 2026)
- Design:
  - tokenizer compression (NFKC, lowercasing; about 23% smaller effective vocabulary for a 128k tokenizer);
  - suffix 2- and 3-grams with K hash heads per order, each a prime-sized table with a multiplicative-XOR hash;
  - context-aware gating, with the hidden state as query and memory as key and value;
  - a depthwise causal convolution (kernel 4).
  — [arXiv 2601.07372](https://arxiv.org/html/2601.07372v1)
- Engram-27B (26.7B total, 3.8B active, 5.7B Engram) vs an iso-parameter, iso-FLOPs MoE-27B, 262B tokens each:
  - MMLU 57.4 → 60.4
  - BBH 50.9 → 55.9
  - ARC-C 70.1 → 73.8
  - DROP 55.7 → 59.0
  - HumanEval 37.8 → 40.8
  - GSM8K 58.4 → 60.6
  The abstract says MMLU +3.4; the table says +3.0. — [arXiv 2601.07372](https://arxiv.org/html/2601.07372v1)
- Allocation follows a U-shaped law, with the optimum at about 20-25% of the sparse budget in Engram. At 6e20 FLOPs, loss goes from 1.7248 (pure MoE) to 1.7109. Loss falls log-linearly with slot count, up to about 13B added parameters. — [arXiv 2601.07372](https://arxiv.org/html/2601.07372v1)
- Ablations (3B backbone):
  - Removing multi-branch fusion, context-aware gating or tokenizer compression gave the largest regressions.
  - Removing the convolution had a minor effect; 4-grams were slightly worse.
  - Early insertion is best: layer 2 alone gives loss 1.770; layers 2 and 6 give 1.768; the MoE baseline is 1.808.
  — [arXiv 2601.07372](https://arxiv.org/html/2601.07372v1)
- Mechanism: Engram makes early layers converge sooner (LogitLens), and Engram layer 5 resembles MoE layer 12 (CKA), so it adds effective depth. Disabling Engram at inference keeps only about 29% of TriviaQA performance but about 93% on C3 reading comprehension. — [arXiv 2601.07372](https://arxiv.org/html/2601.07372v1)
- Offloading a 100B table to host DRAM costs 1.9-2.8% throughput (4B and 8B dense). — [arXiv 2601.07372](https://arxiv.org/html/2601.07372v1)

**Product-key memory** (Lample et al., NeurIPS 2019)
- Setup: 4 heads, k=32, 512² slots, CC-News data.
- 12 layers × 1024 dimensions: 17.66 perplexity → 15.62 with one memory → 14.47 with three. A 12-layer model with memory (15.62) beats the 24-layer baseline (16.02).
- Memory-size ablation (6 layers, memory at layer 5; baseline 22.98):
  - 16k slots: 22.80 (BatchNorm on queries: 22.97)
  - 262k slots: 19.78 with BatchNorm
  - 1M slots: 17.98 with BatchNorm
  Slot usage falls to 26% without BatchNorm at 1M slots (80% with it).
  — [ar5iv 1907.05242](https://ar5iv.labs.arxiv.org/html/1907.05242)

**Memory layers at scale** (Meta, arXiv 2412.09764)
- At a 134M base (Memory+ adds about 1M values, 937M total parameters):
  - NQ 0.91 → 3.16
  - TriviaQA 7.7 → 18.77
  - PIQA 62.13 → 65.94
  - OBQA 16.4 → 17.6
  PEER and MoE give similar gains. Memory+ uses SiLU gating, 3 memory layers sharing one pool, and QK-norm "when training becomes unstable, mainly on small base models".
- Best placement in the 373M ablation is three layers (4, 12, 20).
- The paper notes memory layers are not yet as fast as dense FFNs on current hardware. It gives no CPU figures.
— [arXiv 2412.09764v2](https://arxiv.org/html/2412.09764v2)

**Hashed n-gram input embeddings** (Over-Tokenized Transformer, arXiv 2501.16975, ICML 2025)
- Loss is log-linear in input-vocabulary size. A 400M model with a 128× input vocabulary (hashed 1-, 2- and 3-gram tables, m ≈ 12.8M) matches a 1B baseline. Each 4× in table size gives about −0.015 loss.
- Keeping the 1-gram matters: dropping it gave 2.785 vs 2.714 baseline vs 2.667 for all three orders.
- Use a table size coprime with the vocabulary: m = 64V gives 2.702 vs 2.678 for coprime m.
- Small-model result: in a synthetic CFG task, 3-gram input-side *encoders* helped at every size, but 3-gram *tokenization* or *decoders* (output side) hurt the 2.4M model while helping the 85M one.
— [arXiv 2501.16975v2](https://arxiv.org/html/2501.16975v2)

**N-Grammer** (Roy et al. 2022)
- Product-quantised latent bigram IDs added to embeddings. Reported to beat Transformer and Primer on C4 and SuperGLUE. A 343M N-Grammer reaches C4 perplexity 14.79. No independent replication found. — [arXiv 2207.06366](https://arxiv.org/abs/2207.06366v1)

### Inferences
- **Table size.** Reef's planned tables (2 orders × about 16k slots × 32 dimensions) sit at the *smallest* end of every curve above. PKM at 16k slots gave about 0.8% perplexity improvement, and the gains in all papers come from 10^5-10^9 slots. Engram's own sensitivity test says the module mostly stores facts, while context reading stays in attention.
- **What Reef's tables would store.** The "phrase-level facts" Reef wants ("older than", "in place") number in the hundreds to low thousands. A 128-wide core trained on many paraphrases will learn them anyway, so tables help mainly if training data is thin for long-tail phrases (Stack Exchange titles, man-page vocabulary).
- **Robustness risk.** Hashed exact n-grams reward surface form; Reef's stated weakness is wording and paraphrase (the JEPA motivation). Engram's tokenizer normalisation (lowercasing, NFKC) and gating reduce this but do not remove it.
- **Cost on CPU is trivially small.** About 2-6 lookups per token plus a d×d key/value projection per site. File size is the real cost: 1M parameters at 4 bits is about 0.5 MB.
- **Verdict: test.** One site early in the network (pass 1, or right after the embedding), with:
  - context-aware sigmoid gating (from Engram and Needle);
  - 1-, 2- and 3-gram orders (Over-Tokenized);
  - prime or coprime table sizes;
  - multiple hash heads;
  - normalised token IDs.
  Measure on the held-out *paraphrase* split, not just in-distribution. The comparison should be against putting the same parameters into a wider embedding or the core.
- **Product keys: skip** at Reef's scale. They only make sense for 10^5+ slots, and PKM needs BatchNorm or QK-norm to avoid slots going unused.
- **One idea from product keys does apply to the lift (see Q3).** A product-key lookup is exactly a cheap, structured top-k over N = |K1|·|K2| "neurons".

### Gaps
- No published Engram or hashed-n-gram result on a model under about 50M parameters trained on real language. The only sub-50M datapoint is the synthetic CFG experiment (2.4M).
- No ablation of Engram inside Needle 3, so its contribution to Needle's tool-calling results is unknown.
- No source measures lookup memories on semantic-parsing or text-to-code tasks specifically.

---

## Q3. Structured matrices (Monarch, butterfly, Kronecker, Hadamard) vs low-rank vs dense. Can a Kronecker/Hadamard lift from 128 to 8k-32k sparse neurons replace a dense lift?

### Takeaway
Full-rank structured matrices *without parameter sharing* (Monarch, Block Tensor-Train) can match dense at equal compute: Monarch GPT-2 Small reaches 20.7 perplexity vs dense 20.6 with 58% of the parameters. Structures that *share parameters*, Kronecker included, scale worse. Low-rank is worse whenever rank is cut. A pure Kronecker/Hadamard lift from 128 to 16k neurons has about 3K free parameters for 16K neurons: every neuron's weight vector is a product of two short factor vectors, and the k-WTA winners would form a near-grid of two small codes. There is no evidence it replaces a dense lift, and the arithmetic says it cannot carry 16k independent features. **Skip as the default; test only against dense, Monarch/BTT and a product-key top-k at equal FLOPs.**

### Cited Findings
- **Monarch** (Dao et al., ICML 2022), on WikiText-103:
  - GPT-2 Small: 20.6 perplexity at 124M parameters and 106G FLOPs; Monarch version 20.7 at 72M and 51G, with 1.8× training speedup.
  - GPT-2 Medium: 20.9 at 355M; Monarch 20.3 at 165M, 2.0× faster.
  Speedups are for training on GPU. — [arXiv 2204.00595](https://arxiv.org/pdf/2204.00595); [PMLR](https://proceedings.mlr.press/v162/dao22a.html)
- **Compute-optimal comparison across structures** (Potapczynski et al., NeurIPS 2024):
  - Structures are characterised by ω (parameter sharing), ψ (rank exponent) and ν (compute intensity).
  - "A small ω and large ψ reliably led to better scaling laws". Structures that beat dense in non-compute-optimal settings give "similar but not significantly better compute-optimal scaling laws".
  - Kronecker (ψ=1) "skips a factor", which puts it outside the ω=0 subspace where the best structures lie, and ω>0 scales worse. Low-rank (ψ<1) scales worse. Monarch is ω=0, ψ=1, ν=0.5.
  - Experiments use GPT-2 models of **120k-76M parameters**, d in 256-4096, 3 or 6 layers, so this is directly relevant to Reef's scale.
  - μP-style, structure-aware learning-rate scaling "is crucial".
  - The paper's best performer is BTT-MoE, a mixture of experts inside every linear layer.
  — [arXiv 2410.02117](https://arxiv.org/html/2410.02117v2)
- **Structured FFNs in LLMs** (Wei et al., NeurIPS 2024): LowRank, BlockShuffle and BlockDense FFNs are "harder to train ... loss spikes". They needed a "self-guided training" regime. With it: 32% of FFN parameters, a 2.5× speedup and only +0.4 perplexity at equal training FLOPs. — [arXiv 2406.16450](https://arxiv.org/abs/2406.16450)
- **Cactus' FFN-removal study:** removing the FFN costs 0.26 nats at matched FLOPs but only 0.006 at matched parameters (Q1). Their near-free Hadamard MLP is a square mixer, not a lift. — [Simple Attention Networks](https://cactuscompute.com/blog/simple-attention-networks)
- **Product keys as a structured top-k:** sub-key scores over two halves of the query are summed, so the exact top-k over |K1|·|K2| slots is found from top-k on each half. Cost is about 2√N·d/2 instead of N·d. — [ar5iv 1907.05242](https://ar5iv.labs.arxiv.org/html/1907.05242)

### Inferences
These follow from the findings plus arithmetic; none is a published result.

- **Degrees of freedom.** A Kronecker lift A⊗B from 128 (reshaped as 8×16) to 16,384 = 128×128 neurons has A: 128×8 and B: 128×16, about 3K parameters. Neuron (i,j) computes a_iᵀ X b_j. Hadamard mixing and fixed shuffles add only diagonals.
  - A dense lift gives each of the 16K neurons an *independent* direction in R^128: 2.1M parameters.
  - The BDH-style claim of "one concept per neuron" needs those independent directions. With a Kronecker lift, neurons sharing i or j are strongly correlated.
- **Effect on the sparse code.** Top-k over products s_i·t_j (when the separable case dominates) mostly selects the cross of the top few i and top few j, plus sign-flipped pairs. That is a "grid" code determined by two small codes, so the effective number of distinct sparse patterns is much smaller than C(N,k). This undermines low interference between memories and use as an index key.
- **Compute.** A dense 128×16k int8 lift is 2.1M MACs per token per pass. At 20 tokens × 6 passes that is about 250M MACs per request, roughly 5-15 ms on one modern AVX2/AVX-512 core at 20-50 GMAC/s int8 (estimate, not measured). That is above the 0.7M-ops-per-pass budget, but not prohibitive.
  - A 16k-point fast Walsh-Hadamard transform is about 230K adds, so it saves roughly 9× compute in exchange for the expressivity loss.
- **Better structured options:**
  - **Monarch/BTT:** full rank, no parameter sharing. Rectangular 128→16k for roughly N·√D parameters, about 185K at N=16k. This is the structure with evidence of parity with dense.
  - **Product-key-style selection:** split the 128-dimensional input into two halves and score 128 sub-keys per half. The exact top-k over 16,384 combinations costs about 16K MACs, and the down-projection rows are then exactly PKM/PEER "values". This turns "structured lift + k-WTA + read only active rows" into a known, published mechanism (PKM, Memory+, PEER), with the caveat that PKM's gains needed large N and BatchNorm or QK-norm to keep slots used.
  - **Low-rank-plus-sparse:** no direct evidence found for lifts. Low-rank alone scales worse (ψ<1).
- **Recommended experiment** (ARCHITECTURE.md §7 already proposes "dense versus structured at equal parameters"). Compare at **equal FLOPs and equal parameters** separately:
  - dense lift N=4k-8k;
  - Monarch lift N=16k;
  - Kronecker/Hadamard lift N=16k;
  - product-key selection N=16k.
  Report the execution-accuracy and paraphrase splits, plus slot/neuron usage (dead-neuron rate) and the pairwise overlap of k-WTA codes.
  Use μP learning-rate scaling per Potapczynski, and watch for loss spikes (Wei et al.).
- **Verdict:**
  - Kronecker/Hadamard lift: **skip** as default; keep as one arm of the test.
  - Monarch/BTT lift: **test**.
  - Product-key selection: **test**; it is the most promising way to get a 16k-neuron sparse layer cheaply.
  - Dense lift at N about 4-8k: the baseline to beat.

### Gaps
- No published result on a Kronecker or Hadamard *expansion* followed by top-k sparsity in language models at any scale.
- No CPU inference benchmarks of Monarch or Kronecker layers at d=128. All speedups cited are GPU training.
- No source for low-rank-plus-sparse as an FFN lift in small LMs.

---

## Q4. k-winners-take-all and top-k activation sparsity: accuracy, robustness, compute, trainability

### Takeaway
k-WTA trains fine with straight-through gradients to the winners plus a duty-cycle "boosting" term, and it improves robustness to noise at equal clean accuracy (Numenta, on small CNNs). Real compute savings appear only with kernels that actually skip work. Numenta's sparse nets cut non-zero multiplies 10-35× but ran at the same speed; Google's Spark Transformer gets about 1.6-1.9× CPU wall time from top-k with about 8% active. **Adopt top-k sparsity as a test arm**, with boosting or an auxiliary load term, and a gather-based kernel that computes only active rows (as Reef plans).

### Cited Findings
- **Numenta, "How Can We Be So Dense?" (Ahmad & Scheinkman 2019):**
  - k-WTA replaces ReLU. The gradient is 1 for winners and 0 elsewhere. A boost coefficient from a duty cycle, β = 1.0-1.5, keeps units from dying.
  - On Google Speech Commands (10 seeds):

    | Network | Test accuracy | Noise score |
    |---|---|---|
    | Dense CNN-2 | 96.37 ± 0.37 | 8,730 ± 471 |
    | Dense CNN-2, dropout 0.5 | 95.69 | 7,681 |
    | Sparse CNN-2 | 96.65 ± 0.21 | 11,233 ± 1013 |
    | Super-sparse CNN-2 | 96.57 | 10,752 |

  - Sparsity levels are about 9-33% active units with 30% weight density.
  - Non-zero multiplies fell about 10.5× and 20× (35× for super-sparse), "but our implementations ran at the same speed as the corresponding dense networks".
  — [arXiv 1903.11257](https://arxiv.org/pdf/1903.11257)
- **Critique of the comparison:** forum readers noted the comparison is not size-matched (Sparse CNN-2 has 64 second-layer filters vs Dense CNN-2's 30). — [Numenta forum](https://discourse.numenta.org/t/how-can-we-be-so-dense-the-benefits-of-using-highly-sparse-representations/5693)
- **Spark Transformer** (Google, NeurIPS 2025): Gemma-2 recipe with about 8% of FFN neurons active per token, via "statistical top-k", a linear-time approximation that avoids sorting. Results: 2.5× FLOPs reduction, with CPU decode 141 → 86 ms/token and prefill 28 → 15 ms/token. That is 1.64× and 1.86× in v2; v1 reports up to 1.79× on CPU. — [arXiv 2506.06644](https://arxiv.org/abs/2506.06644v2)
- **Memory layers and PKM** are top-k sparse lookups. They needed QK-norm (Memory+) or BatchNorm (PKM) to keep slot usage high: at 1M slots, usage was 26% without BatchNorm vs 80% with it. — [ar5iv 1907.05242](https://ar5iv.labs.arxiv.org/html/1907.05242); [arXiv 2412.09764](https://arxiv.org/html/2412.09764v2)

### Inferences
- **Dead neurons are the main training risk.** Reef's k ≈ 1-2% of N is sparser than Numenta's (9-33%) or Spark's (8%), which makes them more likely. Plan duty-cycle boosting or a load-balancing auxiliary loss, and log the fraction of neurons that ever fire.
- **Savings on CPU need the right kernel.** They are real only if the down-projection gathers k rows (Reef's plan) *and* the pre-activation for all N neurons is cheap. This ties k-WTA to the lift choice in Q3: product-key selection makes both sides cheap.
- **Robustness evidence is from vision and audio CNNs**, not language. Treat "low interference" and "robustness" as hypotheses to measure.
- **Verdict: test.** Compare top-k against plain ReLU/SiLU at the same width.

### Gaps
- No published k-WTA result for small language models (under 50M) on understanding or parsing tasks.
- Numenta's MNIST Table 1 numbers were not extracted (only Speech Commands numbers are above).

---

## Q5. Confidence heads that predict correctness: best practice

### Takeaway
A head trained on whether outputs were correct ranks well in-distribution (AUROC about 0.77-0.79 for text-to-SQL), but drops sharply on unseen domains (about 0.66). Cheap execution signals alone plateau at about 0.6-0.68 AUROC. Needle combines a post-hoc head that reads the prompt and the produced call with the decode probability, taking the minimum. Its head is calibrated only for the base model, and Cactus warns it fails out of distribution ("correct Spanish calls have been measured at 0.0"). **Adopt** a head trained on sandbox outcomes. Its inputs: hidden states of the produced program, decode log-probability, agreement across beam candidates, and sandbox signals. Calibrate it on a held-out split, and evaluate on held-out *tool families*.

### Cited Findings
- **Needle's design:**
  - `confidence` is the minimum of a post-hoc calibration head (scoring "the full prompt together with the call the model just produced") and the decode probability of the call.
  - The engine withholds calls below 0.1. The suggested product threshold is about 0.7.
  - Fine-tuned weights lose the head and report `None`.
  - Non-English scores need caution: correct Spanish calls have scored 0.0.
  - Training labels, calibration method and metrics are not disclosed.
  — [Cactus: Needle confidence](https://cactuscompute.com/blog/needle-confidence); (source code, llms.txt)
- **Selective prediction for text-to-SQL** (arXiv 2607.06799, July 2026; correct = executes to the reference result):
  - Black-box signals (self-consistency, executability, schema relevance) reach about 0.61-0.68 AUROC; string self-consistency is best at 0.675.
  - Log-probability reaches about 0.67.
  - Fine-tuned verifiers reach 0.77-0.79 in-distribution but about 0.66 on unseen schemas. Scaling to 7B, schema diversity and distillation did not close the gap.
  - An ensemble of LLM judges reaches AUROC 0.82 and ECE 0.03, answering 27% of questions at 24% selective risk.
  — [arXiv 2607.06799](https://arxiv.org/abs/2607.06799)
- **Calibration and ranking are different properties.** Higher AUROC from hidden-state readers did not improve forced-choice selection in looped LMs. — [arXiv 2607.18553](https://arxiv.org/pdf/2607.18553) (via search snippet only; not fetched)
- **Conformal-style selective generation** with false-discovery-rate guarantees, using generated unit tests as supervision. — [arXiv 2505.13553](https://arxiv.org/html/2505.13553v1) (search snippet only)
- **Kadavath et al. 2022** trained P(IK) ("I know") heads and self-evaluation P(True); they calibrate well in-distribution and transfer only partly across tasks. — (prior knowledge, not re-fetched) https://arxiv.org/abs/2207.05221

### Inferences
- **Train on the real label:** sandbox execution matched the reference effect, plus a "did more than asked" penalty label. That label is better grounded than SQL text-match labels. Inputs:
  - pooled hidden states over request and program (Needle's probe pooling across all depths is cheap and lets the head use early-exit states);
  - candidate log-probability and margin to the next beam;
  - agreement among the top-n candidates' execution results (execution self-consistency);
  - sandbox flags (exit code, empty output, files touched);
  - grammar path (typed vs open bash form).
- **Combine with the decode probability using Needle's `min`**, so that either signal can veto.
- **Calibrate after training** (temperature or isotonic) on a held-out split, and **re-calibrate after every fine-tune.** Needle shows a head becomes invalid once the weights change.
- **Report:**
  - AUROC, for ranking;
  - ECE, for calibration;
  - the risk-coverage curve, i.e. the accuracy when only the most confident x% are offered;
  - the same metrics on held-out tool families and other people's wording, where the SQL study predicts a sharp drop.
  Expect a learned head to be a ranking aid in-domain and a weak guard out of domain. The sandbox and guards remain the safety mechanism.
- **Verdict: adopt**, with the router head (typed vs open form vs decline) as the same probe-head design.

### Gaps
- No primary source found on confidence or verifier heads for models under 50M, or for NL-to-bash specifically.
- Needle's head training and calibration metrics are undisclosed.

---

## Summary verdicts for Reef

| Component | Verdict | Main reason |
|---|---|---|
| QK-norm | Adopt | Training diverged without it (Cactus); small memory models need it (Meta) |
| Sigmoid output gate on attention | Adopt | Near-free; positive reports (Qwen, prior knowledge) |
| 3-tap q/k/v causal convolution | Adopt | Negligible cost; used by Needle and Gated DeltaNet; Engram ablation shows small effect |
| Scalar residual gates | Skip | Neutral per Cactus |
| Ladder / deep supervision at every pass with KL to final pass | Adopt | +3.9% loss at full depth and usable early exits (short 3k-step study) |
| Engram hashed 2/3-gram tables (one early site, gated, prime sizes, plus 1-gram) | Test | Gains at large scale are mostly factual; small tables (16k) gave about 1% in PKM; robustness risk on paraphrase |
| Product-key memory as a large memory | Skip | Needs 10^5+ slots to pay off |
| Product-key-style top-k as the sparse lift | Test | Structured, exact top-k over N at √N cost; published mechanism |
| Kronecker/Hadamard lift 128 → 16k | Skip as default; one test arm | About 3K free parameters for 16K neurons; parameter-sharing structures scale worse; no evidence for lifts |
| Monarch/BTT lift | Test | Full rank, no sharing; matches dense in GPT-2 |
| Low-rank lift | Skip | Lower rank gives worse scaling |
| k-WTA (k about 1-2%) | Test | Trains with boosting; robustness shown on CNNs only; CPU gain needs gather kernels |
| Confidence head on sandbox outcomes, min with decode probability, recalibrated per fine-tune | Adopt | Best practice; expect out-of-domain drop |
| mHC residual lanes | Skip for now | No published ablation |
