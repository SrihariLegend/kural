# Linear attention, delta-rule memories and fast weights / test-time learning, for Reef (kural)

Scope note: written for the Reef design in /home/srihari/Projects/kural/docs/ARCHITECTURE.md (few-M parameters, CPU-only,
requests of 44 tokens or fewer, long sessions, delta-rule memory plus exact softmax pointer attention over the request,
short convolution, sigmoid gates, fast weights per session, offline consolidation). Numbers below were read from the
primary papers (arXiv HTML) in November 2026 unless marked **[unverified]**. Most published evidence is at 125M-1.3B
parameters; evidence under 50M is scarce and is called out wherever it exists.

## Q1. Which linear-attention variants do best on recall-heavy tasks at small scale, how large is the gap to softmax, and what do hybrids gain?

### Takeaway
Among pure linear/recurrent layers, the delta rule (DeltaNet, Gated DeltaNet, RWKV-7, which uses a generalized delta rule) recalls clearly better than additive or decay-only models (linear attention, RetNet, GLA, Mamba/Mamba-2). All of them still trail softmax attention badly on recall-heavy tasks (for example FDA 17.2 vs 27.4 at 1.3B). Adding a few softmax layers (about 10% of layers, or 2 global layers) closes the gap and often goes past the pure transformer. The only result I found under 50M parameters (Schlag et al., about 40M) shows the delta rule beating additive linear attention and coming close to softmax on perplexity.

### Cited Findings
**Fast weight programmers / DeltaNet origin (Schlag, Irie, Schmidhuber 2021)**
- Linearised attention is formally equivalent to 1990s Fast Weight Programmers: a slow net writes outer products of keys and values into fast weights. The paper replaces the additive write with a delta-rule write with a learned, input-dependent rate β (sigmoid) — [Schlag et al. 2021](https://arxiv.org/abs/2102.11174)
- Small-scale WikiText-103, about 40M parameters (D=128, L=256): Linear Transformer (sum rule) 37.1/38.3 valid/test ppl, Delta Network 34.1/35.5, Performer sum 39.0/39.6, Performer delta 36.1/37.2. Medium, about 90M: Linear 31.1/33.0, Delta 29.7/31.5. The softmax Transformer row is garbled in the HTML; it reads approximately 33.0/34.1 (small) and 27.9/29.6 (medium) **[approximate]**. The delta update adds only 16K / 33K parameters — [Schlag et al. 2021, Table 2](https://arxiv.org/html/2102.11174v3)
- Unbounded context (medium config): Delta Network 27.8/29.4 ppl with a 0.13M-number state, while the sum-rule linear transformer blew up to over 260 ppl. Transformer-XL with the same 0.13M state gets 65.5 test ppl and needs a 6.29M state to reach 25.5 — [Schlag et al. 2021, Table 4](https://arxiv.org/html/2102.11174v3)

**DeltaNet (Yang et al. 2024, parallel training)**
- β_t = σ(W_β x_t) ∈ (0,1). β=1 fully replaces the old value at key k; β=0 leaves memory unchanged — [Yang et al. 2024](https://arxiv.org/html/2406.06484)
- On the MAD synthetic suite (averages): Transformer 74.5, DeltaNet 71.8, Mamba 69.3, GLA 60.0. DeltaNet scores 100 on in-context recall, noisy recall and selective copy, and 35.7 on fuzzy recall (Transformer 29.8, Mamba 6.7, GLA 6.9). It is weaker on Memorize (52.8 vs 85.2) — [Yang et al. 2024, Fig. 3](https://arxiv.org/html/2406.06484)
- MQAR: DeltaNet "performs perfectly" in the hardest setting (sequence 512, 64 KV pairs) and beats Mamba at low model dimension. Exact values are only in a plot — [Yang et al. 2024](https://arxiv.org/html/2406.06484)
- 340M/15B tokens (Wiki ppl; SWDE/SQuAD/FDA recall): Transformer++ 28.39; 42.2/22.1/21.4. Mamba 28.39; 12.4/23.0/2.1. GLA (conv) 29.47; 24.0/24.7/7.3. DeltaNet (conv) 28.24; 26.4/28.9/12.8. DeltaNet + sliding attention 27.06; 39.3/32.5/18.8. DeltaNet + 2 global attention layers 27.51; 42.9/32.1/23.1 — [Yang et al. 2024, Table 1](https://arxiv.org/html/2406.06484)
- 1.3B/100B tokens: Transformer++ 16.85; 66.6/31.5/27.4. Mamba 17.06; 41.4/35.2/6.2. GLA (conv) 17.25; 52.4/37.4/22.3. DeltaNet 16.87; 49.5/37.4/17.2. DeltaNet + 2 global attention layers 16.55; 71.0/43.0/29.8, beating the transformer on all three — [Yang et al. 2024, Table 1](https://arxiv.org/html/2406.06484)

**Gated DeltaNet (Yang, Kautz, Hatamizadeh; ICLR 2025)**
- Update rule: S_t = S_{t−1}·α_t(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ, with data-dependent decay α_t ∈ (0,1) and β_t ∈ (0,1) (it can be set in (0,2) to allow negative eigenvalues). "Gating enables rapid memory erasure while the delta rule facilitates targeted updates" — [Yang et al. 2024b](https://arxiv.org/html/2412.06464v3)
- S-NIAH at 1.3B (1K/2K/4K/8K). Passkey: DeltaNet 97.4/96.8/99.0/98.8, Mamba2 99.2/98.8/65.4/30.4, Gated DeltaNet 98.4/88.4/91.4/91.8. Number: DeltaNet 98.4/45.6/18.6/14.4, Mamba2 99.4/98.8/56.2/17.0, Gated DeltaNet 100/99.8/92.2/29.6. UUID (1K/2K/4K): DeltaNet 85.2/47.0/22.4, Mamba2 64.4/47.6/4.6, Gated DeltaNet 86.6/84.2/27.6 — [Gated DeltaNet, Table 2](https://arxiv.org/html/2412.06464v3)
- Recall-intensive average at 1.3B (SWDE, SQuAD, FDA, TriviaQA, NQ, Drop): Transformer++ 37.0, Mamba2 29.8, DeltaNet 26.2, Gated DeltaNet 30.6, Samba 37.3, Gated DeltaNet-H1 39.0, Gated DeltaNet-H2 40.1 — [Gated DeltaNet, Table 4](https://arxiv.org/html/2412.06464v3)
- Commonsense zero-shot average: Transformer++ 52.25, RetNet 52.02, HGRN2 51.79, Mamba 53.12, Mamba2 54.89, DeltaNet 52.14, Gated DeltaNet 55.32, H1 56.40, H2 56.18. Models were 400M and 1.3B on 100B FineWeb-Edu tokens with a 4K training length; the hybrids use a 2K sliding window — [Gated DeltaNet, Table 3](https://arxiv.org/html/2412.06464v3)
- Throughput is "essentially the same" as DeltaNet and about 2-3K tokens/s slower than Mamba2 on an H100 at 1.3B — [Gated DeltaNet](https://arxiv.org/html/2412.06464v3)

**Zoology / MQAR (Arora et al. 2023)**
- MQAR: many key→value queries against pairs given earlier in the context — [Zoology](https://arxiv.org/html/2312.04927)
- Associative-recall "hit" tokens are 6.4% of Pile tokens but explain 82% of the perplexity gap between gated convolutions and attention. A 70M attention model beats a 1.4B Hyena on the AR slice (2.41 vs 3.43 ppl) — [Zoology](https://arxiv.org/html/2312.04927)
- Small synthetic (2-layer models, d and N swept from 64 to 512, vocabulary 8,192): attention solves MQAR at every sequence length with d=64. Gated convolutions (RWKV, H3, Hyena, BaseConv) stay below 0.9 accuracy unless d ≥ N. Input-dependent sparse-attention hybrids close 97.4% of the gap — [Zoology](https://arxiv.org/html/2312.04927)

**Based (Arora et al. 2024): recall vs state-size tradeoff**
- A sliding window of 64-128 tokens plus Taylor linear attention recovers 90.8% of full softmax recall accuracy at 1e-5× its latency — [Based](https://arxiv.org/html/2402.18668)
- Real recall tasks at 360M/30B tokens (SWDE/FDA): Transformer++ 70.75/63.79, Based 45.01/16.45, Mamba 27.63/8.71. At 1.3B/50B: Transformer++ 76.50/80.47, Based 64.45/30.40, Mamba 52.75/18.51. Even the best sub-quadratic model sits far below softmax on FDA — [Based, Table 1](https://arxiv.org/html/2402.18668)

**Repeat After Me (Jelassi et al. 2024): copying**
- Theorem: a fixed-state model with state space 𝒮 has error > 1 − |𝒮|/D^L on uniform copying of length L. Below L·log D − 1 bits of state, error exceeds 1/2. A 2-layer transformer can copy strings of exponential length — [Jelassi et al.](https://arxiv.org/html/2402.01032v2)
- About 160M models: transformers need 100× fewer samples than the best GSSM to learn copying. Hard-ALiBi transformers generalize almost perfectly from length ≤50 to 1000, while Mamba and LSTM fall to near zero. Pythia-410M beats Mamba-2.8B on phone-book lookup once the book has ≥70 entries — [Jelassi et al.](https://arxiv.org/html/2402.01032v2)

**Mamba-2 (Dao & Gu 2024)**
- On MQAR, Mamba-2 beats Mamba-1 even at state N=16, and N=16→64→256 improves it steadily (plots only, on small models of unstated size) — [Mamba-2](https://arxiv.org/html/2405.21060)
- Hybrid sweep at 350M/48 layers/7B tokens (ppl vs number of attention layers): 0 → 8.60, 1 → 8.38, 6 → 8.26 (best, about 10%), 24 → 8.50; Transformer++ 8.68. At 2.7B/300B: Mamba-2 6.09 ppl / 60.2 avg; 58 SSD + 6 attention 5.95 / 61.0; Transformer++ 6.13 / 60.2 — [Mamba-2, Tables 2-3](https://arxiv.org/html/2405.21060)
- SSD is 2-8× faster than Mamba-1's scan and supports states about 8× larger — [Mamba-2](https://arxiv.org/html/2405.21060)

**RWKV-7 "Goose" (Peng et al. 2025)**
- Generalized delta rule with vector-valued gating, in-context learning rates and relaxed value replacement. Models range from 0.19B to 2.9B on 3.1T tokens. The 2.9B claims 3B-class SOTA on multilingual tasks and parity on English. The authors claim it does state tracking and recognizes all regular languages (beyond TC⁰ transformers under standard conjectures) — [RWKV-7](https://arxiv.org/abs/2503.14456)

**RetNet, GLA, Griffin, Jamba, Samba**
- RetNet (fixed scalar decay, additive write) scores 52.02 commonsense in the Gated DeltaNet comparison, below Gated DeltaNet's 55.32. It has no separate recall table there — [Gated DeltaNet, Table 3](https://arxiv.org/html/2412.06464v3)
- GLA (data-dependent gated additive write) is weak on MAD fuzzy recall (6.9) and FDA at 340M (7.3) — [Yang et al. 2024](https://arxiv.org/html/2406.06484)
- Samba (Mamba + sliding-window attention) averages 37.3 on recall at 1.3B, on par with Transformer++ 37.0 — [Gated DeltaNet, Table 4](https://arxiv.org/html/2412.06464v3)
- Griffin (gated linear recurrence + local attention, Google DeepMind) and Jamba (Mamba + attention 1:7 + MoE, AI21): I did not fetch these in this pass. Descriptions are from background knowledge **[unverified]**.

### Inferences
- Ranking for recall in a pure fixed-size state: delta-rule family (Gated DeltaNet ≈ DeltaNet > RWKV-7, which is likely similar but not compared head to head here) > Mamba-2 > GLA/RetNet > plain linear attention. The delta rule's advantage is largest exactly where Reef needs it: corrections and overwriting a key's value.
- The gap to softmax is large and does not close with scale: on FDA at 1.3B, DeltaNet scores 17.2 vs 27.4 for the transformer, and Based 30.4 vs 80.5. For anything Reef must reproduce exactly from the session (file names, paths, an earlier command), the delta memory alone will be unreliable.
- Hybrids are the consistent winner. Recall becomes equal to or better than the transformer with about 10% softmax layers, or 2 global layers, or a sliding window. Reef's exact pointer attention over the request is already a hybrid. The open risk is session-level exact recall ("do the same to the file from before"), which the request-only pointer cannot reach.
- **CPU cost reality check (my arithmetic):** exact softmax attention over a session buffer of 2,000 tokens at head width 64 costs about 2·2000·64 ≈ 256k multiply-adds per query token. Reef's own budget gives the memory read/write about 130k per token per pass. A KV cache for 2,000 tokens × 64 dims × K and V in int8 is about 256 KB. At Reef's session lengths (thousands of tokens, not millions), exact attention over a bounded session buffer is affordable on CPU, so the fixed-state argument for the session is weaker than at LLM scale.
- Small-scale evidence: the only sub-50M language result is Schlag's about 40M WikiText-103 (delta beats sum by about 2.8 ppl and sits about 1.4 ppl behind softmax **[approximate]**). Zoology's tiny 2-layer MQAR models show softmax needs far less width than recurrent models for recall, which matters even more at D=128.

### Gaps
- No published Gated DeltaNet, Mamba-2 or RWKV-7 results under 50M parameters on real recall tasks. The fla repo has small configs, but I found no benchmark tables for them.
- Exact MQAR numbers for DeltaNet, Mamba-2 and Based are only in figures.
- Griffin and Jamba numbers were not verified in this pass.
- No head-to-head of RWKV-7 vs Gated DeltaNet on recall from a primary source.

## Q2. Is a short convolution before linear attention necessary or helpful?

### Takeaway
It is helpful for delta-rule and memory-style layers, but not universally. It improved DeltaNet at 340M (28.24 vs 29.08 ppl), and removing it hurt Titans (27.01 → 28.73 ppl). GLA without a convolution was slightly better on perplexity (28.65 vs 29.47). It is cheap, so test it in Reef, expecting a gain.

### Cited Findings
- 340M DeltaNet: with convolution 28.24 ppl / 42.1 avg; without 29.08 / 41.3. Recall: FDA 12.8 vs 4.5, SWDE 26.4 vs 24.6 — [Yang et al. 2024, Table 1](https://arxiv.org/html/2406.06484)
- 340M GLA: with convolution 29.47 ppl, without 28.65 (without is better). At 1.3B GLA w/o conv 17.22 vs with 17.25, with mixed recall (SWDE 52.4 with vs 50.6 without; SQuAD 37.4 vs 42.6) — [Yang et al. 2024, Table 1](https://arxiv.org/html/2406.06484)
- Titans LMM ablation: full 27.01 ppl / 92.68 long-context accuracy; without convolution 28.73 / 90.28 — [Titans, Table 5](https://arxiv.org/html/2501.00663v1)
- Based: short convolutions (filter size 3) add benefit beyond the sliding window. No numbers in the extracted text — [Based](https://arxiv.org/html/2402.18668)
- Key normalization matters as much as the convolution. DeltaNet with L1-norm keys: 31.12 ppl, FDA 6.2. With L2-norm: 28.03, FDA 13.1. SiLU was the best feature map. Schlag's delta networks diverged without normalization — [Yang et al. 2024](https://arxiv.org/html/2406.06484); [Schlag et al. 2021](https://arxiv.org/html/2102.11174v3)

### Inferences
- The convolution likely gives keys and values local context (bigram-like "previous token" information that induction-style recall needs), which a single-layer recurrent memory cannot form otherwise. That fits it helping delta and memory layers on recall most (FDA 4.5 → 12.8).
- For Reef: keep the 3-tap convolution on q/k/v and measure it. More critical is that keys entering the delta write are L2-normalized. A k-WTA binary key with k active units has ‖k‖² = k, so writing it unnormalized with β near 1 would multiply S by an eigenvalue 1 − βk along k, which is unstable for k > 2.

### Gaps
- No convolution ablation under 50M parameters, and none for Gated DeltaNet specifically (its ablation appendix was not in the extracted text).

## Q3. Have TTT, Titans or similar shown benefits from learning at inference, at what scale and cost? Is session-continual fast-weight learning sound, and what are its failure modes?

### Takeaway
TTT, Titans and TTT-E2E all show gains from learning *within one sequence* at 125M-3B parameters. All of them reset (or are only evaluated within a sequence) between sequences. I found no primary evidence of fast weights carrying over across sessions or tasks with measured benefit. Learning within a session is the same mechanism as long-context linear attention, so it is sound if Reef trains on session-length sequences. Persisting fast weights across sessions, and Reef's outcome-scaled β, are unvalidated. Reef's choices to freeze slow weights and gate consolidation offline on test sets are the conservative, defensible part.

### Cited Findings
- TTT (Sun et al. 2024): the hidden state is a model (linear in TTT-Linear, a 2-layer MLP in TTT-MLP), updated by a self-supervised gradient step per token or mini-batch. Evaluated at 125M-1.3B. TTT-Linear and TTT-MLP keep improving perplexity with more context, like transformers, while Mamba plateaus after 16K. TTT-MLP has memory-I/O cost problems — [TTT](https://arxiv.org/abs/2407.04620)
- Titans (Behrouz, Zhong, Mirrokni 2025): memory M is updated at test time with loss ‖M(k_t) − v_t‖², "surprise" = gradient, momentum η_t, step size θ_t and forgetting gate α_t: M_t = (1−α_t)M_{t−1} + S_t, S_t = η_t S_{t−1} − θ_t∇ℓ. With a linear M and no momentum, this reduces to a gated delta rule — [Titans](https://arxiv.org/html/2501.00663v1)
- Titans at 340M/15B tokens: Transformer++ 31.52 ppl, Gated DeltaNet 27.01, TTT 27.44, Titans LMM 26.18, Titans MAC 25.43. S-NIAH passkey at 16K: LMM 96.2, MAC 98.4, TTT 88.4, Mamba2 5.4 — [Titans](https://arxiv.org/html/2501.00663v1)
- Titans ablation (ppl / long-context accuracy): full 27.01/92.68; without momentum 28.98/87.12; without weight decay (forgetting) 29.04/85.60; linear memory instead of MLP 28.49/85.34. The authors rank forgetting as the most important component — [Titans, Table 5](https://arxiv.org/html/2501.00663v1)
- Titans' memory module is slightly slower than Mamba2 and Gated DeltaNet in training throughput, and deeper memories are slower still — [Titans](https://arxiv.org/html/2501.00663v1)
- The Titans paper does not state explicitly whether memory resets between sequences. It updates weights "even at test time" and carries memory across segments within a sequence — [Titans](https://arxiv.org/html/2501.00663v1)
- TTT-E2E (Dec 2025): a sliding-window transformer that does next-token-prediction gradient steps on selected MLP weights at test time, with initialization meta-learned. At 3B/164B tokens it scales with context like full attention, while Mamba 2 and Gated DeltaNet do not, and it is 2.7× faster than full attention at 128K. Fast weights are reset per sequence (per a course write-up; the paper itself not read in full) — [TTT-E2E](https://arxiv.org/abs/2512.23675v1); [course note](https://barisdeniz.is-a.dev/courses/continual-learning/tracks/nested-learning/nl4-end-to-end-ttt.html)
- Failure modes visible in the data:
  - Saturation and interference as the number of stored items grows: DeltaNet S-NIAH-2 falls 98.4 → 14.4 from 1K to 8K — [Gated DeltaNet](https://arxiv.org/html/2412.06464v3)
  - Decay trades long retention for capacity: Gated DeltaNet's passkey at 2K (88.4) is below DeltaNet's (96.8) — [Gated DeltaNet](https://arxiv.org/html/2412.06464v3)
  - Divergence without key normalization — [Schlag et al.](https://arxiv.org/html/2102.11174v3)
  - Unbounded additive memories blow up on unlimited context (over 260 ppl) while the delta rule holds — [Schlag et al.](https://arxiv.org/html/2102.11174v3)
  - Poor length generalization of fixed-state models beyond training length — [Jelassi et al.](https://arxiv.org/html/2402.01032v2)

### Inferences
- **On Reef's write S ← (1−λ)S + β(v − S·k)⊗k:**
  - It equals S((1−λ)I − βkkᵀ) + βvkᵀ, nearly Gated DeltaNet's αS(I − βkkᵀ) + βvkᵀ (Gated DeltaNet also decays the erase term). It is stable for unit-norm k when 0 ≤ β ≤ 2 − λ.
  - With constant λ it is RetNet-style decay. The evidence (Gated DeltaNet, Titans' forgetting ablation) favors a *data-dependent* α_t. Make λ per token (a sigmoid gate), not a fixed slow constant.
  - "β = importance × surprise" double-counts surprise: the delta term already writes only the residual v − Sk. A learned β_t = σ(W x_t), optionally times an external trust/outcome factor, matches what was validated.
  - Pruning links below a threshold, 8-bit storage of S, and updating only k active rows have no published validation that I found. Small β updates can round to zero in 8 bits, which silently disables "loosen rather than erase" low-β writes. Use a higher-precision accumulator or stochastic rounding, and test.
- **Within-session learning:** sound. It is the same computation as linear attention over a long context, provided training uses sequences shaped like sessions (many requests, outputs, retrieved entries) of realistic length. Otherwise Reef hits the length-generalization failure from Repeat After Me.
- **Across sessions (saved fast weights):** unvalidated. Main risks:
  - Stale or contradictory associations (a file renamed, a project changed).
  - Drift: the slow weights were trained to read states from fresh starts, not states that are weeks old.
  - Poisoning: a crafted man page, README or command output written with β > 0 can plant a key→value association. Low β reduces this but does not remove it; repeated exposure accumulates.

  Mitigations:
  - Cap the session state's age and size, and keep a reset path.
  - Keep provenance: separate states, or at least separate heads, for trusted user corrections and untrusted retrieved text.
  - Never let fast-weight content override the grammar or guards. Reef's guards and sandbox already provide this.
- The third factor (outcome scales β, applied after the fact) has no primary evidence I found. It is plausible as a supervised write (key = request code, value = the program the user accepted), but test it as its own ablation step.
- For exact session facts, an explicit store (session file with exact strings, plus exact attention or retrieval over it) is safer and cheaper on CPU than relying on the delta state. Use the delta state for soft preferences and patterns.

### Gaps
- No primary source found on poisoning or adversarial writes into delta-rule or TTT memories.
- No primary source on fast weights persisting across sessions or tasks with measured benefit (continual-learning claims in Titans and TTT-E2E are within-sequence).
- No TTT or Titans results under 125M parameters, and no CPU latency measurements.

## Q4. Capacity of a fixed-size state for N items; effect of sparse keys (k-WTA) on interference

### Takeaway
A linear or delta memory with key dimension d_k holds at most about d_k orthogonal associations. Measured: errors start at about 60 keys for d_k = 64. The delta rule makes better use of that capacity (it overwrites instead of piling up) but does not raise the ceiling. Raising the effective key dimension (DPFP feature expansion, or Reef's lift to N sparse units) raises capacity in proportion (DPFP failures at about 128/256/384). Softmax/modern Hopfield retrieval has exponential capacity in d. Sparse keys cut crosstalk but carry no published small-model linear-attention validation that I found.

### Cited Findings
- Linear attention holds at most d_dot orthogonal keys, and "storing more than d_dot associations will result in a retrieval error." With d_key = 64, linear attention (d_dot = 64) accumulated errors from about 60 associations. DPFP-1/2/3 (d_dot = 128/256/384) failed near 128/256/384. FAVOR+ never reached zero loss. Softmax beat all of them but struggled above 500 keys — [Schlag et al. 2021, Sec. 4.1, 6.1.1](https://arxiv.org/html/2102.11174v3)
- Modern Hopfield networks: with a softmax energy, the retrieval update equals scaled dot-product attention, and capacity is exponential in the dimension (Ramsauer et al. 2021). Classical (linear/quadratic energy) Hopfield nets store on the order of one pattern per dimension — [summary via search; primary: Ramsauer et al., "Hopfield Networks is All You Need", arXiv 2008.02217, not fetched] **[partially unverified]**
- Sparse/structured Hopfield networks (Santos et al.) prove exact retrieval with exponential capacity for well-separated patterns using sparse transformations (Fenchel-Young losses) — [Santos et al. 2024](https://arxiv.org/pdf/2402.13725)
- Linear attention's recurrence is equivalent to a continuously updated classical Hopfield network, so its capacity should scale with dimension, not exponentially (a blog-level argument) — [Millidge note via gwern](https://gwern.net/doc/www/www.beren.io/b158cf2f242ef5b70de93bb3a8d5c3a20229097f.html) **[secondary]**
- The Mamba-2 state size N = 16 → 64 → 256 consistently improves MQAR — [Mamba-2](https://arxiv.org/html/2405.21060)
- The Repeat After Me bound (state bits ≥ L·log D for exact copying of length L) means no fixed state copies arbitrary long session content exactly — [Jelassi et al.](https://arxiv.org/html/2402.01032v2)

### Inferences
- **Reef sizing (my arithmetic, unverified by experiment):**
  - With keys as k-of-N binary codes (N = 32k, k = 512, L2-normalized), two random codes share k²/N ≈ 8 active units on average, a cosine of about 1.6%.
  - Crosstalk from n stored items grows roughly as sqrt(n)·cos, giving about 0.16 at n = 100 and about 0.5 at n = 1,000. Recall stays reasonable to roughly hundreds of items per head, well above a dense 128-dim key (about 128 items).
  - Only about N/k = 64 codes can be perfectly disjoint, so learned codes that cluster (similar requests) will interfere far more than random ones. This is precisely where the delta rule's overwrite matters.
- Sparse keys make each write touch only k rows, which cuts compute and keeps unrelated memories separate. Classical sparse associative memories (Willshaw/Kanerva-style) support high capacity with sparse codes, but I did not fetch sources for this here **[unverified]**.
- Reef state size: N × d_v at 32k × 128 = 4.2M numbers per head. That is larger than the whole model, and about 4 MB per head per session in int8. Even row-sparse, it may grow large over long sessions. Consider keys in a smaller space (for example d_k = 1-4k sparse) or count active rows.

### Gaps
- No primary experiment found on k-WTA or sparse binary keys inside delta-rule linear attention for language. BDH's own claims were not covered here.
- No closed-form capacity result for the delta rule with decay (as opposed to the pure additive rule).

## Q5. Verdicts for Reef (adopt / test / skip), with CPU cost

### Takeaway
Adopt a gated delta rule (Gated DeltaNet form, data-dependent α and β, L2-normalized keys, short convolution) as the session memory. Keep and possibly widen the exact softmax pointer attention (request plus a bounded exact session buffer), because recall/copy gaps are large and CPU cost at Reef's lengths is small. Test sparse keys, pruning, int8 state and persistence across sessions. Skip MLP-memory TTT and Titans and pure decay-only layers.

### Cited Findings
- Delta-rule family > Mamba-2 > GLA/RetNet on recall; hybrids reach or beat transformers — [DeltaNet](https://arxiv.org/html/2406.06484); [Gated DeltaNet](https://arxiv.org/html/2412.06464v3); [Mamba-2](https://arxiv.org/html/2405.21060)
- MLP memory (TTT-MLP, deep Titans memory) costs more I/O and throughput than linear memory — [TTT](https://arxiv.org/abs/2407.04620); [Titans](https://arxiv.org/html/2501.00663v1)

### Inferences
Verdict table (CPU cost per token, for one head, is my arithmetic):

| Technique | Verdict | Reason | CPU inference cost |
|---|---|---|---|
| Gated DeltaNet (gated delta rule) | **Adopt** | Best evidence among fixed-state layers for recall and overwriting. Gating fixes DeltaNet's saturation. Titans' forgetting ablation shows gating matters | Read d_k·d_v plus write about 2·d_k·d_v multiply-adds. Constant in session length |
| DeltaNet (ungated) | Test (as baseline) | Better passkey retention than Gated DeltaNet at 2K, but saturates (S-NIAH-2 14.4 at 8K) | Same as above |
| Short 3-tap convolution on q/k/v | **Adopt, confirm by ablation** | +0.84 ppl and FDA 4.5 → 12.8 for DeltaNet at 340M; also helps Titans | 3·D per channel, negligible |
| L2-normalized keys, SiLU feature map | **Adopt** | L1 → L2: 31.12 → 28.03 ppl; unnormalized writes diverge | Negligible |
| Exact softmax pointer attention over the request | **Adopt** | Copying gap is fundamental (Repeat After Me bound) | At most 44 positions: about 11k multiply-adds per token |
| Exact attention over a bounded session buffer (hybrid) | **Test, high priority** | Hybrids close or reverse the recall gap (DeltaNet + 2 global attention layers: SWDE 71.0 vs 66.6). "Same file as before" needs exact session recall | 2k tokens × 64 dims ≈ 256k multiply-adds; about 256 KB int8 KV |
| RWKV-7 generalized delta rule | Test (only if Gated DeltaNet underperforms) | Richer vector gates and state-tracking claims; no small-scale recall head-to-head found | Similar to Gated DeltaNet |
| Mamba-2 / SSD | Skip as main memory | Weaker recall than delta (S-NIAH passkey 30.4 at 8K) | Similar |
| GLA, RetNet, plain linear attention | Skip | Additive writes cannot overwrite; weakest recall | Similar |
| TTT-MLP / deep Titans memory | Skip | Gains at 340M+ but I/O-heavy. A linear memory with momentum is the cheap part worth borrowing | Several times the linear cost |
| Titans momentum on the delta write | Test (low priority) | Without momentum: 28.98 vs 27.01 ppl in Titans | One extra state-sized buffer |
| k-WTA sparse keys | Test | Plausible crosstalk reduction; no primary evidence in linear attention | Writes touch k rows only |
| Pruning, int8 state | Test | Unvalidated; int8 may erase small-β writes | Saves memory |
| Fast weights within a session | **Adopt** (train on session-length sequences) | Equivalent to long-context linear attention, which is well validated | Constant |
| Fast weights saved across sessions | Test, behind a reset and provenance split | No primary evidence; poisoning, staleness and drift risks | State size per session file |
| Slow weights frozen; offline gated consolidation | **Adopt** | Avoids catastrophic forgetting and poisoning of slow weights; no counter-evidence | Offline only |

### Gaps
- No CPU latency benchmarks for any of these layers at few-M parameters. All costs above are operation counts, not measurements.
- The fla repo's own benchmarks were not fetched in this pass.
