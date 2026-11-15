# CPU-efficient inference techniques for Reef (kural): NNUE, ternary/BitNet, int8 W8A8 with AVX-VNNI, Hadamard-rotation quantization, logic gate networks, VSAs

Context checked locally: `kural/engine/src/kernels.rs` handles int8 weights (`W::I8`) by widening each int8 to int32 and then to float32 (`_mm256_cvtepi8_epi32` then `_mm256_cvtepi32_ps`) and doing a float FMA. That is dequantize-in-float with a scale per row. It is not integer arithmetic, so on the hot loop it costs about two extra conversion instructions per 8 weights compared with one `vcvtph2ps` for float16. This probably explains why the int8 weight-only variant was about 30% slower (my reading of the code; not measured). ARCHITECTURE.md (section 5) puts today's model at about 80M operations per request. At about 1.3 ms that is about 60 G multiply-adds per second, which is already close to one core's float32 FMA peak (2 FMA ports × 8 lanes × about 5 GHz ≈ 80 G multiply-adds per second; my arithmetic, not measured). So the matrix work, not memory, is probably the bottleneck. That is the regime where integer dot products help.

## Q1. How much faster could int8 activations with AVX-VNNI be, and what accuracy loss is typical at small scale?

### Takeaway
W8A8 (int8 weights and int8 activations, int32 accumulation through `vpdpbusd`) gives about 4× more multiply-adds per instruction than float32 FMA. Published model-level CPU speedups are about 1.5-4× on client chips and up to 5.8× on servers, and accuracy loss on small BERT-class encoders and Transformer MT is negligible (under 0.5 BLEU). For kural, a realistic gain is about 2-3× on the matrix-multiply part, *if* the kernel is rewritten to do integer math (today's int8 path does not). Verdict: **TEST, high priority.**

### Cited Findings
- The NNUE reference implementation's int8 dense layer uses `_mm256_dpbusd_epi32(acc, a, b)` when VNNI is available. Otherwise it falls back to `_mm256_maddubs_epi16`, then `_mm256_madd_epi16` with ones, then `_mm256_add_epi32`. It handles 32 int8 inputs per chunk and 4 output rows per unrolled iteration, then does a horizontal sum (`m256_haddx4`). Activations are unsigned int8 (0..127 after ClippedReLU) and weights are signed int8, which is the u8×s8 form `vpdpbusd` needs. — [nnue-pytorch docs](https://raw.githubusercontent.com/official-stockfish/nnue-pytorch/master/docs/nnue.md)
- The NNUE quantization recipe: hidden-layer weights are int8 with scale 64 (maximum weight 127/64 ≈ 1.98, step 1/64), biases are int32 with scale s_A·s_W, accumulation is int32, and the result is shifted right by log2(weight scale). — [nnue-pytorch docs](https://raw.githubusercontent.com/official-stockfish/nnue-pytorch/master/docs/nnue.md)
- VNNI was kept on Alder Lake client cores as the 256-bit AVX-VNNI even though AVX-512 was disabled. The theoretical peak is 4× the inputs of fp32 per instruction, with 1/4 the memory. — [nightr gitbook "Quant and AVX"](https://nightr.gitbook.io/nnopt/quantization/3_quant_avx); [hwcooling Golden Cove analysis](https://www.hwcooling.net/en/intel-alder-lake-golden-cove-cpu-core-unveiled-microarch-analysis/3/)
- Intel says "with int8 ... VNNI technology delivers 256 int8 operations per cycle per core". This is a vendor figure, and it is not stated which vector width it refers to. — [Intel Architecture Day 2021 performance index](https://edc.intel.com/content/www/jp/ja/products/performance/benchmarks/architecture-day-2021/)
- OpenVINO's INT8-vs-FP32 tables show model-level speedups of about 1.5× to about 4× on VNNI-capable Core CPUs. The table layout is hard to map to CPUs, so treat these as indicative. — [OpenVINO 2021.4 INT8 vs FP32](https://docs.openvino.ai/2021.4/openvino_docs_performance_int8_vs_fp32.html)
- Xia, Cui, Cao (arXiv 2608.18182, Aug 2026): a SmoothQuant-based INT8 workflow for BERT, DistilBERT and XLM-RoBERTa on Xeon CPUs (AVX512-VNNI/AMX kernels) reports "up to 5.8x end-to-end throughput speedup" over FP32 with "negligible—and in some cases no measurable—accuracy loss". The abstract gives no per-model numbers. — [arXiv 2608.18182](https://arxiv.org/abs/2608.18182)
- Intel's 2019 study of INT8 with VNNI for a Transformer MT model on Cascade Lake reports a speedup with less than a 0.5 BLEU drop. — [arXiv 1906.00532](https://ar5iv.labs.arxiv.org/html/1906.00532)
- SmoothQuant (W8A8 PTQ, LLM-scale) reports a 1.56× speedup and 2× memory reduction, but on GPU (FasterTransformer), not CPU. It distinguishes static (offline-calibrated) from dynamic (runtime) activation scales. — [arXiv 2211.10438](https://arxiv.org/pdf/2211.10438)
- QuaRot states that with rotations, 8-bit (and 6-bit) LLaMA-2 models are lossless using plain round-to-nearest and no calibration data. Rotation therefore makes W8A8 easy in addition to W4. — [arXiv 2404.00456](https://arxiv.org/abs/2404.00456)

### Inferences
- **Why today's int8 is slower:** the kernel widens int8 to fp32 and still uses fp32 FMAs (see context). It reads fewer bytes but issues more instructions. When compute-bound, that is a loss. Weight-only int8 helps only when bandwidth-bound, and a 3.6M-parameter model in float16 (about 7 MB) mostly sits in L2/L3.
- **What W8A8 changes:** quantize each activation row once (one max-abs plus a scale; cheap for D≈128-256), then use `vpdpbusd`, which does 32 u8×s8 multiply-adds into 8 int32 lanes per instruction, against 8 for an fp32 FMA. The theoretical ceiling is 4× on the matrix multiplies. After quantize/dequantize overhead, horizontal sums, layer norm, softmax and attention kept in float, 2-3× on the linear layers is a reasonable target, giving maybe 1.5-2× end to end. This is my estimate and needs measuring.
- Engineering details: `vpdpbusd` needs one unsigned operand. Either use non-negative activations (ReLU/clipped outputs, as in NNUE), or add 128 to the activations and subtract a precomputed `128·Σw` per row. Hadamard rotation of activations (already present in `kernels.rs`) flattens outliers before per-token int8. Keep the pointer attention, QK-norm and logits in float.
- At this scale int8 accuracy should be safe (the BERT and MT evidence above). For a parser with exact pointer copying, though, check exact-match and execution accuracy, not only perplexity.
- The Reef core is mostly sparse (k-WTA rows, structured Hadamard lift), so VNNI helps only the dense parts: the output matrix over active rows, convolutions, pointer attention projections. NNUE's sparse-input trick (bitmask of nonzero inputs, then accumulate only those columns) is the matching pattern for a k-sparse layer.

### Gaps
- I found no instruction-level measurement of `vpdpbusd` throughput or latency on Lion Cove (Arrow Lake, Core Ultra 9 275HX) P-cores. Check uops.info. It is also unverified here whether Arrow Lake supports AVX-VNNI-INT8 (`vpdpbssd`, signed×signed, no +128 offset needed); I believe it does, but I did not confirm it.
- I found no published W8A8 accuracy result for a model under 10M parameters on a structured-output or semantic-parsing task. kural would need to measure this itself.

## Q2. Does ternary training work under 50M parameters, and how does it compare with post-training 4-bit?

### Takeaway
Yes, it trains, but at this scale it costs about half the effective capacity. "BitNet b1.58 Reloaded" (6-48M parameters) shows ternary needs roughly double the hidden size to match 16-bit perplexity. ParetoQ (QAT, from 125M parameters up) finds ternary, 2-bit and 3-bit QAT tied with each other and often better than 4-bit at equal model *bits*. The "ternary matches FP16" claims hold only above about 1-3B parameters. Since kural's weights already fit in cache, the memory savings matter little and the compute win needs special kernels. Verdict: **SKIP ternary for now**. Use rotated int8 (and later 4-bit) instead, and revisit ternary only as QAT if model size becomes the constraint.

### Cited Findings
- The BitNet b1.58 paper: ternary {-1,0,+1} trained from scratch matches FP16 perplexity "from 3B" parameters, and the authors stress that post-training conversion does not work. — [HF papers 2402.17764](https://huggingface.co/papers/2402.17764)
- BitNet b1.58 Reloaded (arXiv 2407.09527): Mistral-like 4-layer SLMs with hidden sizes 32/64/128/256 (6M/12M/24M/48M parameters), 8k BPE vocabulary, 1.35B tokens. Perplexity, 16-bit vs 1.58-mean vs 1.58-median: 6M 77.8 / 130.1 / 116.6; 12M 36.7 / 67.4 / 60.0; 24M 21.4 / 36.3 / 37.5; 48M 16.6 / 27.1 / 26.8. The authors say "the effective capacities of the models with 1.58-bit weights are around half that of the models with 16-bit weights". The 1.58-bit model at hidden size 128 (36.3) about matches 16-bit at 64 (36.7). Larger learning rates were sub-optimal at small scale. On small vision models ternary matched or beat 16-bit (CIFAR-10 71.47% vs 70.06%; MNIST 96.08% vs 96.93%). — [arXiv 2407.09527](https://arxiv.org/html/2407.09527v1)
- Spectra / TriLM (ICLR 2025): a suite from 99M to 3.9B parameters (300B tokens) with FloatLM, QuantLM (GPTQ 3/4/6/8-bit) and TriLM. TriLMs beat QuantLMs and FloatLMs per bit only "at scales exceeding one billion parameters". TriLM 3.9B matches FloatLM 3.9B. I could not get explicit 99M comparisons. — [arXiv 2407.12327](https://arxiv.org/abs/2407.12327); [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/6499f26637f74f7c0fbfb46668434973-Paper-Conference.pdf)
- ParetoQ (NeurIPS 2025, Meta): unified QAT for 1-4 bits. Binary hurts badly, while ternary, 2-bit and 3-bit are tied and often surpass 4-bit on the accuracy-vs-size trade-off. On MobileLLM-125M, the best result came from about 90% of the budget spent on full-precision pretraining and about 10% on QAT fine-tuning. A ternary 600M model beats the previous best 3B ternary model. — [arXiv 2502.02631](https://arxiv.org/html/2502.02631v2); [PyTorch blog](https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/)
- Community result (unverified, not peer-reviewed): a 9M TinyStories ternary model reports +0.19 nats (+21% perplexity) against FP16 at the same configuration and token budget. — [HF Thox-ai/ThoxMicro-1bit-9M](https://huggingface.co/Thox-ai/ThoxMicro-1bit-9M)
- bitnet.cpp (arXiv 2410.16144): 2.37-6.17× speedup over llama.cpp on x86 (1.37-5.07× on ARM), 71.9-82.2% energy reduction on x86, outputs lossless against a full-precision kernel. The kernels build on T-MAC lookup tables. These results are for multi-billion-parameter, memory-bandwidth-bound decoding. — [HF papers 2410.16144](https://huggingface.co/papers/2410.16144); [Microsoft Research](https://www.microsoft.com/en-us/research/?p=1100691)
- Cactus (vendor blog): INT4 weights are the default. A 350M model loses about 2-4 MMLU points going from INT8 to INT4. Small models degrade noticeably when the KV cache drops below INT4, so Cactus keeps the KV cache at INT8, and keeps embeddings at FP16/INT8. — [Cactus LFM2.5-350M post](https://cactuscompute.com/blog/lfm2-5-350m); [Cactus TurboQuant-H post](https://cactuscompute.com/blog/turboquant-h)

### Inferences
- At 3-8M parameters, ternary means either about 2× more weights or a clear quality loss (+20-70% perplexity in Reloaded at 6-48M). kural must beat 8B-14B models on accuracy, so capacity is the scarce resource, not memory.
- bitnet.cpp's speedups come from cutting memory bandwidth on very large models. kural's model is L2/L3-resident and compute-bound, so most of that gain does not apply. A ternary matrix multiply with additions only on AVX2 would compete with VNNI int8, which already reaches 4× fp32. The "multiplication-free" benefit is mostly a hardware argument, not an x86 one.
- If the file size matters (ARCHITECTURE.md targets about 4 MB at 4 bits), ParetoQ's recipe applies: train in full precision, then spend the last ~10% on 2-3-bit or ternary QAT. That beats both ternary from scratch and 4-bit PTQ.

### Gaps
- I found no head-to-head of ternary-from-scratch against 4-bit PTQ (GPTQ/rotated RTN) on the same model under 50M parameters. Spectra does not report its 99M numbers in the material I read.
- I found no CPU kernel benchmark for ternary matrix multiplies at the tiny, cache-resident sizes kural uses.

## Q2b. Hadamard-rotation quantization (QuaRot, SpinQuant, Cactus CQ)

### Takeaway
Rotating with an orthogonal (Hadamard) matrix spreads outliers so that both weights and activations quantize well. It is lossless at 8 bits and near-lossless at 4 bits on large LLMs. kural already rotates. The missing piece is integer arithmetic on activations. Verdict: **ADOPT (already partly adopted)**. Keep Hadamard rotation as the front end of a W8A8 VNNI kernel. Use learned rotations (SpinQuant) only if 4-bit is pursued and the gap is large.

### Cited Findings
- QuaRot (NeurIPS 2024): rotations remove hidden-state outliers without changing the output. Weights, activations and KV cache are all at 4 bits. LLaMA2-70B loses at most 0.47 WikiText-2 perplexity and keeps 99% of zero-shot performance. Up to 3.33× prefill speedup and 3.89× decode memory saving (GPU). 6- and 8-bit are lossless with RTN and no calibration. An earlier version quoted 0.29 perplexity; the published text says 0.47. — [arXiv 2404.00456](https://arxiv.org/abs/2404.00456); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract.html)
- SpinQuant (ICLR 2025): learned rotations. W4A4KV4 gap to full precision is 2.9 points zero-shot on LLaMA-2 7B. It closes the gap to full precision by 30.2% (LLaMA-2 7B) and 34.1% (LLaMA-3 8B; up to 45.1% in a later version) relative to QuaRot. Gaps on LLaMA-2 13B and 70B are 1.4 and 1.7 points (from a secondary summary). LLaMA-3 1B/3B are covered, but I did not get their numbers. — [arXiv 2405.16406](https://www.arxiv.org/pdf/2405.16406); [ICLR 2025 slides](https://iclr.cc/media/iclr-2025/Slides/28338_XQXvpv6.pdf)
- Cactus CQ / TurboQuant-H: a Hadamard rotation with per-group Lloyd-Max codebooks, applied offline at convert time, reported for 2-bit per-layer embeddings of Gemma 4 E2B (4× compression, +0.06 perplexity). This is a vendor blog. I found no public description of Hadamard rotation used for CQ's general INT4 weights. — [Cactus TurboQuant-H](https://cactuscompute.com/blog/turboquant-h)

### Inferences
- In kural, rotation already preserved accuracy for int8 weights. Rotating the activations too (a fast Walsh-Hadamard transform on a D=128 vector is about 7×128 additions) is cheap. It makes per-token int8 activations safe and unlocks VNNI.
- The Reef lift is itself Hadamard-structured, so rotated quantization and the structured lift share the same FWHT code.

### Gaps
- All the rotation results are on 1B-70B LLMs. I found none under 50M parameters.

## Q3. Can NNUE-style incremental computation help a transformer or recurrent encoder when one word is appended?

### Takeaway
NNUE's trick (an accumulator updated by adding or subtracting the weight columns of the few inputs that changed) works because NNUE's first layer is a *linear function of sparse inputs*. In a transformer, appending a token changes every earlier token's representation through bidirectional attention. In causal or recurrent layers, earlier states stay fixed, so appending costs one step. Verdict: **ADOPT the principle in a different form**: make the encoder layers before the pointer attention causal/recurrent (Reef's short convolution plus delta-rule memory already are) and cache their state. Recompute only the bidirectional part (pointer attention over at most 44 tokens) and the decoder. A literal NNUE accumulator applies only to the Engram/embedding sum. Also note that at 1.3 ms per full request, the live preview may not need any of this; measure first.

### Cited Findings
- NNUE update: `update_accumulator` copies the previous accumulator, subtracts the weight column of each removed feature and adds each added feature's column. `refresh_accumulator` starts from the bias and adds every active feature. A chess move changes at most 4 inputs (castling), with an average below 3, while a refresh touches about 30 active features (HalfKP). A king move forces a full refresh for that perspective. The accumulator is int16 in AVX2 registers (`_mm256_add_epi16`/`_mm256_sub_epi16`), with weights scaled by 127 and ClippedReLU to 0..127 then int8. It is stored on the search stack rather than undone. — [nnue-pytorch docs](https://raw.githubusercontent.com/official-stockfish/nnue-pytorch/master/docs/nnue.md)
- Current NNUE architectures have about 0.1% input sparsity. The first dense layer after the feature transformer sees no more than about 15% density, which the sparse-input linear layer exploits: a bitmask of nonzero int8 inputs via `_mm256_cmpgt_epi8` and `_mm256_movemask_epi8`, then only those columns are accumulated. — [nnue-pytorch docs](https://raw.githubusercontent.com/official-stockfish/nnue-pytorch/master/docs/nnue.md)
- Kahardipraja, Madureira, Schlangen (EMNLP 2021), "Towards Incremental Transformers": the standard approach, restart-incremental (re-run the whole prefix for every new token), needs as many forward passes as tokens. A recurrent Linear Transformer gives better incremental quality and faster inference, at some cost in full-sequence quality. — [arXiv 2109.07364](https://arxiv.org/abs/2109.07364)
- HEAR (ACL Findings 2023): a hybrid encoder whose lower layers are unidirectional (cached) and whose upper layers are bidirectional and restarted only when a learned policy says so. SNIPS: 93.0 offline F1 and 85.1 streaming exact match at 21.6 GFLOPs, against 74.7 GFLOPs (75.1 streaming EM) for a fully restarted bidirectional encoder. — [ACL Findings 2023.257](https://aclanthology.org/2023.findings-acl.257.pdf)

### Inferences
- Mapping NNUE to Reef: (a) the embedding plus Engram hashed-pair/triple sum is literally an NNUE accumulator. A new word adds one token and 1-2 new n-gram slots, so it is add-only. (b) The 3-tap short convolution and the delta-rule memory are causal, so appending a token is one recurrent step if the state after the previous token is cached. (c) Iterated passes (recurrent depth) and the bidirectional pointer attention break this: each later pass depends on the full sequence, so they must be recomputed. That is cheap at 44 tokens.
- A partial word (the user still typing "pn" of "png") changes the BPE tokenization of the last word. Treat the last word as "removed then added" (NNUE's subtract and add), and keep the cached state at the last word boundary.
- Priority: at 1.3 ms per full request, a full re-parse on each keystroke is already well under typing intervals (about 100+ ms; my assumption). Incremental encoding becomes worthwhile only if Reef gets 5-10× more expensive or if the preview runs a beam with sandbox checks.

### Gaps
- I found no work applying NNUE-style accumulators to NLP encoders, and no incremental *semantic parsing* benchmark. HEAR (slot tagging) is the closest.

## Q4. Are logic gate networks or VSAs viable for any part of the system (decline/route classifiers)?

### Takeaway
Differentiable logic gate networks give extreme inference speed on fixed-size inputs (more than 1M MNIST images per second on one CPU core; CIFAR-10 86.29% with 61M gates). On language they are weak (about 4-5 BLEU on WMT'14 en-de) and lose accuracy when discretized. A decline/route classifier is a tiny cost next to the encoder already being run, and it can be a linear head on the encoder's pooled state for almost nothing. Verdict: **SKIP DLGN** (at most an experiment as a bag-of-n-gram pre-filter that runs before the encoder). VSAs: no NLP or program-parsing results strong enough to adopt. Verdict: **SKIP as a component, keep as a design lens**: binding as a learned multiplicative "slot ⊗ value" gate is already what the delta-rule outer product and modus-ponens gate do.

### Cited Findings
- Petersen et al., difflogic (NeurIPS 2022): learned networks of two-input boolean gates with relaxed training. Abstract: "beyond a million images of MNIST per second on a single CPU core". — [arXiv 2210.08277](https://arxiv.org/abs/2210.08277)
- Convolutional differentiable logic gate networks (Petersen et al., NeurIPS 2024): CIFAR-10 86.29% with only 61 million logic gates, improving over the previous state of the art "while being 29x smaller". — [arXiv 2411.04732](https://arxiv.org/abs/2411.04732)
- Recurrent Deep DLGNs (Bührer, Plesner, Aczel, Wattenhofer, ETH, Aug 2025): WMT'14 en-de, 5.00 BLEU and 30.9% accuracy during training (GRU 5.41; RNN 4.59 per a secondary summary), dropping to 4.39 BLEU and 27.7% after conversion to discrete gates. — [arXiv 2508.06097](https://arxiv.org/abs/2508.06097v1)
- Follow-ups on scaling and training: a study of scalability limits from MNIST to ImageNet ([arXiv 2506.07500](https://arxiv.org/abs/2506.07500v1)). A method that reduces the discretization gap by 98% and trains 4.5× faster (Yousefi et al., June 2025, from a search summary). Light DLGNs with 4× less VRAM and 8.5× fewer training steps ([arXiv 2510.03250](https://arxiv.org/html/2510.03250v1)). Up to 2000 classes (Brändle et al., Sept 2025, from a search summary). I found no DLGN text-classification result.
- VSA/HDC survey (Kleyko et al., ACM CSUR 2023, two parts): hypervectors with binding and bundling. Language applications covered are word embeddings from context hypervectors, language identification from letter n-gram statistics, and encodings of finite-state automata and grammars. Nothing on semantic parsing. — [arXiv 2112.15424](https://arxiv.org/abs/2112.15424); [Part I, arXiv 2111.06077](https://arxiv.org/pdf/2111.06077)
- Simple VSA binding struggles with complex compositional structures. Generalized HRR (2024) reports better decoding of trees and higher capacity for bound vectors than FHRR. — [arXiv 2405.09689](https://arxiv.org/html/2405.09689v1)
- VSA binding operations can be mapped to matrix multiplies within ordinary recurrent networks (Frady, Kleyko, Sommer). — [arXiv 1803.00412](https://arxiv.org/pdf/1803.00412)
- LARS-VSA uses explicit hypervector binding inside an attention mechanism for relational sequence-to-sequence tasks. This is a search-snippet description; I did not verify its numbers. — [arXiv 2405.14436](https://arxiv.org/pdf/2405.14436v1)

### Inferences
- A decline/route head needs a pooled encoder state that already exists. A logistic or MLP head costs a few thousand multiply-adds (microseconds), so a gate network saves nothing that matters. The only case where DLGN helps is a pre-filter that runs *without* the encoder (for example on hashed n-gram bits), and kural has no latency need for that.
- DLGN's discretization drop (5.00 → 4.39 BLEU) and low absolute language quality make it a poor fit for anything that must compete with 8B-14B LLMs.
- VSA binding fits Reef as a design lens: the delta-rule write `S += β(v − S·k)⊗k` is a learned key-value binding, and k-WTA sparse codes behave like sparse block codes. There is no evidence that hand-designed VSA codes beat learned ones for parsing, so do not add a separate VSA component.

### Gaps
- I found no published DLGN or VSA result on intent classification, out-of-domain detection or NL-to-code. Their value for the decline classifier is untested.
- I did not verify LARS-VSA's numbers, the Yousefi et al. and Brändle et al. results, or the RNN baseline of 4.59 BLEU against the primary papers.

## Verdict summary
| Technique | Verdict | Why |
|---|---|---|
| W8A8 int8 with AVX-VNNI (`vpdpbusd`), activations Hadamard-rotated | **TEST first** | Compute-bound model; 4× multiply-adds per instruction against fp32 FMA; negligible accuracy loss for BERT/MT. Today's int8 path does float math, which is why it is slower. |
| Hadamard rotation (QuaRot-style), SpinQuant learned rotations | **ADOPT** (rotation) / test (learned) | Lossless at 8 bits with no calibration; needed for int8 activations; shares code with the structured lift. |
| NNUE incremental accumulator | **ADOPT the principle** for the Engram/embedding sum and causal layers; measure need first | Appending a word is add-only for linear sums and one step for causal/recurrent layers; bidirectional passes must be recomputed, which is cheap at 44 tokens. 1.3 ms per request may make it unnecessary. |
| NNUE sparse-input linear layer (nonzero bitmask, then columns) | **ADOPT** for k-WTA layers | Matches Reef's 1-2% active neurons exactly. |
| BitNet b1.58 ternary from scratch | **SKIP for now** | About half the capacity under 50M parameters; bitnet.cpp gains are bandwidth gains that do not apply to a cache-resident model. Revisit as ParetoQ-style final QAT if file size matters. |
| Post-training 4-bit (rotated) | **Later test** | Only to shrink the file; Cactus reports 2-4 MMLU points lost at 350M. |
| Differentiable logic gate networks | **SKIP** | Weak on language; a classifier head on the encoder is already about free. |
| VSA / HDC | **SKIP as a component**, keep as a lens | No NLP/parsing results; delta-rule binding already covers "slot = value". |
