# JEPA-style and related representation objectives for Reef (kural)

Scope: I-JEPA / V-JEPA, LLM-JEPA, other JEPA-for-text work, contrastive and paraphrase-invariance objectives for semantic parsing and NL-to-code, energy/verifier reranking of candidate programs, and Jev (TypeSafe AI). These are judged against Reef as described in /home/srihari/Projects/kural/docs/ARCHITECTURE.md: about 7-8M parameters, CPU inference, a typed program under grammar-constrained decoding with pointer copying, trained on synthetic data from about 900 phrase patterns. The design (section 4.3) plans a JEPA auxiliary loss in which the request embedding predicts the program embedding and paraphrases of one program land on the same point. The ladder in section 6 places "+ JEPA loss" as its own measured step.

Verdict summary (each is argued in the sections below):

| Item | Verdict | One-line reason |
|---|---|---|
| LLM-JEPA-style aux loss (request predicts program embedding) | **Test** (cheap, but low expected gain on unseen wording) | Real gains in the paper, but only at 1B+ parameters, only in-distribution, about 2x training compute, and two touchy hyperparameters; no evidence below 1B or for OOD paraphrase |
| Paraphrase-to-paraphrase invariance (SimCSE-style / contrastive) | **Test, after augmentation** | Needs real paraphrases to bind together; with only template paraphrases it makes the template wording invariant, not unseen wording |
| Paraphrase augmentation of synthetic data (AutoQA-style, with a round-trip filter) | **Adopt** (strongest evidence for kural's actual weakness) | Synthetic-only parsers end up within about 5-6 points of human-data parsers when trained this way; the filter is worth about 18 points |
| Embedding projection onto the synthetic language (Marzoev et al.) | **Test** as a fallback/normaliser | Directly targets "wording it never saw"; beats synthetic-only seq2seq on Overnight |
| JEPA energy as reranker/confidence | **Skip as the primary signal; maybe test as one feature** | No published evidence it reranks programs; execution-based verifiers (LEVER) and kural's sandbox are the proven route |
| I-JEPA / V-JEPA | **Skip** (background only) | Vision; their mechanism (masked-region prediction, EMA target encoder) is not what Reef needs |
| CodeT5+ contrastive/matching | **Skip as a recipe, keep as evidence** | Contrastive and matching help *retrieval* (MRR), not generation; smallest model is 220M |
| Jev (TypeSafe AI) | **Skip** | Hosted, closed decision API; no architecture or calibration data published; nothing transferable beyond "typed decision + score" |

---

## What LLM-JEPA reported: tasks, gains, model sizes, training cost

### Takeaway
LLM-JEPA (Huang, LeCun, Balestriero; arXiv Sept 2025, ICLR 2026) adds λ·(1 − cos(Pred(Enc(text)), Enc(code))) to next-token loss. It reports large fine-tuning gains on NL-RX regexes (Llama-3.2-1B: 57.3 → 71.5%) and smaller gains on Spider (+3.0) and GSM8K (+4.0). Every model tested has about 1B parameters or more, training costs about 2x, and no out-of-distribution paraphrase test was run.

### Cited Findings
- Paper identity: "LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures", Hai Huang, Yann LeCun, Randall Balestriero, submitted 11 Sept 2025 (v2 7 Oct 2025). It was tested on NL-RX, GSM8K, Spider and RottenTomatoes with the Llama3, OpenELM, Gemma2 and OLMo families — [arXiv abs](https://arxiv.org/abs/2509.14252)
- Accepted at ICLR 2026 — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac818ec2da1c976b267de32f59709c69-Abstract-Conference.html); code is at github galilai-group/llm-jepa (per [mlanthology listing](https://mlanthology.org/iclr/2026/huang2026iclr-llmjepa/) / search results, repository not inspected)
- **Formulation:** loss = Σ next-token CE + λ·d(Pred(Enc(Text)), Enc(Code)). Enc is the last-layer hidden state of the last token. Text and Code are packed into one context with a custom attention mask. The predictor has *tied weights*: k special [PRED] tokens are appended to the text, and the last one's embedding is the prediction (k=0 means the predictor is the identity). d is cosine distance. The search grid was k ∈ {0..4}, λ ∈ {0.5, 1, 2, 4} — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **NL-RX-SYNTH fine-tuning (Table 12), baseline → LLM-JEPA, exact-match %** — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2):
  - Llama-3.2-1B-Instruct: 57.29 ± 5.32 → 71.46 ± 1.34
  - gemma-2-2b-it: 33.65 ± 3.24 → 43.12 ± 2.61
  - OpenELM-1.1B-Instruct: 12.07 ± 1.81 → 25.40 ± 2.40
  - OLMo-2-0425-1B-Instruct: 87.09 ± 0.36 → 87.52 ± 0.29 (a strong baseline gains almost nothing)
- **Other datasets, Llama-3.2-1B (Table 13):** NL-RX-TURK 22.49 → 30.94; GSM8K 32.36 → 36.36; Spider 47.52 → 50.55 (± about 2, so the Spider gain is within roughly 1.5 standard deviations) — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **Model sizes (Table 15, NL-RX-SYNTH):** Llama-3.2-3B 74.55 → 77.16; Llama-3.1-8B 35.77 → 63.57 (± 16.81; the paper says the 8B baseline struggled to terminate regexes); OLMo-2-7B 87.26 → 87.75 — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **Q&A, where text and "code" are not truly two views:** NQ-Open 20.12 → 21.59; HellaSwag 69.40 → 70.51. **Reasoning models on GSM8K:** Qwen3-1.7B 44.32 → 45.00; DeepSeek-R1-Distill-Qwen-1.5B 13.87 → 15.04 — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **Ablation (Table 3, Llama-3.2-1B NL-RX-SYNTH):** cosine 71.46; MSE 70.64; prepend predictor 68.07; reversed direction Code→Text 65.70; **InfoNCE (contrastive) 34.40 ± 6.10, worse than the 57.29 baseline**; ℓ2 norm 2.22 (collapse) — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **Pretraining:** from scratch, Llama-3.2-1B architecture on NL-RX-SYNTH: 54.38 → 60.59. Paraphrase-dataset pretraining followed by fine-tuning: Rotten Tomatoes 56.57 → 57.76, Yelp 26.46 → 27.15. The authors call the pretraining experiments "preliminary" — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **Training cost:** the paper is internally inconsistent. Section 3.1 says "only one additional forward pass"; Section 3.2 says "two forward passes instead of three". The conclusion names "the 2-fold increase in compute cost during training" as the primary bottleneck. Random "loss dropout" at rate α makes the per-epoch cost (2 − α)× standard fine-tuning. No inference overhead is reported (the predictor and the JEPA term are training-only) — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **Stated limitations:** the 2x compute; two extra hyperparameters (λ, k) whose optimum "may occur at any point in a grid", with no efficient tuning method; the method is restricted to datasets with non-trivial paired views — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **Smallest model tested:** about 1B parameters (Llama-3.2-1B, OpenELM-1.1B, OLMo-2-1B). Nothing smaller is reported — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- **Unverified (aggregator only):** 2026 follow-ups, namely DLLM-JEPA (masked-diffusion views), a multimodal extension, a scaling study on NL-RX-SYNTH, and a JEPA-style retrieval dual encoder on MS MARCO/BEIR. These are listed by [emergentmind](https://www.emergentmind.com/topics/llm-jepa); I could not confirm them on arXiv.

### Inferences
- NL-RX matters most for kural, because it has the same structure as kural's problem: NL-RX-SYNTH descriptions come from a grammar, and NL-RX-TURK holds crowd paraphrases of them. The paper trains and tests *within* each split; it does not train on SYNTH and test on TURK. So the +8.5 on TURK shows JEPA helps on human wording when you also *train* on human wording, not that it transfers to unseen wording. The SYNTH→TURK transfer experiment is the one Reef needs, and it has not been published (as far as I found).
- Gains are largest where the baseline is weak and unstable (high variance: Llama-1B ± 5.3, OpenELM 12%). They nearly vanish for a strong baseline (OLMo 87%). This reads as JEPA partly acting as a regulariser or stabiliser for fine-tuning, not as a guaranteed boost.
- The predictor shares the LM's weights, and the next-token loss is what prevents collapse; the ℓ2 variant collapsed. A 7M Reef with a small grammar decoder may have weaker protection against collapse, so watch embedding variance (or add a VICReg-style variance term) if it is tried.
- Training cost for Reef: under 2x a small model's fine-tuning (minutes to hours on the laptop GPU), so the experiment is cheap. Inference cost is zero, because the term is dropped at inference.
- The InfoNCE result (34.4, below baseline) is a caution against using a contrastive loss as the text→program alignment term in a generative model, at least with that paper's setup.

### Gaps
- No LLM-JEPA result below about 1B parameters, and no result on an encoder-decoder or grammar-constrained parser.
- No OOD or compositional-split evaluation (for example SYNTH→TURK, or Spider→Spider-Syn).
- No absolute GPU-hours; only "4.83 PFLOPs apart" between steps in one table.
- The 2026 follow-ups are unverified.

---

## Does JEPA or contrastive training help out-of-distribution paraphrase robustness in semantic parsing?

### Takeaway
I found no published result showing a JEPA or contrastive auxiliary loss improving a semantic parser on held-out paraphrases. The interventions with numbers behind them are *data-side*: paraphrase augmentation with a round-trip filter (AutoQA), training on synonym-substituted data or adversarial data, and schema synonym annotation (Spider-Syn), plus projecting unseen inputs onto the synthetic language with sentence embeddings (Marzoev et al.).

### Cited Findings
- **The size of the problem (Spider-Syn):** text-to-SQL models trained on Spider lose about 20-30 points of exact match when schema words are swapped for common synonyms. GNN 48.5 → 23.6; IRNet 53.2 → 28.4; RAT-SQL 62.7 → 33.6; RAT-SQL+BERT 69.7 → 48.2 — [Gan et al., ACL 2021, Table 1](https://arxiv.org/abs/2106.01065)
- **Spider-Syn defences (RAT-SQL+BERT, Spider / Spider-Syn):** baseline 69.7 / 48.2; trained on Spider-Syn 67.8 / 59.9; BERT-Attack adversarial training 68.7 / 58.5; GloVe adversarial 48.7 / 27.7 (it hurts); manual multi-annotation of schema synonyms 67.4 / **62.6**; automatic multi-annotation 68.7 / 56.0 — [Gan et al., Table 3](https://arxiv.org/abs/2106.01065). The paper's verdict is that explicit synonym annotation in the input beats adversarial training — [ACL Anthology](https://aclanthology.org/2021.acl-long.195)
- Pretrained scale reduces but does not remove the drop: on Spider-Syn, RATSQL+BERT loses 21.5 points of exact match and T5-3B loses 12.2 — [TKK, arXiv 2210.12674](https://arxiv.org/pdf/2210.12674)
- **Dr.Spider:** 17 perturbations of the database, question and SQL. The authors state that even the most robust model suffers a 14.0% overall drop and a 50.7% drop on the hardest perturbation. Its question perturbations are LLM-generated paraphrases checked by experts — [Chang et al., arXiv 2301.08881](https://arxiv.org/pdf/2301.08881)
- **Synthetic→natural gap:** the authors cite Blukis et al. (2018) as reporting "a test-time accuracy gap of over 54%" between models trained on synthetic data and models trained on real user utterances — [Marzoev et al., "Unnatural Language Processing", arXiv 2004.13645](https://arxiv.org/abs/2004.13645)
- **Projection (sim-to-real for NLP):** train on grammar-generated data only, then at test time map each natural utterance to the nearest synthetic utterance in a learned sentence-embedding space and parse that. On Overnight, logical-form accuracy averaged over 8 domains (human test questions) — [Marzoev et al., Table 1 (from PDF)](https://arxiv.org/abs/2004.13645):
  - Synthetic training only: seq2seq 0.19; seq2seq+BERT 0.35; **projection 0.37**
  - Real human training data: seq2seq 0.27; seq2seq+BERT 0.42
  - The authors say projection "matches or outperforms state-of-the-art models trained on natural language data in several domains" (comparable in 3 of 8 Overnight domains).
- **AutoQA (paraphrase augmentation of synthetic data):** a pipeline of template synthesis, a BART auto-paraphraser, and a filter that keeps a paraphrase only if the current parser maps it back to the original logical form. Results: 62.9% logical-form accuracy on natural Schema2QA questions, 6.4 points below a model trained on expert annotations plus crowd paraphrases; 69.8% answer accuracy on Overnight, 16.4 points above prior zero-shot and 5.2 points below the same model trained on human data. **Removing the filter costs 18 points on average** — [Xu et al., EMNLP 2020, arXiv 2010.04806](https://arxiv.org/pdf/2010.04806)
- Adversarial training improves semantic-parser generalisation in and out of domain against lexical and stylistic variation — [Marzinotto et al., arXiv 1910.06700](https://arxiv.org/pdf/1910.06700) (abstract-level claim; numbers not extracted)
- **SimCSE (background, recalled from the paper, not re-fetched this session):** dropout-noise contrastive learning (unsupervised) and NLI-pair contrastive learning (supervised) for sentence embeddings, which improved STS over prior methods with BERT-base — [Gao et al., arXiv 2104.08821](https://arxiv.org/abs/2104.08821). It is an embedding method; it was not evaluated as a parser auxiliary loss.
- **CodeT5+ contrastive / matching:** stage-2 bimodal objectives include text-code contrastive and text-code matching. Removing matching drops CodeSearchNet text-to-code retrieval MRR from 77.4 to 74.8 (770M). The contrastive term itself was not ablated separately. The smallest model is 220M, with MRR 77.1 overall and 43.3 on AdvTest — [CodeT5+, arXiv 2305.07922](https://arxiv.org/html/2305.07922). The reported benefit is for *retrieval*. For generation, the causal-LM objective was the one that mattered (removing it in stage 1 dropped PY150 EM from 44.9 to 36.2).

### Inferences
- Kural's weakness is the Overnight / AutoQA situation almost exactly: about 900 hand-written patterns, i.e. a synthetic grammar. In that literature the lift comes from **more varied wording in training** (paraphrase plus filter: about +18 points from the filter alone, and close to the human-data ceiling). A changed loss is not where the lift came from.
- A JEPA/contrastive paraphrase-invariance loss can only make *the paraphrases you have* map together. If every paraphrase comes from the 900 patterns, the loss teaches invariance across template variants, which the decoder largely already handles. It will not by itself reach wording that appears in no training input. It can help in two cases: (a) combined with real or generated paraphrases (Stack Exchange titles, NL2Bash, local-LLM paraphrases filtered by round-trip parse), as a regulariser that ties them to one program embedding; (b) with pretraining on general English, where JEPA may shape the encoder so that unseen synonyms lie near seen ones. (b) is plausible but undemonstrated at under 50M parameters.
- Order of experiments suggested by the evidence: (1) offline paraphrase augmentation with an 8-14B local LLM plus a round-trip filter using the current kural parser (AutoQA recipe), measured on the untouched test set; (2) then add the LLM-JEPA term (request→program, cosine, tied predictor) and paraphrase↔paraphrase pull, and measure the delta; (3) optionally, projection onto nearest synthetic utterance as a fallback for low-confidence requests.
- Spider-Syn suggests an analogue for kural: give the model explicit synonym or alias annotations for tool and flag names. The local index of man and tldr text is a natural source. This beat adversarial training there (62.6 vs 58.5).

### Gaps
- No found paper reports a contrastive or JEPA auxiliary loss improving a semantic parser on held-out paraphrase sets (Spider-Syn, Dr.Spider NLQ, NL-RX-TURK from SYNTH). I searched specifically and found none; this does not prove absence.
- No evidence at all under 50M parameters for any of these objectives in parsing. The AutoQA and Marzoev models are BART/BERT-scale (over 100M), not tiny.
- Dr.Spider's per-perturbation paraphrase numbers were not extracted.

---

## Is JEPA usable as a reranker / confidence signal?

### Takeaway
In principle yes: the energy cos(Pred(Enc(request)), Enc(candidate program)) can be computed for every beam candidate. But no paper I found evaluates JEPA energy for reranking programs or for calibration. Execution-informed learned verifiers do have evidence (LEVER: +4.6 to +10.9 points), and they mostly lose their benefit when execution is unavailable.

### Cited Findings
- LLM-JEPA evaluates only generation accuracy and reports no reranking or calibration experiments — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2) (absence noted from my reading of the extracted results)
- LEVER trains a verifier on (NL, program, execution result) and reranks candidates by combining generation probability with the verifier score. It improves code-davinci-002 by 4.6 to 10.9 points across four datasets (Spider, WikiTQ, GSM8K, MBPP) — [Ni et al., ICML 2023](https://proceedings.mlr.press/v202/ni23b)
- Without test or execution results, the LEVER verifier did not improve results — [Hassid et al., arXiv 2404.00725](https://arxiv.org/pdf/2404.00725)
- Historical precedent: the original Overnight work used the synthetic grammar as a *source of features for reranking logical forms* — as described in [Marzoev et al.](https://arxiv.org/abs/2004.13645)

### Inferences
- For Reef, the JEPA energy is cheap to compute: one program-encoder pass per beam candidate on a 7M model is microseconds to milliseconds. It could be one input feature to the planned confidence head (trained on sandbox outcomes), alongside log-probability and sandbox results. That is a test, not an adopt.
- As a standalone confidence, it is risky. The JEPA term is trained only on positive pairs with cosine (no negatives), so nothing forces wrong-but-similar programs (for example `-mtime +7` vs `-mtime -7`) to have high energy. A reranker needs hard negatives. The LLM-JEPA InfoNCE result shows negatives can hurt *generation* when mixed into the loss, so a separate small, contrastively trained scorer (or the confidence head itself) is the cleaner design.
- The kural sandbox is kural's equivalent of LEVER's execution results. The evidence says execution information is what makes a verifier work, so the sandbox plus confidence head should stay primary.

### Gaps
- No empirical study of JEPA/embedding-predictive energy as a program reranker or calibrated confidence was found.

---

## I-JEPA and V-JEPA (background)

### Takeaway
They are vision self-supervised methods that predict the representations of masked image or video regions from context, using an EMA target encoder. They supply the conceptual ancestry (predict in embedding space, not in input space) and nothing directly usable for a text→program parser.

### Cited Findings
- I-JEPA predicts representations of target image blocks from a context block, with no hand-crafted augmentations. The authors report efficient training of ViT-H/14 on ImageNet (under 72 hours on 16 A100s, recalled from the abstract; not re-fetched) — [Assran et al., CVPR 2023, arXiv 2301.08243](https://arxiv.org/abs/2301.08243)
- V-JEPA extends feature prediction to video, trained on about 2M videos and evaluated with frozen backbones (recalled from the abstract; not re-fetched) — [Bardes et al., arXiv 2404.08471](https://arxiv.org/abs/2404.08471)

### Inferences
- **Skip.** The mechanisms (multi-block masking, EMA teacher, large ViTs) target perception at 100M-600M+ parameters. LLM-JEPA is the relevant adaptation for text and code.

### Gaps
- Exact I-JEPA/V-JEPA numbers were not re-verified this session; treat the compute figures as unverified.

---

## What is Jev (TypeSafe AI), verified?

### Takeaway
Jev is TypeSafe AI's hosted, closed "System One" decision model, in early access. It takes a state (text, JSON or images) and typed questions, and returns structured answers with probabilities instead of generated text. The primitives are Choice, Score, Noul (yes/no) and Box. TypeSafe says it is trained with "RLCD" (reinforcement learning for calibrated decisions), but publishes no architecture, size or calibration measurements.

### Cited Findings
- From the official docs: Jev is "the first System One model". It evaluates typed questions against a state and returns structured results: "No text generation, no parsing." State may be "text, JSON, or images" — [docs.typesafe.ai](https://docs.typesafe.ai)
- Primitives in the official docs — [docs.typesafe.ai](https://docs.typesafe.ai):
  - **Choice** returns `choice`, `probabilities`, `confidence`
  - **Score** (rubric) returns `score`, `probabilities`, `confidence`
  - **Noul** returns a 0-1 truth value with no confidence field
  - **Box** returns an image bounding box with `confidence`
  - Questions are "evaluated in parallel and in isolation"; "adding questions barely changes the response time"
- Training: Jev is trained with RLCD, "Reinforcement learning for calibrated decisions," which "trains Jev to return decisions and calibrated probabilities instead of generated text." Calibration is described only as the intended property (0.8 should be right about 80% of the time); no method or ECE numbers are given — [TypeSafe AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)
- Homepage claims: System One models use "a new architecture, a new sampler, and a new training algorithm". It markets "193.6x Faster, 444.6x Cheaper" against unnamed LLMs on its own workflows. Its example ($0.000081 vs $0.013880; 0.114 s vs 8.566 s) does not reproduce those multiples (about 171x and 75x). It also claims "Zero Hallucinations", "$42 per billion input tokens", early access, and a hosted API — [typesafe.ai](https://typesafe.ai). These are marketing claims, not independently verified.
- **Secondary sources only (unverified):** launch on 15 Sept 2026 with $40M funding led by DCVC; founder Diogo Almeida (ex-OpenAI); typical latency about 100 ms; version IDs jev-1.13.0 / jev-1.14.0 with a jev-latest alias — [letsdatascience](https://letsdatascience.com/news/typesafe-ai-launches-jev-decision-model-889a38c0), [wavect review](https://wavect.io/blog/jev-ai-decision-model-review/), [zenmux](https://zenmux.ai/typesafe/jev-1.14), [flaviocopes](https://flaviocopes.com/jev.md)

### Inferences
- **Skip.** It is cloud-only and closed, which conflicts with kural's local, CPU-only principle. It gives no architectural or training detail beyond the name "RLCD", and it has no published calibration evidence. ARCHITECTURE.md already describes it correctly ("nothing to take from it directly"). The one transferable idea is the interface: a typed decision with per-option probabilities, plus a separately trained calibrated confidence. Kural's planned confidence head trained on sandbox outcomes already has that shape, and it has a better ground-truth signal (execution) than anything TypeSafe has disclosed.
- If the description in ARCHITECTURE.md is revised, it should say "hosted, closed, early-access decision API; trained with RL for calibrated decisions (vendor claim); no published architecture or calibration data".

### Gaps
- Model architecture, parameter count, training data, calibration metrics and independent benchmarks: none are published by TypeSafe (as of this search). The funding and founder details come only from secondary coverage.

---

## Answer to the key question: would a JEPA loss make a tiny parser robust to unseen wording?

### Takeaway
Probably only a little, and not on its own. The published gains (LLM-JEPA) are in-distribution, at 1B+ parameters, and largest for weak baselines. The documented fixes for "trained on synthetic templates, fails on natural wording" are paraphrase augmentation with round-trip filtering and embedding projection. Recommendation: **test** JEPA as a cheap ladder step *after* paraphrase augmentation, measured on an untouched human-written test set, and keep it only if it adds points there.

### Cited Findings
- LLM-JEPA's smallest model is about 1B parameters; it has no OOD test; training costs about 2x — [arXiv HTML v2](https://arxiv.org/html/2509.14252v2)
- Paraphrase plus filter brings synthetic-only parsers to within 5.2-6.4 points of human-data parsers; the filter alone is worth 18 points — [AutoQA](https://arxiv.org/pdf/2010.04806)
- Projecting onto synthetic utterances beats synthetic-only seq2seq+BERT (0.37 vs 0.35 mean; seq2seq without BERT 0.19) on Overnight — [Marzoev et al.](https://arxiv.org/abs/2004.13645)
- Training on synonym or paraphrase data recovers 10-14 of the 21 points lost on Spider-Syn — [Gan et al.](https://arxiv.org/abs/2106.01065)

### Inferences
- A concrete experiment design for Reef's ladder step "+ JEPA loss":
  1. Hold out a human-written test set (NL2Bash plus requests from other people), never used for tuning.
  2. Baseline: the current model plus paraphrase-augmented data, i.e. local-LLM paraphrases of the generator's outputs, kept only if the current parser round-trips them to the same program.
  3. Add L_jepa = 1 − cos(P(h_request), sg(h_program)), with P a small MLP or tied [PRED] token, and h_program from the decoder's encoding of the gold program. Add L_para = 1 − cos(h_req_a, h_req_b) for paraphrases of one program. Sweep λ ∈ {0.5, 1, 2} and watch embedding variance for collapse.
  4. Report the delta on the held-out set, plus SYNTH-only→human transfer (the experiment missing from the literature). Seeds matter: LLM-JEPA's baselines vary by ±5 points, so use at least 3 seeds.
- Expected cost: under 2x training compute for the fine-tuning stage; zero inference cost; optional reranking feature at about one encoder pass per beam candidate.

### Gaps
- No published evidence either way for JEPA at under 50M parameters, or for SYNTH→human transfer. Kural would be producing the first data point, so the ladder's "keep only if it helps" rule is the right guard.
