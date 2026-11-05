# TRACE–VERO merging study: findings and data

Study `trace_vero_qwen25_v1`, complete as of 2026-11-03. It compares the released TRACE and VERO Qwen2.5-VL-7B checkpoints with their shared backbone, then tests three training-free 50:50 weight merges. Nothing was trained.

This file collects the study's key findings, its experimental settings and its data in one place. It covers only what this study set up and measured; the TRACE evaluation protocol it reuses (benchmarks, prompts, scorers, judge) is cited by revision and not re-described.
- **Interpretation, caveats and the decision.** These are in `analysis_report.md` and `decision_memo.md`, in this directory. The frozen design is `../../DESIGN.md`.
- **Where the numbers come from.** Numbers are read programmatically from the aggregate tables in `../tables/`, and each section names its source. §2 also reads the study config (`../../configs/`), `../manifests/`, `../splits/`, `../merges/`, `../audit/`, and the work root's run logs and scoring directories (vLLM settings, attempt hardware, judge use); §8.2 also uses the runner summaries. A few values are quoted from the study's reports instead, each with its source: the pre-repair M050 − T estimate and the 0.10 pp A7 interval change (`analysis_report.md`), the development runs' 1e-14 adapter check (`merge_screen.md`), the follow-up's cost (`decision_memo.md`) and the fault signature (DESIGN.md A9).
- **Rounding.** Two decimals for percentage-point values; one for rates and for differences between rates (persistence excess); integers for token counts and mean group sizes. Rounding is half-up on the stored value. Differences and ranges are computed before rounding, so they can differ by 0.01 from the difference of the displayed values.

## How to read the numbers

**Models.**
- B: Qwen2.5-VL-7B-Instruct, the base.
- T: the released TRACE model.
- V: the released VERO model. V@P0 is VERO with its own think/answer chat template (the archived canonical runs); V@P1 is VERO with the backbone's template.
- M050 = ½V + ½T, CV050 = ½V + ½B, CT050 = ½T + ½B. Unsuffixed merge labels are served with the backbone's interface (P1).
- Labels ending @P2 (merges, B and T) are served with VERO's interface. That is a development-only diagnostic.

**Metric.** The macro is the unweighted mean of the 24 official benchmark scores, averaged over seeds.

**Intervals and tests.** Every interval is a 95% paired cluster bootstrap: 10,000 replicates over the bootstrap units (image/problem groups, with near-duplicate twins joined; DESIGN.md A7), with all models and seeds resampled together. p-values are two-sided bootstrap p-values; with 10,000 replicates the smallest attainable p is 0.0002, shown as < 0.001.
- Seeds 42–44 are decoding seeds. The intervals cover item and decoding variability for these fixed checkpoints, not training-seed variability: each recipe is one released checkpoint.
- Only the four comparisons of the prespecified primary family are confirmatory and Holm-adjusted.
- Everything else is descriptive or exploratory. That includes categories, benchmarks, capture/damage, sensitivity, the development screen and P2.

**Tiers.**
- Stage B: the archived runs on the full split (32,805 rows, seeds 42–44).
- Development screen: 2,910 rows, seed 42.
- Confirmation: 26,228 rows, seeds 42–44.

## 1. Key findings

1. **TRACE and VERO gain similar amounts over the base** (Stage B, native interfaces).
   - T − B = +4.06 [+3.66, +4.45] and V@P0 − B = +4.30 [+3.78, +4.82].
   - Their difference, T − V@P0 = −0.24 [−0.75, +0.28], is not resolved.
   - Exploratory category contrasts put TRACE ahead in Charts & Tables (+2.42 [+1.40, +3.43]) and VERO ahead in Visual Math (−2.02 [−3.15, −0.86]).
2. **The parents' unique successes are partly reproducible.**
   - On a single seed they disagree on 21.8% of binary-scored rows.
   - A T-only success set recurs on another decoding seed 47.1% of the time (label-swap null 26.9%). A V-only set recurs 50.0% of the time (null 26.9%).
3. **The two weight updates differ in size and are nearly orthogonal** (§2.3; a diagnostic, not a test).
   - Over all 729 tensors, ‖V − B‖ is 4.4 times ‖T − B‖ (0.406% against 0.092% of ‖B‖).
   - The cosine between the two updates is 0.0073: small, though far above the ±1e-05 expected of two random directions in this many dimensions. Norms and cosines alone do not show that the updates encode separable skills.
4. **VERO's gain depends on its interface.**
   - With the backbone template, V@P1 − V@P0 = −2.94 [−3.33, −2.56] on confirmation.
   - 66% of that macro drop is in PuzzleVQA (−33.10) and VisualPuzzles (−13.29). There, many V@P1 answers end in a LaTeX-wrapped option letter that the pinned scorers do not read: 32.0% of its PuzzleVQA rows are scored 0 with the correct final letter.
   - TableVQABench (−15.97) is another 23% and is not diagnosed.
5. **The merge beats both controls and VERO-on-the-backbone-template, but is not shown to beat TRACE**
   (confirmation, Holm-adjusted):
   - M050 − T = +0.23 [−0.19, +0.65], Holm p = 0.29.
   - M050 − V@P1 = +3.07 [+2.62, +3.51].
   - M050 − CV050 = +1.29 [+0.93, +1.64].
   - M050 − CT050 = +1.88 [+1.45, +2.31].
   - The last three have Holm p < 0.001. No noninferiority margin was set.
6. **M050 scores 52.36**, against T 52.13 and native VERO 52.23. M050 − V@P0 is a difference of point estimates outside the primary family; no paired interval was computed.
7. **Halving each update costs differently.**
   - CT050 keeps 59% of TRACE's gain over the base.
   - CV050 scores 1.77 pp above VERO itself under P1, consistent with half of VERO's update softening its answer-format shift.
8. **Category pattern (descriptive).**
   - M050 is below T only in Charts & Tables with an interval excluding 0 (−0.91 [−1.76, −0.07]).
   - It is above T in Science & General (+1.15 [+0.24, +2.04]) and Spatial Reasoning (+1.13 [+0.14, +2.15]).
   - VERO's half-update on top of the TRACE blend gains most in Visual Math (+3.73 [+2.69, +4.78]).
   - TRACE's half-update on top of the VERO blend gains most in Science & General (+2.33 [+1.52, +3.15]).
9. **Row-level capture** (independent repeat; the base reference is post hoc, with no paired intervals):
   - M050 keeps VERO-only successes at 52.8%, against the base model's 36.7% on the same rows.
   - It keeps TRACE-only successes at 49.0%, close to the base model's 47.6%.
   - In accuracy terms (§7.7; differences of point estimates, no paired intervals), M050's pooled binary-accuracy lead over CV050, CT050 and B (+1.19 / +2.01 / +4.25 pp) has its largest term in the shared successes it keeps (+0.99 / +2.21 / +4.09 pp); against CT050, the VERO-only successes it keeps add +1.59 pp. It rescues fewer joint failures than any of them.
10. **Frozen hypotheses** (§2.10).
    - H1 (integration beyond shrinkage) holds in direction.
    - H2 (the category trade-off transfers) is half supported.
    - H3 (interface dependence) is supported for the macro and for Puzzles & Logic. It is not supported for Visual Math being one of the two largest drops: Charts & Tables drops more (−3.89 [−4.73, −3.04], through TableVQABench), although Visual Math also drops (−1.77 [−2.68, −0.90]).
11. **Measurement checks** (§4).
    - All 25 new runs and the 2 local reruns match their archived references on every one of the 407,076 rows checked: the same prompt payload and the same rendered prompt length (27 of 27 conform).
    - All 24 metric adapters reproduce the archived aggregates (largest error 4.5e-11 pp).
    - No row is missing or extra in any of the 864 model × seed × benchmark slices, and no stored row score is missing.
    - Local reruns of B and T in the new environment move the macro by −0.25 and −0.02 pp, inside the archive's own seed-to-seed range. B's rerun writes shorter responses (mean −11.8 completion tokens; 11 length stops against 107 / 112 / 94 per archived seed). With local-rerun B and T the seed-42 macro contrasts lead to the same conclusions, but category contrasts involving B or T move by up to 1.23 pp, and 3 category intervals change side of 0 (M050 − T in Spatial Reasoning; CV050 − B in Puzzles & Logic; CT050 − B in Puzzles & Logic) (§7.11).
12. **One technical repair** (DESIGN amendments A8, A9). One confirmation run's judge (Qwen3-32B at tensor-parallel 4 on A800 cards) produced degraded extractions. The run was rescored at tensor-parallel 1, and every table here uses the repaired scores (§8).
13. **Decision** (`decision_memo.md`).
    - Do not adopt a merge as an improvement.
    - Stop the merge search.
    - Do not train.
    - One exploratory, evaluation-only follow-up is proposed: serve M050 and CV050 with VERO's interface on the confirmation rows (six runs, roughly 30–37 GPU-hours by the memo's estimate, plus preferably three VERO reruns under P2, roughly 15–20 more). It needs a DESIGN amendment, researcher approval and GPU authorization.

## 2. Experimental setup

The settings below are this study's own. Everything not listed (benchmark data, prompts, scorer code, judge prompts) is the published TRACE evaluation protocol at the pinned revisions, used unchanged.

### 2.1 Models, archive and pinned stack (`configs/trace_merge_qwen25_v1.yaml`)

| role | model | Hugging Face repository | pinned revision | archived model id | declared config dtype |
|---|---|---|---|---|---|
| B | base: the shared backbone | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | `qwen2.5-vl-7b-base` | bfloat16 |
| T | TRACE, released | `maveryn/trace-qwen2.5-vl-7b` | `4d0f1ae8ee25022058090dbdbff61957ece7331d` | `trace-qwen2.5-vl-7b` | float32 |
| V | VERO, released | `zlab-princeton/Vero-Qwen25-7B` | `180e84be5acb2aa887cf51015b84b6a6e453ee90` | `vero-qwen2.5-vl-7b` | float32 |

- **Declared bases.** T: `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`. V: Qwen/Qwen2.5-VL-7B-Instruct (model card; base revision not declared).
- **Archive.** `maveryn/trace-eval-runs@4178a839b689`, run `qwen2.5-vl-7b-comparison-temp06-seeds42-44-v1`: responses, extractions and scores of B, T and V (V@P0) for seeds 42, 43, 44 on all 32,805 rows of the 24 benchmarks. They are never regenerated; every comparison with B, T or V@P0 uses them.
- **Local reruns.** Seed-42 runs of B and T on the full split, made earlier for another study with the same protocol and VLMEvalKit commits and the same prepared data as the new runs (but a different runner code hash), serve only the environment check (§4.3) and one sensitivity analysis (§7.11).
- **Evaluation stack (reused, pinned).** Protocol code `rlvr@af32fe0`, VLMEvalKit `a8b12bf1`, judge `Qwen/Qwen3-32B@9216db57`, served with vLLM 0.10.2 (every one of the 95 server starts logs this version).

### 2.2 Parent compatibility (`audit/compatibility.json`)

- **Architecture.** Identical in all three parents (status `compatible`, no hard stop): `Qwen2_5_VLForConditionalGeneration` with a 28-layer language model (hidden size 3,584, 28 attention and 4 key-value heads, vocabulary 152,064, untied input and output embeddings) and a 32-block vision encoder (hidden size 1,280).
- **Tensors.** 729 tensors and 8,292,166,656 parameters in each parent, all stored as BF16 and matched by name, shape and dtype; no non-floating tensors. The configs declare B bfloat16, T float32 and V float32; vLLM serves every model in bf16.
- **Tokenizer.** The same tokenizer signature in all parents: 151,665 tokens including 22 added tokens, the same vocabulary, and 152,064 embedding rows. The files are not all byte-identical: `tokenizer.json`, `tokenizer_config.json`, `merges.txt` differ between parents, and `added_tokens.json` and `special_tokens_map.json` are shipped (identically) by T and V but not by B. Which files are served is fixed by the inference view (§2.5).
- **Recorded differences** (none affects merging):
  - T: declares Qwen2VLImageProcessorFast vs Qwen2VLImageProcessor; the pinned transformers loads the fast Qwen2-VL processor for both (vLLM log), same pixel bounds and normalization.
  - T: generation eos_token_id 151645 vs [151645, 151643] (stop-token policy belongs to the inference view, not to the weights).
  - V: declares Qwen2VLImageProcessorFast vs Qwen2VLImageProcessor; the pinned transformers loads the fast Qwen2-VL processor for both (vLLM log), same pixel bounds and normalization.
  - The chat templates differ: B ships `chat_template.json`, T and V ship different `chat_template.jinja` files. T's renders the same prompts as B's (§2.5).

### 2.3 How far each parent moved from the base (`audit/parent_update_statistics.json`; diagnostic)

Frobenius norms per tensor group, computed in float64 from the stored BF16 weights. Relative update = ‖parent − B‖ / ‖B‖. The cosine is between the flattened updates T − B and V − B. The all-tensor row is derived from the group rows.

| tensor group | tensors | ‖B‖ | ‖T − B‖ | ‖V − B‖ | T relative update (%) | V relative update (%) | ‖V − B‖ / ‖T − B‖ | cosine(T − B, V − B) | tensors bitwise equal to B (T / V) |
|---|---|---|---|---|---|---|---|---|---|
| language: input embeddings | 1 | 354.0 | 0.173 | 1.192 | 0.049 | 0.337 | 6.90 | −0.0003 | 0 / 0 |
| language: decoder layers | 336 | 2011.6 | 1.881 | 8.156 | 0.093 | 0.405 | 4.34 | 0.0082 | 3 / 3 |
| language: final norm | 1 | 240.7 | 0.000 | 0.000 | 0.000 | 0.000 | – | – | 1 / 1 |
| language: output head | 1 | 292.4 | 0.234 | 1.368 | 0.080 | 0.468 | 5.84 | 0.0029 | 0 / 0 |
| vision: patch embedding | 1 | 17.7 | 0.031 | 0.118 | 0.175 | 0.667 | 3.81 | 0.0645 | 0 / 0 |
| vision: blocks | 384 | 589.6 | 0.572 | 2.669 | 0.097 | 0.453 | 4.67 | 0.0002 | 62 / 58 |
| vision: merger | 5 | 158.7 | 0.160 | 0.694 | 0.101 | 0.438 | 4.34 | 0.0071 | 1 / 1 |
| **all tensors** | 729 | 2165.3 | 1.994 | 8.800 | 0.092 | 0.406 | 4.41 | 0.0073 | 67 / 63 |

- VERO moved the weights 4.4 times as far as TRACE did, and further in every group that changed: the ratio runs from 3.81 (vision: patch embedding) to 6.90 (language: input embeddings).
- The cosines are small: 0.0073 over all tensors, largest in absolute value 0.0645 (vision: patch embedding), against about ±1e-05 for two random directions in 8,292,166,656 dimensions. Averaging therefore cancels little of either update, but each enters a 50:50 merge at half strength. The audit's caveat applies: update norms and cosines do not establish independent skills.
- Untouched by both: language: final norm. Of the 67 (T) and 63 (V) tensors bitwise equal to B, 62 and 58 are in the vision blocks.

### 2.4 Merge construction (`merges/*/merge_manifest.json`, `manifests/merges.json`)

| merge | weights | role | weight-level identity | recipe sha256 | output content revision |
|---|---|---|---|---|---|
| M050 | ½V + ½T | complementarity test | M050 − B = ½(T − B) + ½(V − B) | `215555d7e148…` | `sha256set:eda4ee59e696…` |
| CV050 | ½V + ½B | VERO update-shrinkage control | M050 − CV050 = ½(T − B) | `9f1204920e57…` | `sha256set:99222450add2…` |
| CT050 | ½T + ½B | TRACE update-shrinkage control | M050 − CT050 = ½(V − B) | `82e1fed2e635…` | `sha256set:c1f8a9933e8f…` |

- **Arithmetic.** All 729 tensors are interpolated. Each parent's BF16 values are upcast to float32, combined with the coefficients above, and cast once to bfloat16 (round to nearest even). The untied input embeddings and output head are interpolated as separate tensors. Non-floating buffers would be copied only if bitwise identical in every parent; there are none.
- **Streaming.** Tensors are processed in slices of at most 33,554,432 elements; each output keeps B's tensor-to-shard layout (5 shards).
- **Weights only.** No optimizer, trainer state or gradient is involved. The merged directories hold weights only; the tokenizer, chat template, processor and config come from the inference view (§2.5).
- **Provenance.** Parent shard hashes are in `audit/parent_weight_sha256.json`. Each merge manifest records its output file hashes and build environment (Python 3.12.15, torch 2.8.0+cu128). The three merges take 46.34 GiB.
- **Build code.** The merges were built from an uncommitted working tree on top of `af32fe0` (`implementation_dirty` in every manifest); the hashes of the merge and view code used are in `audit/merge_code_hashes.json` (recorded 2026-11-02T05:27:30Z).

### 2.5 Inference interfaces (`manifests/views.json`, `audit/protocol_comparison.json`)

A merged checkpoint has no chat template of its own, so the non-weight files served with each model's weights (its inference view) are fixed per protocol in advance. Templates are never averaged and never chosen by score.

| protocol | non-weight files from | role | models served | median prompt tokens |
|---|---|---|---|---|
| P0 (native) | each parent's own | the archived canonical runs | B, T, V@P0 | full split: B 674, T 674, V@P0 1,160 |
| P1 (backbone) | B (8 files) | primary, for every merge and control | V@P1, M050, CV050, CT050; B and T need no rerun (their P0 prompts are P1 prompts) | smoke 565, screen 738, confirmation 670 |
| P2 (VERO) | V (11 files) | diagnostic, development screen only, never used to select | B@P2, T@P2, M050@P2, CV050@P2, CT050@P2; V@P2 is the archived V@P0 | screen 1,224 |

Prompt medians differ between tiers because the row sets differ; compare them only within a tier.
- **P1 files:** `config.json`, `generation_config.json`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `chat_template.json`.
- **P2 files:** `config.json`, `generation_config.json`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`, `chat_template.jinja`, `video_preprocessor_config.json`.
- **B and T** render every prompt identically: equal prompt token counts on 100.0% of 98,415 archived rows. T's view lacks `<|endoftext|>` (151643) as an extra stop id; B's and V's list it.
- **V's template** replaces any system message with a fixed reasoning prompt (think inside `<think>`, give the final answer inside `<answer>` with `\boxed{}`) and pre-fills `<think>\n` after the assistant header. On the same rows its prompts are a median 486 tokens longer than B's.
- **Images.** All three parents load the fast Qwen2-VL image processor with the same pixel bounds and normalization.

### 2.6 Decoding, serving and hardware

| setting | value |
|---|---|
| temperature | 0.6 |
| top-p | 1.0 |
| top-k | off (−1) |
| presence penalty | 0.0 |
| repetition penalty | 1.0 |
| max new tokens | 4,096 |
| image pixels | 3,136 to 12,845,056 |
| media transport | file-url |
| weights served as | bf16 |
| decoding seeds | 42 (smoke, screen); 42, 43, 44 (Stage B, confirmation) |

- **One settings set.** Apart from model id and seed, these settings are identical on all 864 archived and new model × seed × benchmark slices (1 distinct set).
- **Generation servers** (65 starts). One vLLM replica per GPU, data-parallel across a node's GPUs; tensor-parallel degree 1 in 64, not logged in 1. GPU memory utilization 0.80 in 32, 0.90 in 32, not logged in 1: 0.90 by default, 0.80 on smaller cards after two A40 servers ran out of memory at 0.90 (A5(b)). Requests in flight per endpoint: 64 on cards below 80 GB and 128 on 80 GB or more (protocol default 32; A7(c)). None of this changes a decoding setting.
- **Judge servers** (30 starts). Qwen3-32B at tensor-parallel degree 1 in 18, 2 in 11, 4 in 1: 1 on ≥90 GB cards and 2 on 44–80 GB cards (A7(c)); the single degree-4 start is the faulty scoring of A8 (§8). Max batched tokens 32768 in 30. Judge weights are verified by marker, the job scripts' default; the timing files record it only for attempts after A7 (A9(c)).
- **GPUs** (card type of the 44 recorded attempts that name their cards): NVIDIA RTX PRO 6000 Blackwell 95GiB (15), NVIDIA A40 44GiB (10), NVIDIA H200 NVL 140GiB (8), NVIDIA RTX 6000 Ada Generation 47GiB (7), NVIDIA A100 80GB PCIe 80GiB (3), NVIDIA A800 40GB Active 40GiB (1).
- **Archive versus new runs.** The archived runs come from the canonical producer (recorded code revision `4475b04`, not assumed identical to the published `af32fe0`); §4.3 measures what the change of environment does.

### 2.7 Scoring and metrics (`manifests/metric_registry.json`, `audit/metric_reproduction.json`)

- **Scorers.** Each benchmark's official primary metric, computed by the pinned scorer. The Qwen3-32B judge is the extraction or grading route on 8 benchmarks: CharXivReason, TableVQABench, EvoChart, MathVision, MathVista, MathVerse, LogicVista, MME-Reasoning. The official VLMEvalKit scorer also writes judge outputs on 7 more (WeMath, MMMU-ProVis, MMStar, SpatialVizBench COT, CV-Bench 3D, ERQA, BLINK), where it grades or extracts answers or matches multiple-choice answers. So 15 of the 24 benchmarks can depend on the judge. On the 8 route benchmarks every row goes through the judge; for the 7 others the bundle does not record how many rows reached it.
- **Fail-closed fixes** (scorer side, DESIGN.md §1). Three are added: the CharXiv judge parser, which the public release omits; MME-Reasoning official-function exceptions are scored incorrect; judge output still unusable after the official five attempts is scored incorrect. They change only rows on which the canonical producer would have raised an error. The bundle does not record how many rows took these paths.
- **Row scores.** Paired resampling needs official scores per row. 16 benchmarks have 0/1 row scores whose mean is the official value. WeMath's official unit is the problem group. On 7 benchmarks the archive holds only the aggregate, so row scores are re-derived from the archived extractions with the pinned scorer functions, with no new judge calls.
- **Adapter acceptance.** Each adapter had to reproduce the archived aggregate of all 9 model × seed slices of its benchmark to 1e-06 pp. All 24 were accepted (0 rejected); the largest error is 4.5e-11 pp.
- **WeMath on subsets** (A2). The pinned score counts every problem group absent from a subset as mastered, so the study uses 100 × (G − 0.5 IG − RM − IK) / G over the G complete problem groups present. It equals the pinned value on the full split.
- **Row diagnostics** (A3). Corrections, regressions, outcome cells, overlap, persistence and capture/damage use the 21 benchmarks whose official value is the mean of 0/1 row scores. ChartQAPro, TableVQABench, WeMath stay in every macro but not in these diagnostics (fractional row scores, a mean of five split scores, and a group metric).
- **Macro.** Equal weight on each of the 24 benchmarks; each of the 6 category scores is the mean of its 4 benchmarks.

### 2.8 Benchmarks, partition and resampling units (`splits/split_manifest.json`, `splits/bootstrap_units.json`)

| category | benchmark | official value | judge | row diagnostics | rows | components | development rows (share) | screen rows | smoke rows | confirmation rows | confirmation rows with a development twin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | aggregate; rows re-derived |  |  | 1,948 | 1,252 | 390 (20.0%) | 128 | 5 | 1,558 |  |
| Charts & Tables | CharXivReason | row mean | route | yes | 1,000 | 1,000 | 200 (20.0%) | 128 | 4 | 800 |  |
| Charts & Tables | TableVQABench | aggregate; rows re-derived | route |  | 1,500 | 752 | 300 (20.0%) | 128 | 7 | 1,200 |  |
| Charts & Tables | EvoChart | row mean | route | yes | 1,250 | 602 | 251 (20.1%) | 128 | 4 | 999 |  |
| Visual Math | MathVision | aggregate; rows re-derived | route | yes | 3,040 | 3,028 | 608 (20.0%) | 128 | 4 | 2,432 |  |
| Visual Math | MathVista | aggregate; rows re-derived | route | yes | 1,000 | 972 | 200 (20.0%) | 128 | 4 | 800 | 45 |
| Visual Math | MathVerse | row mean | route | yes | 788 | 784 | 158 (20.1%) | 128 | 4 | 630 |  |
| Visual Math | WeMath | problem groups (A2) | in scorer |  | 1,740 | 518 | 348 (20.0%) | 129 | 6 | 1,392 |  |
| Science & General | PhyX mini MC | row mean |  | yes | 1,000 | 999 | 200 (20.0%) | 128 | 4 | 800 |  |
| Science & General | MMMU-ProVis | row mean | in scorer | yes | 1,730 | 1,730 | 346 (20.0%) | 128 | 4 | 1,384 |  |
| Science & General | RealWorldQA | row mean |  | yes | 765 | 762 | 153 (20.0%) | 128 | 4 | 612 |  |
| Science & General | MMStar | row mean | in scorer | yes | 1,500 | 1,430 | 300 (20.0%) | 128 | 4 | 1,200 | 62 |
| Spatial Reasoning | EmbSpatial | row mean |  | yes | 3,640 | 2,141 | 728 (20.0%) | 128 | 4 | 2,912 |  |
| Spatial Reasoning | SpatialVizBench COT | row mean | in scorer | yes | 1,180 | 1,133 | 236 (20.0%) | 128 | 4 | 944 |  |
| Spatial Reasoning | CV-Bench 3D | row mean | in scorer | yes | 1,200 | 1,135 | 240 (20.0%) | 128 | 4 | 960 |  |
| Spatial Reasoning | ERQA | row mean | in scorer | yes | 400 | 355 | 82 (20.5%) | 82 | 4 | 318 |  |
| Perception & Counting | BLINK | row mean | in scorer | yes | 1,901 | 1,890 | 380 (20.0%) | 128 | 4 | 1,521 |  |
| Perception & Counting | CountBenchQA | aggregate; rows re-derived |  | yes | 487 | 483 | 97 (19.9%) | 97 | 4 | 390 |  |
| Perception & Counting | CountQA | row mean |  | yes | 1,528 | 1,000 | 307 (20.1%) | 128 | 4 | 1,221 |  |
| Perception & Counting | TreeBench | row mean |  | yes | 405 | 395 | 81 (20.0%) | 81 | 4 | 324 |  |
| Puzzles & Logic | PuzzleVQA | aggregate; rows re-derived |  | yes | 2,000 | 1,616 | 405 (20.3%) | 128 | 32 | 1,595 | 8 |
| Puzzles & Logic | VisualPuzzles | aggregate; rows re-derived |  | yes | 1,168 | 1,152 | 235 (20.1%) | 128 | 9 | 933 |  |
| Puzzles & Logic | LogicVista | row mean | route | yes | 447 | 375 | 89 (19.9%) | 89 | 4 | 358 |  |
| Puzzles & Logic | MME-Reasoning | row mean | route | yes | 1,188 | 1,097 | 243 (20.5%) | 128 | 4 | 945 | 15 |
| **all** |  |  | 8 route, 7 in scorer | 21 | 32,805 | 26,422 | 6,577 (20.0%) | 2,910 | 135 | 26,228 | 130 |

The `all` row counts components that span several benchmarks once; per-benchmark component counts include them.
- **Components.** Rows that share any image (by SHA-256, across all benchmarks) or a WeMath problem family are joined into 26,422 connected components: 22,922 single rows, 179 that span several benchmarks, and at most 28 rows in one. Components are the unit of splitting.
- **Partition** (seed 271828, frozen 2026-11-02T05:16:48+00:00). Each component is ranked by `blake2b("271828|" + component id)`. Cross-benchmark components with rank fraction below 0.20 go to development; then, per benchmark, single-benchmark components join development in rank order until it holds round(0.20 × rows). The rest is confirmation. Every seed and question variant of a component stays in one partition.
- **Development screen.** Per benchmark, development components in rank order until at least 128 rows (all development rows when fewer), plus the best-ranked component carrying each missing value of the strata the official scorer divides by (TableVQABench `split`; PuzzleVQA `category`; VisualPuzzles `category` and `difficulty`).
- **Smoke.** The target is 4 rows per benchmark; whole components and stratum coverage make it 135 rows (A6(c)).
- **Near-duplicate twins** (A7). 465 row pairs across MathVista/MMStar and MME-Reasoning/PuzzleVQA show the same picture re-encoded, with the same question and answer, so byte-level image matching did not join them. The partition is kept; for every bootstrap their components are merged, giving 26,070 resampling units from 26,422 components (frozen 2026-11-02T10:33:49+00:00). The list is a lower bound: resized copies and within-benchmark duplicates were not searched.
- **Resampling units per analysis:** Stage B, full split 26,070; Stage B, development partition 5,242; Stage B, confirmation partition 20,957; development screen 2,421; confirmation 20,957.
- **Prior exposure.** The suite was examined in earlier work. The confirmation partition was reserved from merge tuning, but it is not pristine.

### 2.9 Evaluation matrix

| tier | rows | seeds | new generation runs | reused from the archive | use |
|---|---|---|---|---|---|
| Stage B | 32,805 | 42, 43, 44 | none | B, T, V@P0 | parent behaviour; frozen hypotheses |
| smoke | 135 | 42 | P1: V@P1, M050, CV050, CT050 | none | technical gate for each P1 model (the P2 diagnostic runs had none) |
| development screen | 2,910 | 42 | P1: V@P1, M050, CV050, CT050; P2: B@P2, T@P2, M050@P2, CV050@P2, CT050@P2 | B, T, V@P0 (= V@P2) | descriptive screen; selects nothing |
| confirmation | 26,228 | 42, 43, 44 | P1: V@P1, M050, CV050, CT050 (12 runs) | B, T, V@P0 | primary family and every confirmation table |

- 25 new generation runs in all; 25 are complete and fully scored (`tables/evaluation_status.json`).
- Not run: training of any kind, coefficients other than 0.5, component-wise merges and other merge algorithms (Stage D was not opened).
- Confirmation ran concurrently with the development screen (A4). The confirmed set and the family were fixed in the design, so the screen could not select anything.

### 2.10 Statistics, selection and frozen hypotheses

- **Bootstrap.** 10,000 replicates (seed 314159). Resampling units are drawn with replacement globally, with one draw shared by every model and seed, so every contrast is paired. Benchmark weights stay equal, and WeMath's group metric is recomputed in each replicate.
- **Intervals.** 95%, equal-tailed percentile interval of the replicate distribution (numpy linear interpolation).
- **p-values.** Two-sided percentile-inversion bootstrap p-values: p = min(1, 2 (min(#{d* ≤ 0}, #{d* ≥ 0}) + 1) / (R + 1)) over the R replicate differences d*. p < 0.05 exactly when 0 lies outside the 95% interval, up to the add-one correction.
- **Multiplicity.** Holm over the primary family only: M050 − T, M050 − V@P1, M050 − CV050, M050 − CT050 (confirmation, 24-benchmark macro). Everything else is descriptive or exploratory and unadjusted.
- **Small slices.** The behaviour and capture/damage tables flag slices with fewer than 100 resampling groups; the only flagged ones are the development screen's capture/damage slices at benchmark level, which this file does not report. Score and contrast tables carry no such flag: on the development screen 10 benchmarks have fewer than 100 components (ChartQAPro, TableVQABench, EvoChart, WeMath, EmbSpatial, ERQA, CountBenchQA, CountQA, TreeBench, LogicVista), so their per-benchmark values (§6.7–§6.8) and intervals (§6.9–§6.10) rest on few groups.
- **Selection.** None. There is no coefficient search, so all three midpoints and both controls went to confirmation as prespecified. No category regression margin or tie tolerance was set, so no noninferiority or no-harm claim is made.
- **Lock.** Before any confirmation run had scores, `reports/selection_lock.json` froze the config, design, split, merge and view hashes and the confirmed labels and seeds (2026-11-02T09:04:27+00:00). `confirm` re-checks them; amendments appended since the lock: A5, A6, A7, A8, A9. The lock does not record `accepted_benchmarks`, `design_amendments_at_lock`, `metric_registry_sha256`, `runner_code_sha256`, `split_config_sha256`, so it cannot show runner changes: the pandas guard was added (A5) and then narrowed (A7(b)) after it.

Frozen hypotheses (`behavior_report.md` §6, written from Stage B and frozen at 2026-11-02T07:08:36Z, before any merged model had generated a response) and their outcome on confirmation:

| hypothesis | frozen prediction | confirmation outcome | verdict |
|---|---|---|---|
| H1: integration beyond shrinkage | M050 keeps more TRACE-only successes than CV050, and more VERO-only successes than CT050 (independent repeat) | TRACE-only: M050 49.0% vs CV050 46.1%; VERO-only: M050 52.8% vs CT050 36.4%; same direction in all three variants | holds in direction; but the base model itself keeps 47.6% of TRACE-only successes (post hoc) |
| H2: the category trade-off transfers | M050 − CV050 is largest in Charts & Tables; M050 − CT050 is largest in Visual Math | M050 − CV050 largest in Science & General (+2.33; Charts & Tables +1.59); M050 − CT050 largest in Visual Math (+3.73) | half supported |
| H3: part of VERO's gain is interface-dependent | V@P1 < V@P0 on the macro, with the largest drops in Visual Math and Puzzles & Logic | macro −2.94; largest drops Puzzles & Logic (−11.20) and Charts & Tables (−3.89); Visual Math −1.77 | supported for the macro and for Puzzles & Logic; not for Visual Math |

### 2.11 Design amendments and timeline (`DESIGN.md`)

Amendment stamps are DESIGN.md's, in MST (UTC−7); file timestamps are UTC.

| amendment | written (MST) | subject | what it changed | effect on reported numbers |
|---|---|---|---|---|
| A1 | 2026-11-02 | row identity | Archived CountQA rows take the prepared manifest's row hash: the public code hashes the cached answer as a string, the canonical producer as an integer (both recomputed on all 1,528 rows) | none |
| A2 | 2026-11-02 | WeMath on subsets | Study metric over the complete problem groups present (§2.7) | WeMath subset values; full-split values unchanged |
| A3 | 2026-11-02 | row diagnostics | Binary diagnostics restricted to the 21 benchmarks with 0/1 official rows | defines the diagnostic set |
| A4 | 2026-11-02 02:05 | schedule | Confirmation runs concurrently with the development screen, to use idle GPUs | none |
| A5 | 2026-11-02 02:15 | environment | A pandas guard against a parser crash on hash strings; GPU memory utilization 0.80 on 44–48 GB cards | none |
| A6 | 2026-11-02 02:37 | record | Amendment timing, the lock's prefix rule for DESIGN.md, the 135-row smoke tier | none |
| A7 | 2026-11-02 03:35 | resampling units and serving | Near-duplicate twins joined into bootstrap units, with a twin-free sensitivity analysis; guard wording; serving parameters | intervals (those quoted in the frozen Stage B report moved by at most 0.10 pp, per analysis_report.md); point estimates unchanged |
| A8 | 2026-11-03 00:10 | judge fault | M050 confirmation seed 44 rescored with the judge at tensor-parallel 1 after a degraded tensor-parallel-4 scoring; faulty outputs moved to quarantine, generation kept | M050 seed-44 scores on judge-dependent benchmarks; the M050 − T estimate moved from −0.29 (superseded tables) to +0.23 (§8.2) |
| A9 | 2026-11-03 02:10 | record | Corrects A8's description of the fault mechanism and A7(c)'s note on timing files | none |

| event | time (UTC) |
|---|---|
| split frozen | 2026-11-02T05:16:48+00:00 |
| selection lock written | 2026-11-02T09:04:27+00:00 |
| bootstrap units frozen (A7) | 2026-11-02T10:33:49+00:00 |

## 3. Runs

Every new generation run, with its final attempt's hardware. Source: `../tables/evaluation_costs.csv` and each run's `timing.seed<S>.json`.
- B, T and V@P0 are not re-run; they come from the archive.
- GPU-hours and the attempts column count the 41 attempts with a timed phase. The timing files record 47 attempts; 6 stopped right after starting, with no timed phase (smoke M050, smoke V@P1, V@P1 confirmation seed 44).
- Wall hours and GPU-hours sum all timed attempts; a resumed attempt continues where the previous one stopped.
- Generated tokens are counted from the ingested responses.

| tier | model | seed | rows | attempts | final attempt: host, GPUs | wall hours | GPU-hours | generated tokens |
|---|---|---|---|---|---|---|---|---|
| smoke | CT050 | 42 | 135 | 1 | notch426, 3x NVIDIA A40 44GiB | 0.27 | 0.80 | 25,976 |
| smoke | CV050 | 42 | 135 | 1 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 0.25 | 0.25 | 29,439 |
| smoke | M050 | 42 | 135 | 4 | grn067, 1x NVIDIA H200 NVL 140GiB | 0.38 | 0.38 | 26,398 |
| smoke | V@P1 | 42 | 135 | 2 | grn067, 1x NVIDIA H200 NVL 140GiB | 0.17 | 0.17 | 27,209 |
| development screen | CT050 | 42 | 2,910 | 1 | notch426, 3x NVIDIA A40 44GiB | 0.46 | 1.39 | 608,038 |
| development screen | CT050@P2 | 42 | 2,910 | 1 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 0.44 | 0.44 | 761,700 |
| development screen | CV050 | 42 | 2,910 | 1 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 0.44 | 0.44 | 599,284 |
| development screen | CV050@P2 | 42 | 2,910 | 1 | notch426, 2x NVIDIA A40 44GiB | 0.55 | 1.09 | 722,222 |
| development screen | M050 | 42 | 2,910 | 1 | grn067, 1x NVIDIA H200 NVL 140GiB | 0.24 | 0.24 | 651,122 |
| development screen | M050@P2 | 42 | 2,910 | 1 | notch369, 2x NVIDIA A100 80GB PCIe 80GiB | 0.45 | 0.90 | 793,755 |
| development screen | B@P2 | 42 | 2,910 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 0.64 | 1.29 | 718,888 |
| development screen | T@P2 | 42 | 2,910 | 1 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 0.44 | 0.44 | 812,149 |
| development screen | V@P1 | 42 | 2,910 | 1 | grn067, 1x NVIDIA H200 NVL 140GiB | 0.24 | 0.24 | 601,447 |
| confirmation | CT050 | 42 | 26,228 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 3.39 | 6.78 | 5,475,899 |
| confirmation | CT050 | 43 | 26,228 | 2 | notch369, 2x NVIDIA A100 80GB PCIe 80GiB | 2.48 | 3.15 | 5,480,405 |
| confirmation | CT050 | 44 | 26,228 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 1.90 | 3.80 | 5,388,013 |
| confirmation | CV050 | 42 | 26,228 | 2 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 2.42 | 2.42 | 5,417,796 |
| confirmation | CV050 | 43 | 26,228 | 2 | notch426, 3x NVIDIA A40 44GiB | 2.39 | 7.18 | 5,475,198 |
| confirmation | CV050 | 44 | 26,228 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 2.53 | 5.07 | 5,298,910 |
| confirmation | M050 | 42 | 26,228 | 1 | grn073, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 2.42 | 2.42 | 5,911,585 |
| confirmation | M050 | 43 | 26,228 | 2 | grn067, 1x NVIDIA H200 NVL 140GiB | 3.97 | 3.97 | 5,994,008 |
| confirmation | M050 | 44 | 26,228 | 2 | grn072, 2x NVIDIA RTX PRO 6000 Blackwell 95GiB (scoring redone, A8) | 1.11 | 4.88 | 5,904,482 |
| confirmation | V@P1 | 42 | 26,228 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 2.64 | 5.29 | 5,370,273 |
| confirmation | V@P1 | 43 | 26,228 | 2 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 2.45 | 2.45 | 5,428,293 |
| confirmation | V@P1 | 44 | 26,228 | 2 | grn067, 1x NVIDIA H200 NVL 140GiB | 2.09 | 2.09 | 5,345,857 |

Totals: 57.57 recorded GPU-hours over 41 attempts (15 failed or incomplete, all resumed) and 72,868,346 generated tokens. Merged weights take 46.34 GiB; run directories take 5.33 GiB.

One attempt is not in the timing files: M050 confirmation seed 44 first generated all its rows on 2× A40 in about 2.2 hours by its step log (about 4.5 GPU-hours) before its allocation expired. Its rows were kept.

## 4. Measurement checks

### 4.1 Prompts (`audit/protocol_comparison.json`)

Each run's prompts were compared row by row with an archived run of the same seed: P1 runs with archived B (B and T render identically), P2 runs with archived V@P0, and each local rerun with its own archived model. The prompt digest covers the message payload before the chat template is applied; the rendered prompt's token count is the check on the template.

| tier, protocol | archived reference | runs | conforming | rows | same prompt payload digest | same rendered prompt token count | same request digest without model id and seed |
|---|---|---|---|---|---|---|---|
| smoke, P1 | B | 4 | 4 | 540 | 540 | 540 | 540 |
| development screen, P1 | B | 4 | 4 | 11,640 | 11,640 | 11,640 | 11,640 |
| development screen, P2 | V@P0 | 5 | 5 | 14,550 | 14,550 | 14,550 | 14,550 |
| confirmation, P1 | B | 12 | 12 | 314,736 | 314,736 | 314,736 | 314,736 |
| local reruns, full split | B | 1 | 1 | 32,805 | 32,805 | 32,805 | 32,805 |
| local reruns, full split | T | 1 | 1 | 32,805 | 32,805 | 32,805 | 32,805 |


Ingestion (the work root's per-run ingest reports): 25 of the 25 new runs are `verified`, with the study's aggregate equal to the run's own score-campaign aggregate on all 24 benchmark slices. For the nine development-screen runs, `merge_screen.md` also reports the largest adapter difference (below 1e-14) and that the served weights hash to the audited parent or merge.

### 4.2 Completeness (`audit/missingness.csv`)

Across all 864 model × seed × benchmark slices (216 archived, 48 local-rerun, 600 new-run), 702,321 rows were expected and 702,321 have responses: 0 missing, 0 extra, 0 excluded and 0 without a finish reason. The 464,807 rows of the row-scored benchmarks all carry a stored row score (0 unavailable); the 237,514 rows of the aggregate-only benchmarks store none, and their row scores are re-derived (§2.7).

### 4.3 Environment: local reruns of B and T against the archive (`audit/protocol_comparison.json`, `tables/environment_drift.csv`)

The archived runs come from the canonical producer; the new runs from the published code on the GPUs listed in §2.6. Seed-42 reruns of B and T on the full split in the new environment are compared with the archived seed-42 runs. As a yardstick, the same quantities are shown for the three pairs of archived seeds (42–43, 42–44, 43–44), which differ only in decoding seed.

| quantity | B: rerun − archive, seed 42 | B: archived seed pairs | T: rerun − archive, seed 42 | T: archived seed pairs |
|---|---|---|---|---|
| macro difference (pp) | −0.25 | −0.53 to −0.04 | −0.02 | −0.28 to +0.32 |
| mean absolute benchmark difference (pp) | 0.56 | 1.40 to 1.56 | 0.81 | 0.86 to 0.95 |
| largest absolute benchmark difference (pp) | 1.52 | 5.30 to 6.80 | 2.46 | 2.47 to 4.20 |
| row-score agreement (%) | 92.6 | 78.2 to 81.9 | 92.4 | 83.5 to 85.3 |
| identical response text (%) | 57.2 | 26.1 to 29.1 | 49.4 | 23.2 to 24.9 |
| mean paired change in completion tokens | −11.8 | −7.8 to +2.6 | −0.6 | −5.1 to +2.5 |
| length stops (rerun; archived seeds 42 / 43 / 44) | 11 | 107 / 112 / 94 | 0 | 4 / 7 / 3 |

- The macro difference is within the seed-pair range for both B and T. The mean and the largest absolute per-benchmark differences are below the seed-pair range for both.
- Responses are where the environment shows: the mean completion-token change is below the seed-pair range for B and within it for T, and both reruns have fewer length stops than any archived seed.
- Row agreement and identical text are higher than between archived seeds because each rerun shares its sampling seed with the archived slice it is compared with; they mix that seed coupling with environment drift.
- T's rerun was served from the protocol clone's 18-file snapshot, while the archived T used a content-bound 10-file view of the same weights. Prompt tokens agree on every row, but T's comparison also contains that view difference.
- The confirmation contrasts with local-rerun B and T are in §7.11.

Per benchmark, seed 42 (`tables/environment_drift.csv`):

| category | benchmark | B archive | B rerun | B difference | B largest seed-pair difference | B row agreement (%) | T archive | T rerun | T difference | T largest seed-pair difference | T row agreement (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 45.73 | 45.01 | −0.72 | 0.45 | – | 47.80 | 47.35 | −0.44 | 0.71 | – |
| Charts & Tables | CharXivReason | 39.50 | 39.30 | −0.20 | 2.30 | 98.2 | 47.50 | 46.20 | −1.30 | 0.70 | 93.1 |
| Charts & Tables | TableVQABench | 74.16 | 74.03 | −0.13 | 1.63 | – | 78.48 | 78.61 | +0.13 | 0.35 | – |
| Charts & Tables | EvoChart | 57.44 | 56.64 | −0.80 | 1.28 | 96.3 | 64.96 | 64.56 | −0.40 | 0.08 | 95.0 |
| Visual Math | MathVision | 25.49 | 25.10 | −0.39 | 1.48 | – | 26.74 | 27.43 | +0.69 | 1.22 | – |
| Visual Math | MathVista | 68.90 | 67.90 | −1.00 | 0.70 | – | 72.90 | 73.60 | +0.70 | 0.90 | – |
| Visual Math | MathVerse | 42.39 | 40.86 | −1.52 | 1.65 | 83.2 | 47.46 | 47.59 | +0.13 | 0.89 | 87.2 |
| Visual Math | WeMath | 37.71 | 37.52 | −0.19 | 5.71 | 87.6 | 47.62 | 45.52 | −2.10 | 2.57 | 89.9 |
| Science & General | PhyX mini MC | 38.20 | 39.30 | +1.10 | 6.80 | 94.3 | 49.40 | 48.20 | −1.20 | 1.60 | 81.2 |
| Science & General | MMMU-ProVis | 35.84 | 35.72 | −0.12 | 0.75 | 82.5 | 38.55 | 40.29 | +1.73 | 1.33 | 86.9 |
| Science & General | RealWorldQA | 66.01 | 66.27 | +0.26 | 1.57 | 99.0 | 68.10 | 68.37 | +0.26 | 1.18 | 97.6 |
| Science & General | MMStar | 62.87 | 63.00 | +0.13 | 1.67 | 97.2 | 65.73 | 65.00 | −0.73 | 0.67 | 96.9 |
| Spatial Reasoning | EmbSpatial | 70.36 | 70.19 | −0.16 | 1.79 | 98.8 | 71.24 | 71.15 | −0.08 | 1.29 | 98.9 |
| Spatial Reasoning | SpatialVizBench COT | 35.08 | 34.07 | −1.02 | 0.17 | 75.8 | 35.34 | 35.42 | +0.08 | 0.68 | 81.9 |
| Spatial Reasoning | CV-Bench 3D | 76.75 | 76.42 | −0.33 | 3.83 | 92.0 | 80.75 | 82.08 | +1.33 | 0.75 | 92.5 |
| Spatial Reasoning | ERQA | 40.75 | 41.25 | +0.50 | 3.00 | 99.0 | 41.50 | 42.25 | +0.75 | 2.50 | 98.3 |
| Perception & Counting | BLINK | 52.76 | 52.76 | +0.00 | 2.26 | 98.0 | 56.76 | 55.92 | −0.84 | 1.42 | 95.1 |
| Perception & Counting | CountBenchQA | 83.78 | 83.98 | +0.21 | 2.87 | – | 86.45 | 86.24 | −0.21 | 2.67 | – |
| Perception & Counting | CountQA | 18.26 | 19.04 | +0.79 | 2.23 | 95.3 | 21.47 | 21.79 | +0.33 | 1.64 | 94.6 |
| Perception & Counting | TreeBench | 36.79 | 37.28 | +0.49 | 2.72 | 98.5 | 40.00 | 38.27 | −1.73 | 4.20 | 97.8 |
| Puzzles & Logic | PuzzleVQA | 44.25 | 43.35 | −0.90 | 1.15 | – | 50.20 | 49.80 | −0.40 | 0.95 | – |
| Puzzles & Logic | VisualPuzzles | 30.91 | 31.08 | +0.17 | 0.26 | – | 34.76 | 35.62 | +0.86 | 1.54 | – |
| Puzzles & Logic | LogicVista | 44.97 | 43.85 | −1.12 | 4.47 | 81.9 | 47.20 | 49.66 | +2.46 | 1.34 | 83.2 |
| Puzzles & Logic | MME-Reasoning | 26.01 | 24.92 | −1.09 | 2.19 | 86.1 | 28.70 | 28.11 | −0.59 | 1.68 | 87.1 |

Row agreement is shown only where row scores exist in both runs. The largest seed-pair difference is the largest absolute difference between two archived seeds of the same model on that benchmark.
- The rerun difference is larger than the largest archived seed-pair difference on 3 of 24 benchmarks for B (ChartQAPro, MathVista, SpatialVizBench COT) and 6 for T (CharXivReason, EvoChart, MMMU-ProVis, MMStar, CV-Bench 3D, LogicVista).

### 4.4 Archive ingestion (`audit/archive_ingest.json`)

Status `verified`: 295,245 archived rows in 216 model × seed × benchmark slices, 0 problems. Checks run:
- Archive manifest sha256 equals the configured receipt.
- Every part's sha256/size/part manifest verified.
- Response/extraction/score digests recomputed with the pinned exporter helpers.
- Content identity unique per slice; extraction and row-score joins one-to-one on identity + response id.
- Every row matches exactly one dataset-manifest row (5-tuple + media set + ordered media), directly or through a verified producer row-hash alias (PRODUCER_ROW_HASH_ALIASES; originals kept in archived_*).
- Identical key sets across the 9 slices of each benchmark.
- One normalized generation setting (minus model_id/seed) across slices, equal to protocol.generation.
- Slice aggregates equal the archive results file.

## 5. Stage B: TRACE and VERO against the base (archived runs, full split, seeds 42–44)

### 5.1 Official scores (`stage_b/official_scores.all.csv`)

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B | 47.93 [47.41, 48.45] | 54.45 | 43.03 | 51.00 | 54.95 | 48.26 | 35.87 |
| T | 51.99 [51.43, 52.52] | 59.60 | 48.68 | 55.55 | 57.20 | 51.03 | 39.85 |
| V@P0 | 52.22 [51.62, 52.83] | 57.18 | 50.70 | 55.48 | 57.59 | 51.28 | 41.11 |

### 5.2 Contrasts (`stage_b/differences.all.csv`; categories exploratory, unadjusted)

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| T − B | +4.06 [+3.66, +4.45] | < 0.001 | +5.15 [+4.27, +5.99] | +5.65 [+4.64, +6.66] | +4.55 [+3.64, +5.49] | +2.25 [+1.32, +3.17] | +2.77 [+1.92, +3.65] | +3.99 [+2.81, +5.14] |
| V@P0 − B | +4.30 [+3.78, +4.82] | < 0.001 | +2.73 [+1.73, +3.73] | +7.67 [+6.54, +8.77] | +4.48 [+3.31, +5.64] | +2.64 [+1.25, +4.07] | +3.02 [+1.64, +4.38] | +5.24 [+3.79, +6.71] |
| T − V@P0 | −0.24 [−0.75, +0.28] | 0.36 | +2.42 [+1.40, +3.43] | −2.02 [−3.15, −0.86] | +0.07 [−0.95, +1.13] | −0.39 [−1.82, +0.98] | −0.25 [−1.64, +1.16] | −1.25 [−2.73, +0.22] |

### 5.3 Per-benchmark scores and contrasts (`stage_b/official_scores.all.csv`, `stage_b/differences.all.csv`; exploratory, unadjusted)

| category | benchmark | B | T | V@P0 | T − B [95% CI] | V@P0 − B [95% CI] | T − V@P0 [95% CI] |
|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 45.81 | 48.05 | 40.87 | +2.24 [+0.83, +3.65] | −4.94 [−6.91, −2.98] | +7.18 [+5.07, +9.33] |
| Charts & Tables | CharXivReason | 39.73 | 47.13 | 45.83 | +7.40 [+5.27, +9.52] | +6.10 [+3.62, +8.58] | +1.30 [−0.97, +3.60] |
| Charts & Tables | TableVQABench | 75.20 | 78.31 | 78.47 | +3.11 [+1.92, +4.34] | +3.27 [+1.75, +4.84] | −0.16 [−1.80, +1.48] |
| Charts & Tables | EvoChart | 57.07 | 64.91 | 63.55 | +7.84 [+5.95, +9.71] | +6.48 [+4.56, +8.44] | +1.36 [−0.65, +3.32] |
| Visual Math | MathVision | 24.81 | 27.42 | 28.15 | +2.61 [+1.53, +3.68] | +3.33 [+2.07, +4.60] | −0.72 [−1.94, +0.45] |
| Visual Math | MathVista | 68.67 | 73.37 | 76.80 | +4.70 [+2.92, +6.53] | +8.13 [+6.16, +10.07] | −3.43 [−5.38, −1.48] |
| Visual Math | MathVerse | 43.44 | 47.76 | 50.93 | +4.31 [+2.11, +6.53] | +7.49 [+5.03, +10.00] | −3.17 [−5.61, −0.71] |
| Visual Math | WeMath | 35.20 | 46.16 | 46.92 | +10.96 [+8.33, +13.71] | +11.72 [+8.87, +14.60] | −0.76 [−3.84, +2.41] |
| Science & General | PhyX mini MC | 40.97 | 48.70 | 46.63 | +7.73 [+5.11, +10.39] | +5.67 [+2.74, +8.57] | +2.07 [−0.26, +4.49] |
| Science & General | MMMU-ProVis | 35.70 | 39.36 | 39.67 | +3.66 [+2.19, +5.13] | +3.97 [+2.32, +5.63] | −0.31 [−1.96, +1.37] |
| Science & General | RealWorldQA | 65.45 | 68.50 | 69.85 | +3.05 [+1.41, +4.74] | +4.40 [+1.75, +7.01] | −1.35 [−3.83, +1.16] |
| Science & General | MMStar | 61.89 | 65.64 | 65.78 | +3.76 [+2.38, +5.15] | +3.89 [+2.06, +5.72] | −0.13 [−1.82, +1.52] |
| Spatial Reasoning | EmbSpatial | 69.42 | 70.95 | 70.60 | +1.53 [+0.94, +2.11] | +1.18 [+0.05, +2.34] | +0.35 [−0.81, +1.46] |
| Spatial Reasoning | SpatialVizBench COT | 35.14 | 35.68 | 32.06 | +0.54 [−1.67, +2.78] | −3.08 [−5.79, −0.31] | +3.62 [+0.86, +6.36] |
| Spatial Reasoning | CV-Bench 3D | 76.25 | 81.00 | 83.11 | +4.75 [+3.08, +6.44] | +6.86 [+4.80, +8.88] | −2.11 [−4.10, −0.17] |
| Spatial Reasoning | ERQA | 39.00 | 41.17 | 44.58 | +2.17 [−0.17, +4.47] | +5.58 [+1.23, +10.16] | −3.42 [−7.80, +0.96] |
| Perception & Counting | BLINK | 53.31 | 56.57 | 57.51 | +3.26 [+2.02, +4.54] | +4.21 [+2.06, +6.32] | −0.95 [−2.87, +1.02] |
| Perception & Counting | CountBenchQA | 82.14 | 84.80 | 82.96 | +2.67 [+1.02, +4.30] | +0.82 [−2.10, +3.68] | +1.85 [−0.91, +4.68] |
| Perception & Counting | CountQA | 19.59 | 22.51 | 23.17 | +2.92 [+1.32, +4.58] | +3.58 [+1.73, +5.49] | −0.65 [−2.72, +1.32] |
| Perception & Counting | TreeBench | 38.02 | 40.25 | 41.48 | +2.22 [+0.00, +4.53] | +3.46 [−0.24, +7.26] | −1.23 [−5.18, +2.63] |
| Puzzles & Logic | PuzzleVQA | 44.57 | 50.02 | 49.22 | +5.45 [+3.64, +7.28] | +4.65 [+2.39, +6.94] | +0.80 [−1.17, +2.78] |
| Puzzles & Logic | VisualPuzzles | 31.08 | 34.02 | 36.96 | +2.94 [+0.87, +5.03] | +5.88 [+3.43, +8.34] | −2.94 [−5.37, −0.50] |
| Puzzles & Logic | LogicVista | 42.65 | 47.35 | 47.65 | +4.70 [+1.34, +7.98] | +5.00 [+0.93, +9.06] | −0.30 [−4.71, +4.13] |
| Puzzles & Logic | MME-Reasoning | 25.17 | 28.03 | 30.61 | +2.86 [+1.02, +4.72] | +5.44 [+3.25, +7.61] | −2.58 [−4.75, −0.40] |

### 5.4 Row-level behaviour (21 binary-eligible benchmarks, 27,617 rows per seed; mean over seeds, rows pooled)

Corrections and regressions against B (`stage_b/corrections_regressions.csv`):

| model | correction rate (% of rows) | regression rate | net [95% CI] | corrections / B errors | regressions / B successes |
|---|---|---|---|---|---|
| T | 12.31 | 8.66 | +3.65 [+3.28, +4.01] | 23.2% | 18.4% |
| V@P0 | 14.52 | 10.49 | +4.03 [+3.57, +4.51] | 27.4% | 22.3% |

Eight outcome cells, % of rows (`stage_b/joint_outcomes.csv`):

|  | B = 0 | B = 1 |
|---|---|---|
| T = 0, V@P0 = 0 | 33.08 | 5.21 |
| T = 0, V@P0 = 1 | 7.63 | 3.45 |
| T = 1, V@P0 = 0 | 5.42 | 5.28 |
| T = 1, V@P0 = 1 | 6.89 | 33.03 |

Parent overlap (`stage_b/parent_overlap.csv`):
- T-only 10.7%, V-only 11.1%, both correct 39.9%, both wrong 38.3%; agreement 78.2% [77.8, 78.6].
- The hindsight "either parent correct" rate is 61.7% [61.1, 62.3]. It uses two responses and correctness labels, so it is neither a router nor a bound on a merge.

Persistence of outcome sets across seeds (`stage_b/persistence.csv`; defined on one seed, checked on another, all ordered seed pairs):

| set | persists (%) | label-swap null (%) | excess (pp) [95% CI] | T's label recurs (%) | V@P0's label recurs (%) | B's label recurs (%) |
|---|---|---|---|---|---|---|
| T-only (T=1, V@P0=0) | 47.1 | 26.9 | +20.2 [+19.2, +21.3] | 58.5 | 83.4 | – |
| V-only (T=0, V@P0=1) | 50.0 | 26.9 | +23.1 [+22.1, +24.1] | 65.5 | 79.5 | – |
| T corrections (B=0, T=1) | 32.9 | 20.5 | +12.4 [+11.6, +13.2] | 63.1 | – | 59.5 |
| V corrections (B=0, V@P0=1) | 49.3 | 25.2 | +24.0 [+23.2, +24.8] | – | 82.6 | 62.2 |
| T regressions (B=1, T=0) | 23.3 | 20.5 | +2.8 [+1.8, +3.8] | 67.2 | – | 41.5 |
| V regressions (B=1, V@P0=0) | 39.7 | 25.2 | +14.4 [+13.4, +15.4] | – | 85.4 | 48.0 |

A set persists on the other seed when every label of its pattern recurs there; the last three columns show how often each model's own label recurs. On T-only rows, T's success recurs 58.5% of the time and V@P0's failure 83.4%. On V-only rows, V@P0's success recurs 79.5% and T's failure 65.5%.

### 5.5 Macro per seed (`stage_b/seed_scores.all.csv`; unweighted mean of the 24 benchmarks)

| model | seed 42 | seed 43 | seed 44 | mean | range |
|---|---|---|---|---|---|
| B | 48.12 | 48.08 | 47.59 | 47.93 | 0.53 |
| T | 52.07 | 51.79 | 52.10 | 51.99 | 0.32 |
| V@P0 | 52.31 | 52.18 | 52.18 | 52.22 | 0.13 |

### 5.6 The same comparisons on each partition (`stage_b/official_scores.*.csv`, `stage_b/differences.*.csv`)

| model or contrast (macro) | full split (32,805 rows) | development partition (6,577 rows) | confirmation partition (26,228 rows) |
|---|---|---|---|
| B | 47.93 [47.41, 48.45] | 47.23 [46.04, 48.40] | 48.10 [47.53, 48.68] |
| T | 51.99 [51.43, 52.52] | 51.42 [50.20, 52.61] | 52.13 [51.51, 52.74] |
| V@P0 | 52.22 [51.62, 52.83] | 52.19 [50.82, 53.55] | 52.23 [51.56, 52.90] |
| T − B | +4.06 [+3.66, +4.45] | +4.19 [+3.30, +5.10] | +4.02 [+3.58, +4.45] |
| V@P0 − B | +4.30 [+3.78, +4.82] | +4.96 [+3.81, +6.12] | +4.13 [+3.55, +4.71] |
| T − V@P0 | −0.24 [−0.75, +0.28] | −0.77 [−1.90, +0.38] | −0.11 [−0.68, +0.47] |

Category contrasts on the confirmation partition (`stage_b/differences.confirmation.csv`):

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| T − B | +4.02 [+3.58, +4.45] | < 0.001 | +5.11 [+4.17, +6.05] | +5.76 [+4.66, +6.91] | +4.48 [+3.44, +5.50] | +1.95 [+0.93, +2.94] | +2.80 [+1.83, +3.81] | +4.03 [+2.75, +5.32] |
| V@P0 − B | +4.13 [+3.55, +4.71] | < 0.001 | +3.09 [+1.98, +4.20] | +7.49 [+6.28, +8.74] | +4.40 [+3.08, +5.72] | +2.65 [+1.06, +4.22] | +2.54 [+1.00, +4.06] | +4.61 [+3.07, +6.19] |
| T − V@P0 | −0.11 [−0.68, +0.47] | 0.71 | +2.02 [+0.88, +3.12] | −1.72 [−3.00, −0.45] | +0.09 [−1.10, +1.27] | −0.70 [−2.24, +0.85] | +0.26 [−1.32, +1.81] | −0.58 [−2.18, +1.01] |

Category contrasts on the development partition (`stage_b/differences.development.csv`):

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| T − B | +4.19 [+3.30, +5.10] | < 0.001 | +5.29 [+3.36, +7.24] | +5.17 [+3.02, +7.37] | +4.82 [+2.78, +6.85] | +3.41 [+1.33, +5.53] | +2.64 [+0.79, +4.51] | +3.83 [+0.96, +6.52] |
| V@P0 − B | +4.96 [+3.81, +6.12] | < 0.001 | +1.30 [−0.96, +3.55] | +8.37 [+5.92, +10.79] | +4.82 [+2.30, +7.36] | +2.59 [−0.79, +5.89] | +4.92 [+1.93, +7.97] | +7.76 [+4.43, +11.35] |
| T − V@P0 | −0.77 [−1.90, +0.38] | 0.18 | +3.99 [+1.66, +6.30] | −3.20 [−5.51, −0.86] | +0.00 [−2.18, +2.16] | +0.82 [−2.57, +4.28] | −2.28 [−5.06, +0.48] | −3.93 [−7.71, −0.45] |

### 5.7 Corrections and regressions per benchmark (`stage_b/corrections_regressions.csv`; mean over seeds; % of rows)

| category | benchmark | T corrections | T regressions | T net [95% CI] | V@P0 corrections | V@P0 regressions | V@P0 net [95% CI] |
|---|---|---|---|---|---|---|---|
| Charts & Tables | CharXivReason | 15.4 | 8.0 | +7.40 [+5.28, +9.49] | 16.1 | 10.0 | +6.10 [+3.67, +8.60] |
| Charts & Tables | EvoChart | 14.6 | 6.7 | +7.84 [+5.99, +9.75] | 13.3 | 6.8 | +6.48 [+4.56, +8.44] |
| Visual Math | MathVision | 12.6 | 10.0 | +2.61 [+1.55, +3.68] | 13.8 | 10.4 | +3.33 [+2.09, +4.61] |
| Visual Math | MathVista | 11.4 | 6.7 | +4.70 [+2.85, +6.48] | 13.2 | 5.0 | +8.13 [+6.17, +10.10] |
| Visual Math | MathVerse | 13.9 | 9.6 | +4.31 [+2.13, +6.51] | 16.5 | 9.1 | +7.49 [+5.03, +10.05] |
| Science & General | PhyX mini MC | 24.1 | 16.4 | +7.73 [+5.06, +10.41] | 23.8 | 18.1 | +5.67 [+2.71, +8.59] |
| Science & General | MMMU-ProVis | 14.0 | 10.4 | +3.66 [+2.18, +5.12] | 14.3 | 10.3 | +3.97 [+2.25, +5.61] |
| Science & General | RealWorldQA | 6.5 | 3.5 | +3.05 [+1.40, +4.68] | 11.3 | 6.9 | +4.40 [+1.79, +7.05] |
| Science & General | MMStar | 8.6 | 4.8 | +3.76 [+2.39, +5.16] | 12.0 | 8.1 | +3.89 [+2.08, +5.74] |
| Spatial Reasoning | EmbSpatial | 3.9 | 2.4 | +1.53 [+0.95, +2.13] | 8.8 | 7.6 | +1.18 [+0.04, +2.34] |
| Spatial Reasoning | SpatialVizBench COT | 18.6 | 18.1 | +0.54 [−1.73, +2.75] | 17.1 | 20.2 | −3.08 [−5.80, −0.32] |
| Spatial Reasoning | CV-Bench 3D | 12.4 | 7.6 | +4.75 [+3.04, +6.45] | 14.1 | 7.2 | +6.86 [+4.82, +8.95] |
| Spatial Reasoning | ERQA | 6.8 | 4.7 | +2.17 [−0.16, +4.45] | 16.6 | 11.0 | +5.58 [+1.11, +9.89] |
| Perception & Counting | BLINK | 9.3 | 6.0 | +3.26 [+2.01, +4.54] | 17.7 | 13.5 | +4.21 [+2.11, +6.31] |
| Perception & Counting | CountBenchQA | 4.9 | 2.3 | +2.67 [+1.03, +4.32] | 8.1 | 7.3 | +0.82 [−2.14, +3.70] |
| Perception & Counting | CountQA | 11.1 | 8.1 | +2.92 [+1.32, +4.56] | 11.4 | 7.8 | +3.58 [+1.71, +5.51] |
| Perception & Counting | TreeBench | 6.5 | 4.3 | +2.22 [−0.08, +4.54] | 11.9 | 8.4 | +3.46 [−0.24, +7.13] |
| Puzzles & Logic | PuzzleVQA | 20.7 | 15.3 | +5.45 [+3.61, +7.18] | 20.0 | 15.3 | +4.65 [+2.40, +6.90] |
| Puzzles & Logic | VisualPuzzles | 17.6 | 14.7 | +2.94 [+0.87, +4.94] | 19.6 | 13.7 | +5.88 [+3.45, +8.25] |
| Puzzles & Logic | LogicVista | 18.0 | 13.3 | +4.70 [+1.36, +8.08] | 18.2 | 13.2 | +5.00 [+1.00, +9.24] |
| Puzzles & Logic | MME-Reasoning | 13.3 | 10.5 | +2.86 [+1.01, +4.72] | 15.7 | 10.3 | +5.44 [+3.17, +7.69] |

### 5.8 Persistence of the parents' unique successes by category (`stage_b/persistence.csv`; all ordered seed pairs)

| category (binary-eligible benchmarks) | T-only: persists / null (%) | T-only: excess (pp) [95% CI] | V-only: persists / null (%) | V-only: excess (pp) [95% CI] |
|---|---|---|---|---|
| Charts & Tables (partial: 2 of 4 benchmarks) | 55.9 / 29.4 | +26.5 [+23.0, +29.7] | 52.3 / 29.4 | +22.9 [+19.0, +26.5] |
| Visual Math (partial: 3 of 4 benchmarks) | 34.3 / 23.6 | +10.8 [+8.0, +13.3] | 44.0 / 23.6 | +20.5 [+18.2, +22.7] |
| Science & General | 44.9 / 26.1 | +18.9 [+16.4, +21.3] | 44.9 / 26.1 | +18.8 [+16.4, +21.1] |
| Spatial Reasoning | 57.6 / 29.5 | +28.2 [+26.1, +30.2] | 56.2 / 29.5 | +26.7 [+24.6, +28.8] |
| Perception & Counting | 54.2 / 30.0 | +24.2 [+21.7, +26.6] | 59.6 / 30.0 | +29.6 [+27.3, +31.8] |
| Puzzles & Logic | 39.0 / 24.4 | +14.7 [+12.6, +16.8] | 45.3 / 24.4 | +21.0 [+18.9, +23.0] |

Persistence of corrections and regressions by category: excess over the label-swap null, pp [95% CI] (persists / null in %):

| category (binary-eligible benchmarks) | T corrections | V corrections | T regressions | V regressions |
|---|---|---|---|---|
| Charts & Tables (partial: 2 of 4 benchmarks) | +22.2 [+19.5, +24.7] (46.5 / 24.3) | +26.3 [+23.6, +28.8] (53.7 / 27.4) | +6.3 [+1.9, +10.4] (30.5 / 24.3) | +16.4 [+11.9, +20.4] (43.8 / 27.4) |
| Visual Math (partial: 3 of 4 benchmarks) | +9.8 [+7.9, +11.6] (30.0 / 20.2) | +21.0 [+19.2, +22.8] (44.1 / 23.1) | +1.9 [−0.4, +4.1] (22.1 / 20.2) | +6.7 [+4.1, +9.2] (29.8 / 23.1) |
| Science & General | +14.3 [+12.4, +16.2] (36.1 / 21.8) | +23.4 [+21.6, +25.2] (49.1 / 25.7) | +2.3 [+0.1, +4.4] (24.1 / 21.8) | +12.0 [+9.5, +14.3] (37.7 / 25.7) |
| Spatial Reasoning | +9.5 [+7.5, +11.4] (28.1 / 18.7) | +23.3 [+21.4, +25.2] (49.8 / 26.5) | +6.4 [+4.1, +8.7] (25.1 / 18.7) | +24.3 [+22.2, +26.3] (50.8 / 26.5) |
| Perception & Counting | +17.7 [+15.3, +20.1] (37.1 / 19.4) | +29.7 [+27.7, +31.6] (58.1 / 28.4) | +4.6 [+1.6, +7.6] (24.0 / 19.4) | +19.5 [+16.7, +22.2] (47.9 / 28.4) |
| Puzzles & Logic | +8.0 [+6.5, +9.6] (28.3 / 20.3) | +22.9 [+21.2, +24.5] (45.3 / 22.4) | +0.2 [−1.5, +1.9] (20.5 / 20.3) | +7.6 [+5.6, +9.5] (30.0 / 22.4) |

- TRACE's regressions persist beyond the null on 6 of 21 benchmarks (interval above 0); the largest point estimates are on MathVista (+12.8, about 67 rows per seed pair), EmbSpatial (+12.1, about 88 rows per seed pair), CountBenchQA (+11.9, about 11 rows per seed pair).
- VERO's regressions persist beyond the null on 20 of 21 benchmarks (interval above 0); the largest point estimates are on CountBenchQA (+38.3, about 35 rows per seed pair), CV-Bench 3D (+31.0, about 87 rows per seed pair), EmbSpatial (+29.9, about 276 rows per seed pair).

### 5.9 Row-level behaviour by category (`stage_b/parent_overlap.csv`, `stage_b/corrections_regressions.csv`; mean over seeds, rows pooled)

Parent overlap (% of rows):

| category (binary-eligible benchmarks) | rows per seed | T-only [95% CI] | V-only [95% CI] | both correct | both wrong | agreement [95% CI] | hindsight either-correct [95% CI] |
|---|---|---|---|---|---|---|---|
| Charts & Tables (partial: 2 of 4 benchmarks) | 2,250 | 10.5 [9.5, 11.6] | 9.2 [8.2, 10.2] | 46.5 | 33.8 | 80.2 [78.9, 81.5] | 66.2 [64.4, 68.1] |
| Visual Math (partial: 3 of 4 benchmarks) | 4,828 | 9.4 [8.8, 10.0] | 11.1 [10.4, 11.7] | 30.9 | 48.7 | 79.6 [78.7, 80.4] | 51.3 [50.1, 52.6] |
| Science & General | 4,995 | 10.6 [9.9, 11.2] | 10.5 [9.9, 11.2] | 43.0 | 35.9 | 78.9 [78.0, 79.8] | 64.1 [62.9, 65.3] |
| Spatial Reasoning | 6,420 | 9.8 [9.2, 10.4] | 9.6 [9.0, 10.2] | 54.7 | 26.0 | 80.6 [79.8, 81.4] | 74.0 [73.0, 75.1] |
| Perception & Counting | 4,321 | 10.9 [10.1, 11.6] | 11.4 [10.6, 12.3] | 35.3 | 42.4 | 77.7 [76.6, 78.8] | 57.6 [55.8, 59.4] |
| Puzzles & Logic | 4,803 | 13.3 [12.5, 14.0] | 14.3 [13.5, 15.1] | 27.2 | 45.2 | 72.4 [71.4, 73.4] | 54.8 [53.3, 56.2] |
| **all 21** | 27,617 | 10.7 [10.4, 11.0] | 11.1 [10.8, 11.4] | 39.9 | 38.3 | 78.2 [77.8, 78.6] | 61.7 [61.1, 62.3] |

Corrections and regressions against B (% of rows; conditional rates in % of B's errors or successes):

| category | T corrections | T regressions | T net [95% CI] | T corrections / B errors | T regressions / B successes | V@P0 corrections | V@P0 regressions | V@P0 net [95% CI] | V@P0 corrections / B errors | V@P0 regressions / B successes |
|---|---|---|---|---|---|---|---|---|---|---|
| Charts & Tables (partial: 2 of 4 benchmarks) | 14.9 | 7.3 | +7.64 [+6.23, +9.07] | 29.5 | 14.7 | 14.6 | 8.3 | +6.31 [+4.78, +7.85] | 28.8 | 16.7 |
| Visual Math (partial: 3 of 4 benchmarks) | 12.6 | 9.2 | +3.32 [+2.47, +4.16] | 19.9 | 25.0 | 14.1 | 9.1 | +5.01 [+4.03, +6.00] | 22.4 | 24.6 |
| Science & General | 13.3 | 8.9 | +4.41 [+3.51, +5.31] | 26.1 | 18.0 | 15.0 | 10.7 | +4.35 [+3.27, +5.40] | 29.6 | 21.7 |
| Spatial Reasoning | 8.4 | 6.4 | +1.99 [+1.37, +2.61] | 22.4 | 10.3 | 11.8 | 10.0 | +1.73 [+0.77, +2.70] | 31.4 | 16.1 |
| Perception & Counting | 9.1 | 6.2 | +2.98 [+2.12, +3.83] | 16.1 | 14.3 | 13.8 | 10.3 | +3.53 [+2.30, +4.78] | 24.4 | 23.9 |
| Puzzles & Logic | 17.9 | 13.8 | +4.13 [+3.04, +5.16] | 28.1 | 37.9 | 18.7 | 13.5 | +5.18 [+3.87, +6.46] | 29.3 | 37.2 |

## 6. Development screen (seed 42, 2,910 rows; descriptive)

### 6.1 Scores under P1 (`screen/scores.P1.csv`)

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B | 47.97 [46.18, 49.76] | 53.13 | 42.54 | 49.61 | 60.30 | 47.62 | 34.64 |
| T | 51.40 [49.56, 53.24] | 60.09 | 45.90 | 54.88 | 60.48 | 49.11 | 37.97 |
| V@P0 | 53.11 [51.19, 55.08] | 57.61 | 49.06 | 56.45 | 59.41 | 54.60 | 41.55 |
| V@P1 | 48.44 [46.58, 50.39] | 51.92 | 47.50 | 54.69 | 58.30 | 51.03 | 27.23 |
| M050 | 50.38 [48.50, 52.24] | 57.13 | 47.54 | 53.71 | 58.24 | 49.15 | 36.49 |
| CV050 | 49.53 [47.66, 51.42] | 53.38 | 44.96 | 54.10 | 58.52 | 48.51 | 37.69 |
| CT050 | 48.51 [46.64, 50.37] | 56.71 | 42.30 | 52.93 | 58.74 | 46.63 | 33.77 |

### 6.2 Contrasts under P1 (`screen/differences.P1.csv`)

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| M050 − T | −1.03 [−2.57, +0.54] | 0.20 | −2.95 [−6.52, +0.57] | +1.64 [−3.13, +6.35] | −1.17 [−4.38, +1.93] | −2.23 [−6.15, +1.58] | +0.04 [−3.01, +3.34] | −1.48 [−5.62, +2.76] |
| M050 − V@P1 | +1.93 [+0.34, +3.45] | 0.02 | +5.22 [+2.05, +8.43] | +0.04 [−4.53, +4.52] | −0.98 [−4.17, +2.23] | −0.06 [−3.93, +3.87] | −1.88 [−5.72, +1.62] | +9.27 [+4.71, +13.66] |
| M050 − CV050 | +0.85 [−0.67, +2.35] | 0.26 | +3.75 [+1.03, +6.61] | +2.58 [−2.53, +7.73] | −0.39 [−3.40, +2.68] | −0.28 [−3.31, +2.79] | +0.64 [−2.18, +3.57] | −1.20 [−5.83, +3.30] |
| M050 − CT050 | +1.86 [+0.20, +3.54] | 0.03 | +0.43 [−3.34, +4.21] | +5.23 [+0.45, +9.95] | +0.78 [−2.30, +3.88] | −0.50 [−4.67, +3.51] | +2.52 [−0.71, +6.18] | +2.72 [−2.28, +7.81] |
| CV050 − B | +1.55 [−0.15, +3.31] | 0.07 | +0.25 [−3.43, +4.01] | +2.42 [−2.50, +7.35] | +4.49 [+0.69, +8.38] | −1.78 [−5.89, +2.29] | +0.89 [−2.54, +4.52] | +3.05 [−2.06, +8.37] |
| CT050 − B | +0.54 [−1.08, +2.16] | 0.50 | +3.58 [+0.37, +6.78] | −0.23 [−5.09, +4.59] | +3.32 [−0.33, +7.00] | −1.56 [−5.07, +1.98] | −1.00 [−3.74, +1.77] | −0.87 [−6.22, +4.55] |
| T − B | +3.43 [+1.70, +5.18] | < 0.001 | +6.95 [+3.03, +10.90] | +3.36 [−1.37, +8.18] | +5.27 [+1.21, +9.53] | +0.17 [−3.63, +3.97] | +1.48 [−1.83, +4.99] | +3.33 [−1.95, +8.44] |
| V@P1 − B | +0.47 [−1.29, +2.37] | 0.58 | −1.22 [−5.28, +2.97] | +4.96 [+0.24, +9.79] | +5.08 [+1.19, +8.91] | −2.00 [−6.94, +2.73] | +3.41 [−0.91, +8.32] | −7.41 [−12.16, −2.53] |
| V@P1 − V@P0 | −4.67 [−6.09, −3.21] | < 0.001 | −5.69 [−9.23, −2.29] | −1.56 [−6.02, +2.95] | −1.76 [−5.07, +1.55] | −1.11 [−3.76, +1.41] | −3.57 [−6.59, −0.43] | −14.32 [−18.33, −10.18] |

### 6.3 Scores under P2, VERO's interface (`screen/scores.P2.csv`)

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B@P2 | 42.71 [40.94, 44.44] | 39.64 | 33.01 | 46.09 | 57.55 | 47.93 | 32.02 |
| T@P2 | 47.21 [45.44, 49.05] | 43.78 | 39.77 | 52.93 | 59.34 | 49.29 | 38.16 |
| V@P0 | 53.11 [51.19, 55.08] | 57.61 | 49.06 | 56.45 | 59.41 | 54.60 | 41.55 |
| M050@P2 | 52.34 [50.46, 54.29] | 59.77 | 47.89 | 54.30 | 60.83 | 52.10 | 39.14 |
| CV050@P2 | 52.98 [51.05, 54.90] | 60.54 | 47.27 | 58.40 | 58.55 | 50.87 | 42.24 |
| CT050@P2 | 46.80 [44.96, 48.63] | 44.49 | 41.17 | 53.13 | 61.47 | 46.77 | 33.77 |

### 6.4 Contrasts under P2 (`screen/differences.P2.csv`)

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| M050@P2 − V@P0 | −0.77 [−2.40, +0.86] | 0.36 | +2.16 [−2.45, +6.66] | −1.17 [−5.94, +3.41] | −2.15 [−5.52, +1.20] | +1.41 [−2.49, +5.31] | −2.50 [−5.99, +1.12] | −2.41 [−6.39, +1.46] |
| M050@P2 − CV050@P2 | −0.64 [−2.13, +0.87] | 0.43 | −0.77 [−4.80, +3.29] | +0.63 [−3.35, +4.75] | −4.10 [−7.16, −0.96] | +2.28 [−1.13, +5.70] | +1.23 [−1.99, +4.64] | −3.10 [−7.41, +1.20] |
| M050@P2 − CT050@P2 | +5.54 [+3.61, +7.52] | < 0.001 | +15.28 [+10.73, +19.80] | +6.72 [+1.72, +11.95] | +1.17 [−2.68, +5.10] | −0.64 [−5.81, +4.47] | +5.33 [+0.68, +10.28] | +5.37 [+0.35, +10.36] |
| M050@P2 − T@P2 | +5.13 [+3.36, +6.93] | < 0.001 | +16.00 [+11.54, +20.39] | +8.13 [+3.34, +12.94] | +1.37 [−2.35, +5.12] | +1.49 [−2.83, +5.77] | +2.81 [−1.71, +7.61] | +0.98 [−3.34, +5.38] |
| T@P2 − B@P2 | +4.50 [+2.67, +6.42] | < 0.001 | +4.13 [+0.23, +8.07] | +6.76 [+1.37, +12.22] | +6.84 [+2.63, +11.09] | +1.79 [−2.73, +6.28] | +1.37 [−3.35, +6.10] | +6.14 [+1.30, +11.10] |
| V@P0 − B@P2 | +10.41 [+8.40, +12.44] | < 0.001 | +17.96 [+13.26, +22.59] | +16.05 [+10.68, +21.56] | +10.35 [+5.79, +14.91] | +1.86 [−3.47, +7.15] | +6.68 [+2.14, +11.46] | +9.52 [+4.42, +14.86] |

### 6.5 Capture and damage, same-seed parents (`screen/merge_capture_damage.csv`)

| merge | TRACE-only capture (n = 313) | VERO-only capture (n = 256) | shared-success damage (n = 974) | joint-failure rescue (n = 982) |
|---|---|---|---|---|
| M050 | 51.4 [45.7, 57.3] | 52.7 [46.5, 59.0] | 7.3 [5.7, 9.0] | 6.8 [5.3, 8.5] |
| CV050 | 44.4 [38.5, 50.2] | 53.5 [47.4, 59.6] | 10.4 [8.5, 12.4] | 10.6 [8.6, 12.7] |
| CT050 | 56.9 [51.2, 62.6] | 27.0 [21.7, 32.6] | 12.0 [10.0, 14.1] | 11.8 [9.8, 14.0] |

### 6.6 Answer format (`screen/answer_format_sensitivity.csv`; % of rows)

| model | PuzzleVQA accuracy | PuzzleVQA: ends in \( X \) | PuzzleVQA: scored 0, final letter correct | VisualPuzzles accuracy | VisualPuzzles: ends in \( X \) | VisualPuzzles: scored 0, final letter correct |
|---|---|---|---|---|---|---|
| B | 43.8 | 0.0 | 3.9 | 25.8 | 1.6 | 0.0 |
| T | 54.7 | 0.0 | 3.1 | 28.9 | 0.8 | 0.8 |
| V@P0 | 52.3 | 0.0 | 0.0 | 33.6 | 0.0 | 0.0 |
| V@P1 | 15.6 | 30.5 | 37.5 | 18.8 | 21.1 | 9.4 |
| M050 | 46.9 | 1.6 | 7.0 | 24.2 | 7.0 | 3.1 |
| CV050 | 50.8 | 0.0 | 2.3 | 28.1 | 3.1 | 1.6 |
| CT050 | 50.8 | 0.0 | 0.8 | 24.2 | 0.0 | 1.6 |
| B@P2 | 46.9 | 0.0 | 0.0 | 18.0 | 0.0 | 0.0 |
| T@P2 | 52.3 | 0.0 | 0.0 | 25.0 | 0.0 | 0.0 |
| M050@P2 | 59.4 | 0.0 | 0.0 | 25.0 | 0.0 | 0.0 |
| CV050@P2 | 57.0 | 0.0 | 0.0 | 33.6 | 0.0 | 0.0 |
| CT050@P2 | 47.7 | 0.0 | 0.0 | 26.6 | 0.0 | 0.0 |

### 6.7 Per-benchmark scores under P1 (`screen/scores.P1.csv`)

| category | benchmark | B | T | V@P0 | V@P1 | M050 | CV050 | CT050 |
|---|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 44.37 | 49.42 | 42.46 | 40.85 | 48.53 | 46.01 | 44.53 |
| Charts & Tables | CharXivReason | 35.94 | 49.22 | 50.78 | 41.41 | 43.75 | 40.63 | 46.09 |
| Charts & Tables | TableVQABench | 69.73 | 71.39 | 73.91 | 59.00 | 65.94 | 62.04 | 68.24 |
| Charts & Tables | EvoChart | 62.50 | 70.31 | 63.28 | 66.41 | 70.31 | 64.84 | 67.97 |
| Visual Math | MathVision | 18.75 | 27.34 | 21.88 | 27.34 | 26.56 | 25.00 | 28.13 |
| Visual Math | MathVista | 63.28 | 65.63 | 69.53 | 69.53 | 67.19 | 66.41 | 57.03 |
| Visual Math | MathVerse | 46.88 | 46.88 | 52.34 | 53.13 | 53.91 | 48.44 | 45.31 |
| Visual Math | WeMath | 41.25 | 43.75 | 52.50 | 40.00 | 42.50 | 40.00 | 38.75 |
| Science & General | PhyX mini MC | 43.75 | 50.00 | 52.34 | 45.31 | 50.00 | 49.22 | 46.09 |
| Science & General | MMMU-ProVis | 26.56 | 36.72 | 35.94 | 38.28 | 32.81 | 36.72 | 34.38 |
| Science & General | RealWorldQA | 64.84 | 69.53 | 73.44 | 69.53 | 68.75 | 67.97 | 67.19 |
| Science & General | MMStar | 63.28 | 63.28 | 64.06 | 65.63 | 63.28 | 62.50 | 64.06 |
| Spatial Reasoning | EmbSpatial | 76.56 | 75.78 | 79.69 | 79.69 | 79.69 | 79.69 | 75.78 |
| Spatial Reasoning | SpatialVizBench COT | 40.63 | 43.75 | 35.94 | 35.94 | 34.38 | 36.72 | 37.50 |
| Spatial Reasoning | CV-Bench 3D | 78.91 | 79.69 | 78.13 | 77.34 | 75.00 | 75.00 | 76.56 |
| Spatial Reasoning | ERQA | 45.12 | 42.68 | 43.90 | 40.24 | 43.90 | 42.68 | 45.12 |
| Perception & Counting | BLINK | 60.94 | 59.38 | 71.88 | 60.94 | 62.50 | 62.50 | 59.38 |
| Perception & Counting | CountBenchQA | 74.23 | 78.35 | 81.44 | 78.35 | 76.29 | 77.32 | 75.26 |
| Perception & Counting | CountQA | 19.53 | 17.97 | 21.88 | 26.56 | 19.53 | 17.19 | 14.84 |
| Perception & Counting | TreeBench | 35.80 | 40.74 | 43.21 | 38.27 | 38.27 | 37.04 | 37.04 |
| Puzzles & Logic | PuzzleVQA | 43.75 | 54.69 | 52.34 | 15.63 | 46.88 | 50.78 | 50.78 |
| Puzzles & Logic | VisualPuzzles | 25.78 | 28.91 | 33.59 | 18.75 | 24.22 | 28.13 | 24.22 |
| Puzzles & Logic | LogicVista | 39.33 | 47.19 | 50.56 | 47.19 | 48.31 | 46.07 | 38.20 |
| Puzzles & Logic | MME-Reasoning | 29.69 | 21.09 | 29.69 | 27.34 | 26.56 | 25.78 | 21.88 |

### 6.8 Per-benchmark scores under P2 (`screen/scores.P2.csv`)

| category | benchmark | B@P2 | T@P2 | V@P0 | M050@P2 | CV050@P2 | CT050@P2 |
|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 43.87 | 44.78 | 42.46 | 48.15 | 52.63 | 49.01 |
| Charts & Tables | CharXivReason | 32.03 | 45.31 | 50.78 | 50.00 | 45.31 | 48.44 |
| Charts & Tables | TableVQABench | 17.83 | 19.39 | 73.91 | 68.28 | 73.90 | 19.57 |
| Charts & Tables | EvoChart | 64.84 | 65.63 | 63.28 | 72.66 | 70.31 | 60.94 |
| Visual Math | MathVision | 17.19 | 17.19 | 21.88 | 26.56 | 27.34 | 23.44 |
| Visual Math | MathVista | 59.38 | 61.72 | 69.53 | 66.41 | 64.84 | 56.25 |
| Visual Math | MathVerse | 30.47 | 41.41 | 52.34 | 46.09 | 53.13 | 43.75 |
| Visual Math | WeMath | 25.00 | 38.75 | 52.50 | 52.50 | 43.75 | 41.25 |
| Science & General | PhyX mini MC | 34.38 | 46.09 | 52.34 | 43.75 | 57.03 | 44.53 |
| Science & General | MMMU-ProVis | 30.47 | 33.59 | 35.94 | 36.72 | 38.28 | 34.38 |
| Science & General | RealWorldQA | 60.16 | 71.88 | 73.44 | 75.00 | 75.78 | 68.75 |
| Science & General | MMStar | 59.38 | 60.16 | 64.06 | 61.72 | 62.50 | 64.84 |
| Spatial Reasoning | EmbSpatial | 53.91 | 72.66 | 79.69 | 79.69 | 76.56 | 73.44 |
| Spatial Reasoning | SpatialVizBench COT | 28.91 | 31.25 | 35.94 | 33.59 | 35.94 | 30.47 |
| Spatial Reasoning | CV-Bench 3D | 74.22 | 77.34 | 78.13 | 81.25 | 76.56 | 77.34 |
| Spatial Reasoning | ERQA | 73.17 | 56.10 | 43.90 | 48.78 | 45.12 | 64.63 |
| Perception & Counting | BLINK | 58.59 | 60.16 | 71.88 | 64.06 | 60.94 | 57.03 |
| Perception & Counting | CountBenchQA | 76.29 | 76.29 | 81.44 | 79.38 | 77.32 | 76.29 |
| Perception & Counting | CountQA | 14.84 | 18.75 | 21.88 | 24.22 | 19.53 | 17.97 |
| Perception & Counting | TreeBench | 41.98 | 41.98 | 43.21 | 40.74 | 45.68 | 35.80 |
| Puzzles & Logic | PuzzleVQA | 46.88 | 52.34 | 52.34 | 59.38 | 57.03 | 47.66 |
| Puzzles & Logic | VisualPuzzles | 17.97 | 25.00 | 33.59 | 25.00 | 33.59 | 26.56 |
| Puzzles & Logic | LogicVista | 46.07 | 47.19 | 50.56 | 47.19 | 49.44 | 38.20 |
| Puzzles & Logic | MME-Reasoning | 17.19 | 28.13 | 29.69 | 25.00 | 28.91 | 22.66 |

### 6.9 Per-benchmark merge contrasts under P1 (`screen/differences.P1.csv`; one seed, wide intervals)

| category | benchmark | M050 − T | M050 − V@P1 | M050 − CV050 | M050 − CT050 | V@P1 − V@P0 |
|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | −0.89 [−6.94, +5.35] | +7.68 [+0.53, +14.61] | +2.52 [−4.22, +9.02] | +4.00 [−2.20, +10.36] | −1.61 [−7.15, +3.54] |
| Charts & Tables | CharXivReason | −5.47 [−13.79, +2.61] | +2.34 [−4.24, +9.02] | +3.13 [−3.03, +9.30] | −2.34 [−11.03, +6.35] | −9.38 [−15.83, −3.23] |
| Charts & Tables | TableVQABench | −5.45 [−12.77, +1.68] | +6.94 [−0.04, +14.13] | +3.90 [+0.42, +8.03] | −2.29 [−10.14, +6.19] | −14.91 [−24.00, −6.23] |
| Charts & Tables | EvoChart | +0.00 [−6.84, +6.87] | +3.91 [−0.67, +8.57] | +5.47 [+0.00, +11.02] | +2.34 [−4.35, +9.02] | +3.13 [−3.85, +9.92] |
| Visual Math | MathVision | −0.78 [−8.82, +7.19] | −0.78 [−9.02, +7.14] | +1.56 [−6.90, +9.92] | −1.56 [−8.15, +4.97] | +5.47 [+0.00, +11.64] |
| Visual Math | MathVista | +1.56 [−5.71, +8.70] | −2.34 [−9.16, +4.35] | +0.78 [−5.59, +7.14] | +10.16 [+3.60, +17.39] | +0.00 [−6.56, +6.62] |
| Visual Math | MathVerse | +7.03 [−0.89, +15.25] | +0.78 [−6.48, +8.21] | +5.47 [−2.24, +13.16] | +8.59 [+0.00, +17.36] | +0.78 [−6.47, +8.15] |
| Visual Math | WeMath | −1.25 [−15.00, +11.84] | +2.50 [−10.23, +15.12] | +2.50 [−13.33, +18.42] | +3.75 [−10.29, +17.65] | −12.50 [−26.25, +1.25] |
| Science & General | PhyX mini MC | +0.00 [−9.40, +9.45] | +4.69 [−3.45, +12.61] | +0.78 [−7.87, +9.43] | +3.91 [−5.50, +13.49] | −7.03 [−15.17, +0.85] |
| Science & General | MMMU-ProVis | −3.91 [−11.02, +3.01] | −5.47 [−12.40, +1.52] | −3.91 [−10.81, +3.00] | −1.56 [−8.06, +5.00] | +2.34 [−4.39, +8.96] |
| Science & General | RealWorldQA | −0.78 [−4.20, +2.52] | −0.78 [−4.17, +2.52] | +0.78 [−3.67, +5.43] | +1.56 [−2.02, +5.47] | −3.91 [−9.09, +0.85] |
| Science & General | MMStar | +0.00 [−3.17, +3.17] | −2.34 [−8.40, +3.50] | +0.78 [−1.71, +3.54] | −0.78 [−3.54, +1.71] | +1.56 [−4.55, +7.76] |
| Spatial Reasoning | EmbSpatial | +3.91 [−0.84, +8.97] | +0.00 [−3.57, +4.00] | +0.00 [+0.00, +0.00] | +3.91 [−1.53, +9.17] | +0.00 [−3.20, +3.13] |
| Spatial Reasoning | SpatialVizBench COT | −9.38 [−18.64, +0.00] | −1.56 [−10.74, +7.14] | −2.34 [−11.89, +7.19] | −3.13 [−14.07, +7.90] | +0.00 [−6.61, +6.45] |
| Spatial Reasoning | CV-Bench 3D | −4.69 [−12.31, +2.63] | −2.34 [−10.67, +5.97] | +0.00 [−7.34, +7.56] | −1.56 [−9.23, +5.83] | −0.78 [−5.11, +2.99] |
| Spatial Reasoning | ERQA | +1.22 [−7.53, +9.59] | +3.66 [−4.88, +12.82] | +1.22 [+0.00, +4.12] | −1.22 [−9.46, +6.52] | −3.66 [−10.26, +2.41] |
| Perception & Counting | BLINK | +3.13 [−3.97, +10.40] | +1.56 [−3.88, +6.84] | +0.00 [−5.69, +5.79] | +3.13 [−4.03, +10.29] | −10.94 [−17.89, −4.03] |
| Perception & Counting | CountBenchQA | −2.06 [−8.51, +4.30] | −2.06 [−8.74, +4.30] | −1.03 [−4.67, +2.25] | +1.03 [−5.81, +7.92] | −3.09 [−8.60, +2.15] |
| Perception & Counting | CountQA | +1.56 [−3.42, +8.51] | −7.03 [−17.78, +1.47] | +2.34 [−5.41, +11.11] | +4.69 [−1.68, +13.89] | +4.69 [+1.28, +10.11] |
| Perception & Counting | TreeBench | −2.47 [−8.64, +3.33] | +0.00 [−6.90, +7.04] | +1.23 [−2.70, +5.80] | +1.23 [−4.17, +7.06] | −4.94 [−12.64, +2.60] |
| Puzzles & Logic | PuzzleVQA | −7.81 [−16.15, +0.00] | +31.25 [+22.43, +39.85] | −3.91 [−12.50, +4.41] | −3.91 [−14.68, +6.62] | −36.72 [−45.22, −28.12] |
| Puzzles & Logic | VisualPuzzles | −4.69 [−12.95, +3.23] | +5.47 [−3.28, +14.20] | −3.91 [−12.98, +4.97] | +0.00 [−9.70, +9.52] | −14.84 [−24.19, −5.33] |
| Puzzles & Logic | LogicVista | +1.12 [−8.75, +12.50] | +1.12 [−9.20, +10.89] | +2.25 [−7.46, +12.33] | +10.11 [−1.18, +22.08] | −3.37 [−11.59, +4.65] |
| Puzzles & Logic | MME-Reasoning | +5.47 [+0.00, +11.51] | −0.78 [−8.20, +6.56] | +0.78 [−8.53, +9.63] | +4.69 [−3.10, +12.00] | −2.34 [−8.80, +4.17] |

### 6.10 Per-benchmark merge contrasts under P2 (`screen/differences.P2.csv`; one seed, wide intervals)

| category | benchmark | M050@P2 − V@P0 | M050@P2 − CV050@P2 | M050@P2 − CT050@P2 | M050@P2 − T@P2 |
|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | +5.69 [−2.79, +13.82] | −4.48 [−11.26, +1.74] | −0.85 [−7.53, +5.74] | +3.38 [−2.97, +9.88] |
| Charts & Tables | CharXivReason | −0.78 [−9.79, +8.05] | +4.69 [−4.55, +13.99] | +1.56 [−9.17, +12.12] | +4.69 [−4.55, +13.67] |
| Charts & Tables | TableVQABench | −5.64 [−16.78, +5.70] | −5.62 [−14.91, +3.32] | +48.71 [+37.72, +59.55] | +48.89 [+37.96, +59.54] |
| Charts & Tables | EvoChart | +9.38 [+2.56, +16.81] | +2.34 [−4.00, +9.48] | +11.72 [+4.55, +19.08] | +7.03 [−0.98, +15.28] |
| Visual Math | MathVision | +4.69 [−2.67, +12.20] | −0.78 [−8.00, +6.54] | +3.13 [−6.19, +12.12] | +9.38 [+1.35, +17.52] |
| Visual Math | MathVista | −3.13 [−10.34, +4.10] | +1.56 [−4.51, +7.76] | +10.16 [+2.84, +17.86] | +4.69 [−3.29, +12.84] |
| Visual Math | MathVerse | −6.25 [−13.76, +0.88] | −7.03 [−14.69, +0.72] | +2.34 [−5.74, +10.32] | +4.69 [−3.43, +12.84] |
| Visual Math | WeMath | +0.00 [−13.64, +13.64] | +8.75 [−1.52, +20.00] | +11.25 [−3.66, +26.67] | +13.75 [+0.00, +27.18] |
| Science & General | PhyX mini MC | −8.59 [−16.81, −0.69] | −13.28 [−21.31, −5.38] | −0.78 [−9.48, +7.52] | −2.34 [−10.43, +5.51] |
| Science & General | MMMU-ProVis | +0.78 [−5.83, +7.50] | −1.56 [−8.49, +5.15] | +2.34 [−5.67, +10.24] | +3.13 [−4.84, +10.95] |
| Science & General | RealWorldQA | +1.56 [−4.10, +7.32] | −0.78 [−4.92, +3.25] | +6.25 [+0.00, +13.33] | +3.13 [−3.36, +9.85] |
| Science & General | MMStar | −2.34 [−8.33, +3.45] | −0.78 [−5.97, +4.26] | −3.13 [−10.57, +4.17] | +1.56 [−5.41, +8.28] |
| Spatial Reasoning | EmbSpatial | +0.00 [−4.10, +4.67] | +3.13 [−0.93, +7.69] | +6.25 [+0.83, +11.68] | +7.03 [+0.79, +13.41] |
| Spatial Reasoning | SpatialVizBench COT | −2.34 [−11.71, +6.92] | −2.34 [−13.01, +8.04] | +3.13 [−7.76, +14.29] | +2.34 [−6.94, +11.54] |
| Spatial Reasoning | CV-Bench 3D | +3.13 [−3.31, +9.65] | +4.69 [−1.72, +11.11] | +3.91 [−3.88, +11.59] | +3.91 [−3.88, +11.68] |
| Spatial Reasoning | ERQA | +4.88 [−4.82, +14.52] | +3.66 [+0.00, +8.45] | −15.85 [−30.23, −2.10] | −7.32 [−18.28, +3.13] |
| Perception & Counting | BLINK | −7.81 [−14.06, −1.71] | +3.13 [−3.03, +9.38] | +7.03 [−0.83, +15.38] | +3.91 [−4.41, +12.14] |
| Perception & Counting | CountBenchQA | −2.06 [−8.70, +4.26] | +2.06 [−2.83, +7.22] | +3.09 [−7.41, +13.79] | +3.09 [−5.68, +11.90] |
| Perception & Counting | CountQA | +2.34 [−3.65, +9.40] | +4.69 [+1.36, +9.68] | +6.25 [−1.46, +16.28] | +5.47 [−2.31, +15.66] |
| Perception & Counting | TreeBench | −2.47 [−11.84, +6.67] | −4.94 [−14.71, +4.65] | +4.94 [−5.19, +15.07] | −1.23 [−11.84, +9.21] |
| Puzzles & Logic | PuzzleVQA | +7.03 [+0.00, +14.53] | +2.34 [−7.26, +11.94] | +11.72 [+2.82, +21.05] | +7.03 [−0.99, +15.04] |
| Puzzles & Logic | VisualPuzzles | −8.59 [−16.80, −0.72] | −8.59 [−17.65, +0.76] | −1.56 [−11.56, +8.33] | +0.00 [−9.63, +9.45] |
| Puzzles & Logic | LogicVista | −3.37 [−12.37, +5.00] | −2.25 [−10.84, +5.66] | +8.99 [−4.62, +21.79] | +0.00 [−10.00, +10.13] |
| Puzzles & Logic | MME-Reasoning | −4.69 [−12.28, +2.10] | −3.91 [−11.28, +3.62] | +2.34 [−4.81, +9.68] | −3.13 [−10.62, +4.76] |

### 6.11 Length and format on the development screen (`length_format_judge_audit.csv`; seed 42)

The screen runs under P1 and P2, with archived V@P0 seed 42 (full split, not the screen rows) for reference. Rates are row-weighted. These are observable correlates of the P2 results in §6, not causes.

| model | responses | length stops | mean completion tokens | median of per-benchmark median completion tokens | with </think> (%) | with <answer> (%) | with \boxed{} (%) |
|---|---|---|---|---|---|---|---|
| V@P1 | 2,910 | 1 | 206.7 | 136 | 4.4 | 4.4 | 41.7 |
| M050 | 2,910 | 0 | 223.8 | 162 | 4.4 | 4.4 | 31.6 |
| CV050 | 2,910 | 3 | 205.9 | 124 | 4.3 | 4.4 | 26.5 |
| CT050 | 2,910 | 1 | 208.9 | 141 | 1.7 | 4.4 | 22.9 |
| B@P2 | 2,910 | 1 | 247.0 | 184 | 95.1 | 81.0 | 72.9 |
| T@P2 | 2,910 | 0 | 279.1 | 231 | 98.5 | 85.7 | 85.5 |
| M050@P2 | 2,910 | 0 | 272.8 | 205 | 99.2 | 98.8 | 66.8 |
| CV050@P2 | 2,910 | 0 | 248.2 | 179 | 99.1 | 98.9 | 65.9 |
| CT050@P2 | 2,910 | 0 | 261.8 | 213 | 97.9 | 83.2 | 80.3 |
| V@P0 (archive, full split) | 32,805 | 4 | 296.6 | 229 | 100.0 | 100.0 | 67.8 |

Median completion tokens per benchmark under P2:

| category | benchmark | B@P2 | T@P2 | M050@P2 | CV050@P2 | CT050@P2 | V@P0 (archive) |
|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 148 | 260 | 162 | 115 | 213 | 25 |
| Charts & Tables | CharXivReason | 181 | 263 | 271 | 201 | 233 | 193 |
| Charts & Tables | TableVQABench | 116 | 213 | 56 | 23 | 147 | 21 |
| Charts & Tables | EvoChart | 132 | 164 | 164 | 149 | 147 | 244 |
| Visual Math | MathVision | 305 | 345 | 449 | 461 | 352 | 641 |
| Visual Math | MathVista | 187 | 232 | 242 | 224 | 214 | 308 |
| Visual Math | MathVerse | 329 | 330 | 380 | 364 | 329 | 542 |
| Visual Math | WeMath | 221 | 255 | 282 | 268 | 254 | 370 |
| Science & General | PhyX mini MC | 530 | 499 | 535 | 532 | 517 | 544 |
| Science & General | MMMU-ProVis | 384 | 418 | 378 | 349 | 412 | 411 |
| Science & General | RealWorldQA | 108 | 149 | 83 | 83 | 107 | 119 |
| Science & General | MMStar | 167 | 216 | 140 | 126 | 189 | 207 |
| Spatial Reasoning | EmbSpatial | 107 | 175 | 73 | 60 | 152 | 15 |
| Spatial Reasoning | SpatialVizBench COT | 308 | 336 | 326 | 313 | 316 | 383 |
| Spatial Reasoning | CV-Bench 3D | 120 | 184 | 176 | 158 | 142 | 213 |
| Spatial Reasoning | ERQA | 88 | 130 | 15 | 15 | 90 | 15 |
| Perception & Counting | BLINK | 189 | 223 | 166 | 138 | 191 | 196 |
| Perception & Counting | CountBenchQA | 63 | 90 | 15 | 15 | 80 | 15 |
| Perception & Counting | CountQA | 106 | 137 | 147 | 125 | 120 | 183 |
| Perception & Counting | TreeBench | 126 | 190 | 85 | 63 | 138 | 15 |
| Puzzles & Logic | PuzzleVQA | 213 | 230 | 234 | 206 | 225 | 249 |
| Puzzles & Logic | VisualPuzzles | 292 | 288 | 298 | 277 | 293 | 339 |
| Puzzles & Logic | LogicVista | 278 | 340 | 378 | 358 | 316 | 397 |
| Puzzles & Logic | MME-Reasoning | 336 | 353 | 410 | 415 | 342 | 416 |

Responses with an `<answer>` tag under P2 (%):

| category | benchmark | B@P2 | T@P2 | M050@P2 | CV050@P2 | CT050@P2 | V@P0 (archive) |
|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 93.8 | 89.1 | 99.2 | 99.2 | 93.8 | 100.0 |
| Charts & Tables | CharXivReason | 87.5 | 88.3 | 100.0 | 100.0 | 79.7 | 100.0 |
| Charts & Tables | TableVQABench | 72.7 | 99.2 | 100.0 | 100.0 | 87.5 | 100.0 |
| Charts & Tables | EvoChart | 85.2 | 97.7 | 100.0 | 100.0 | 89.8 | 100.0 |
| Visual Math | MathVision | 50.0 | 66.4 | 78.9 | 78.9 | 60.2 | 99.9 |
| Visual Math | MathVista | 84.4 | 96.1 | 100.0 | 100.0 | 93.0 | 100.0 |
| Visual Math | MathVerse | 83.6 | 80.5 | 100.0 | 100.0 | 86.7 | 100.0 |
| Visual Math | WeMath | 75.2 | 95.3 | 100.0 | 100.0 | 75.2 | 100.0 |
| Science & General | PhyX mini MC | 76.6 | 70.3 | 100.0 | 100.0 | 69.5 | 100.0 |
| Science & General | MMMU-ProVis | 57.8 | 61.7 | 98.4 | 98.4 | 64.8 | 100.0 |
| Science & General | RealWorldQA | 95.3 | 92.2 | 100.0 | 100.0 | 99.2 | 100.0 |
| Science & General | MMStar | 83.6 | 94.5 | 100.0 | 100.0 | 89.1 | 100.0 |
| Spatial Reasoning | EmbSpatial | 76.6 | 86.7 | 100.0 | 100.0 | 89.1 | 100.0 |
| Spatial Reasoning | SpatialVizBench COT | 85.2 | 93.8 | 100.0 | 100.0 | 79.7 | 100.0 |
| Spatial Reasoning | CV-Bench 3D | 95.3 | 96.9 | 100.0 | 100.0 | 92.2 | 100.0 |
| Spatial Reasoning | ERQA | 98.8 | 96.3 | 100.0 | 100.0 | 98.8 | 100.0 |
| Perception & Counting | BLINK | 89.1 | 90.6 | 100.0 | 100.0 | 98.4 | 100.0 |
| Perception & Counting | CountBenchQA | 95.9 | 99.0 | 100.0 | 100.0 | 96.9 | 100.0 |
| Perception & Counting | CountQA | 93.8 | 97.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| Perception & Counting | TreeBench | 93.8 | 85.2 | 100.0 | 100.0 | 88.9 | 100.0 |
| Puzzles & Logic | PuzzleVQA | 76.6 | 68.8 | 99.2 | 99.2 | 67.2 | 100.0 |
| Puzzles & Logic | VisualPuzzles | 68.8 | 79.7 | 100.0 | 100.0 | 71.9 | 100.0 |
| Puzzles & Logic | LogicVista | 59.6 | 49.4 | 100.0 | 100.0 | 51.7 | 99.8 |
| Puzzles & Logic | MME-Reasoning | 74.2 | 78.1 | 97.7 | 100.0 | 75.8 | 100.0 |

Responses containing `\boxed{}` under P2 (%):

| category | benchmark | B@P2 | T@P2 | M050@P2 | CV050@P2 | CT050@P2 | V@P0 (archive) |
|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 39.1 | 60.2 | 1.6 | 0.8 | 46.9 | 1.0 |
| Charts & Tables | CharXivReason | 95.3 | 100.0 | 100.0 | 96.9 | 100.0 | 96.4 |
| Charts & Tables | TableVQABench | 85.2 | 97.7 | 25.8 | 18.8 | 93.8 | 13.1 |
| Charts & Tables | EvoChart | 100.0 | 100.0 | 100.0 | 100.0 | 99.2 | 100.0 |
| Visual Math | MathVision | 68.0 | 81.3 | 82.8 | 85.9 | 77.3 | 99.9 |
| Visual Math | MathVista | 99.2 | 99.2 | 100.0 | 100.0 | 99.2 | 100.0 |
| Visual Math | MathVerse | 99.2 | 98.4 | 100.0 | 100.0 | 100.0 | 100.0 |
| Visual Math | WeMath | 96.9 | 100.0 | 89.9 | 89.1 | 100.0 | 87.6 |
| Science & General | PhyX mini MC | 93.8 | 100.0 | 99.2 | 98.4 | 100.0 | 95.6 |
| Science & General | MMMU-ProVis | 42.2 | 61.7 | 18.8 | 18.0 | 57.0 | 77.2 |
| Science & General | RealWorldQA | 93.0 | 96.1 | 99.2 | 96.9 | 93.8 | 98.7 |
| Science & General | MMStar | 97.7 | 100.0 | 100.0 | 100.0 | 98.4 | 100.0 |
| Spatial Reasoning | EmbSpatial | 17.2 | 73.4 | 0.0 | 0.0 | 42.2 | 0.0 |
| Spatial Reasoning | SpatialVizBench COT | 85.2 | 100.0 | 100.0 | 100.0 | 96.1 | 100.0 |
| Spatial Reasoning | CV-Bench 3D | 98.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Spatial Reasoning | ERQA | 13.4 | 45.1 | 0.0 | 0.0 | 19.5 | 0.0 |
| Perception & Counting | BLINK | 93.0 | 94.5 | 96.9 | 96.1 | 96.1 | 99.3 |
| Perception & Counting | CountBenchQA | 32.0 | 99.0 | 0.0 | 0.0 | 55.7 | 0.0 |
| Perception & Counting | CountQA | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Perception & Counting | TreeBench | 37.0 | 80.2 | 16.0 | 16.0 | 48.1 | 16.8 |
| Puzzles & Logic | PuzzleVQA | 15.6 | 35.2 | 5.5 | 1.6 | 41.4 | 32.9 |
| Puzzles & Logic | VisualPuzzles | 21.1 | 22.7 | 21.1 | 18.0 | 36.7 | 65.3 |
| Puzzles & Logic | LogicVista | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.8 |
| Puzzles & Logic | MME-Reasoning | 90.6 | 98.4 | 98.4 | 97.7 | 91.4 | 98.7 |

## 7. Confirmation (P1, 26,228 rows, seeds 42–44)

The confirmed set, the primary family and every input hash were locked at 2026-11-02T09:04:27+00:00 (`selection_lock.json`) before any confirmation run had scores.

### 7.1 Official scores (`confirmation/scores.csv`)

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B | 48.10 [47.53, 48.68] | 54.27 | 42.92 | 51.30 | 54.94 | 48.79 | 36.38 |
| T | 52.13 [51.51, 52.74] | 59.38 | 48.69 | 55.78 | 56.89 | 51.60 | 40.41 |
| V@P0 | 52.23 [51.56, 52.90] | 57.36 | 50.41 | 55.70 | 57.59 | 51.34 | 41.00 |
| V@P1 | 49.29 [48.64, 49.94] | 53.48 | 48.64 | 55.89 | 57.44 | 50.50 | 29.80 |
| M050 | 52.36 [51.73, 52.97] | 58.47 | 49.08 | 56.94 | 58.02 | 51.93 | 39.69 |
| CV050 | 51.06 [50.45, 51.68] | 56.89 | 47.55 | 54.60 | 56.78 | 51.38 | 39.17 |
| CT050 | 50.48 [49.88, 51.07] | 57.32 | 45.35 | 54.75 | 56.01 | 50.63 | 38.82 |

### 7.2 Macro per seed (`confirmation/seed_scores.csv`; unweighted mean of the 24 benchmarks)

| model | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|
| B | 48.17 | 48.31 | 47.83 | 48.10 |
| T | 52.33 | 51.89 | 52.16 | 52.13 |
| V@P0 | 52.23 | 52.17 | 52.30 | 52.23 |
| V@P1 | 49.22 | 49.33 | 49.32 | 49.29 |
| M050 | 52.48 | 52.32 | 52.27 | 52.36 |
| CV050 | 50.83 | 51.16 | 51.19 | 51.06 |
| CT050 | 50.54 | 50.44 | 50.45 | 50.48 |

### 7.3 Per-benchmark scores, mean over seeds (`confirmation/scores.csv`)

| category | benchmark | B | T | V@P0 | V@P1 | M050 | CV050 | CT050 |
|---|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 45.70 | 48.27 | 40.91 | 41.34 | 49.99 | 47.43 | 48.15 |
| Charts & Tables | CharXivReason | 39.50 | 46.33 | 45.83 | 45.79 | 47.54 | 45.00 | 43.75 |
| Charts & Tables | TableVQABench | 74.99 | 78.37 | 78.67 | 62.70 | 70.74 | 71.29 | 75.75 |
| Charts & Tables | EvoChart | 56.89 | 64.56 | 64.03 | 64.06 | 65.63 | 63.83 | 61.63 |
| Visual Math | MathVision | 24.23 | 26.67 | 27.66 | 26.64 | 28.17 | 27.70 | 26.58 |
| Visual Math | MathVista | 69.33 | 74.50 | 77.96 | 76.21 | 77.46 | 75.58 | 71.54 |
| Visual Math | MathVerse | 42.80 | 47.25 | 49.58 | 47.35 | 44.44 | 43.12 | 42.01 |
| Visual Math | WeMath | 35.33 | 46.33 | 46.45 | 44.34 | 46.25 | 43.78 | 41.27 |
| Science & General | PhyX mini MC | 40.42 | 48.71 | 45.96 | 45.75 | 48.50 | 44.96 | 47.88 |
| Science & General | MMMU-ProVis | 35.89 | 39.31 | 39.74 | 40.58 | 41.04 | 39.09 | 37.98 |
| Science & General | RealWorldQA | 66.12 | 68.95 | 70.21 | 70.26 | 70.10 | 67.97 | 67.48 |
| Science & General | MMStar | 62.78 | 66.17 | 66.89 | 66.97 | 68.11 | 66.39 | 65.67 |
| Spatial Reasoning | EmbSpatial | 69.15 | 70.62 | 70.42 | 71.14 | 71.84 | 71.36 | 69.97 |
| Spatial Reasoning | SpatialVizBench COT | 35.06 | 35.31 | 31.92 | 33.05 | 37.50 | 34.75 | 36.16 |
| Spatial Reasoning | CV-Bench 3D | 76.56 | 80.97 | 83.37 | 84.48 | 82.29 | 80.76 | 77.95 |
| Spatial Reasoning | ERQA | 38.99 | 40.67 | 44.65 | 41.09 | 40.46 | 40.25 | 39.94 |
| Perception & Counting | BLINK | 52.95 | 56.39 | 56.89 | 56.30 | 58.01 | 55.64 | 54.70 |
| Perception & Counting | CountBenchQA | 83.68 | 86.41 | 83.59 | 83.93 | 87.35 | 87.52 | 85.47 |
| Perception & Counting | CountQA | 20.28 | 23.15 | 23.91 | 24.11 | 24.19 | 24.08 | 22.93 |
| Perception & Counting | TreeBench | 38.27 | 40.43 | 40.95 | 37.65 | 38.17 | 38.27 | 39.40 |
| Puzzles & Logic | PuzzleVQA | 43.91 | 49.74 | 48.76 | 15.65 | 45.12 | 45.85 | 47.46 |
| Puzzles & Logic | VisualPuzzles | 32.12 | 34.66 | 37.69 | 24.40 | 33.23 | 31.87 | 32.90 |
| Puzzles & Logic | LogicVista | 43.58 | 47.95 | 46.46 | 48.98 | 50.65 | 49.72 | 47.58 |
| Puzzles & Logic | MME-Reasoning | 25.93 | 29.31 | 31.08 | 30.16 | 29.77 | 29.24 | 27.34 |

### 7.4 Per-benchmark scores per seed (`confirmation/seed_scores.csv`)

Archived models on the confirmation partition:

| benchmark | B s42 | B s43 | B s44 | T s42 | T s43 | T s44 | V@P0 s42 | V@P0 s43 | V@P0 s44 |
|---|---|---|---|---|---|---|---|---|---|
| ChartQAPro | 45.98 | 45.45 | 45.67 | 47.82 | 48.07 | 48.93 | 41.42 | 40.62 | 40.70 |
| CharXivReason | 39.63 | 40.50 | 38.38 | 47.25 | 44.88 | 46.88 | 46.38 | 45.50 | 45.63 |
| TableVQABench | 73.77 | 75.73 | 75.48 | 78.63 | 78.33 | 78.14 | 78.82 | 79.35 | 77.84 |
| EvoChart | 57.56 | 56.16 | 56.96 | 64.56 | 64.46 | 64.66 | 64.06 | 64.26 | 63.76 |
| MathVision | 25.08 | 24.05 | 23.56 | 26.11 | 26.81 | 27.10 | 27.55 | 27.80 | 27.63 |
| MathVista | 69.38 | 69.63 | 69.00 | 74.25 | 74.75 | 74.50 | 77.63 | 77.63 | 78.63 |
| MathVerse | 41.27 | 43.49 | 43.65 | 47.30 | 47.62 | 46.83 | 48.57 | 49.21 | 50.95 |
| WeMath | 37.80 | 35.77 | 32.42 | 48.44 | 44.98 | 45.57 | 46.77 | 46.05 | 46.53 |
| PhyX mini MC | 37.00 | 44.38 | 39.88 | 49.38 | 47.63 | 49.13 | 45.63 | 45.63 | 46.63 |
| MMMU-ProVis | 35.84 | 35.19 | 36.63 | 38.66 | 40.10 | 39.16 | 39.88 | 39.67 | 39.67 |
| RealWorldQA | 66.99 | 66.83 | 64.54 | 68.46 | 68.46 | 69.93 | 70.26 | 70.92 | 69.44 |
| MMStar | 63.58 | 62.92 | 61.83 | 66.42 | 66.08 | 66.00 | 66.33 | 67.25 | 67.08 |
| EmbSpatial | 69.99 | 69.13 | 68.34 | 70.78 | 71.12 | 69.95 | 70.60 | 70.36 | 70.30 |
| SpatialVizBench COT | 34.11 | 35.91 | 35.17 | 34.75 | 35.59 | 35.59 | 32.20 | 31.46 | 32.10 |
| CV-Bench 3D | 76.46 | 74.58 | 78.65 | 81.04 | 80.21 | 81.67 | 84.17 | 83.13 | 82.81 |
| ERQA | 39.62 | 38.05 | 39.31 | 41.19 | 42.45 | 38.36 | 44.34 | 44.34 | 45.28 |
| BLINK | 52.14 | 54.44 | 52.27 | 57.07 | 56.61 | 55.49 | 56.61 | 56.15 | 57.92 |
| CountBenchQA | 86.15 | 82.05 | 82.82 | 88.46 | 85.13 | 85.64 | 83.59 | 83.33 | 83.85 |
| CountQA | 18.67 | 20.39 | 21.79 | 22.44 | 23.59 | 23.42 | 23.51 | 24.16 | 24.08 |
| TreeBench | 37.04 | 37.96 | 39.81 | 39.81 | 38.58 | 42.90 | 41.36 | 41.36 | 40.12 |
| PuzzleVQA | 43.95 | 43.13 | 44.64 | 50.47 | 49.03 | 49.72 | 48.59 | 48.65 | 49.03 |
| VisualPuzzles | 31.62 | 32.15 | 32.58 | 35.26 | 34.51 | 34.19 | 37.41 | 37.62 | 38.05 |
| LogicVista | 46.37 | 44.41 | 39.94 | 47.21 | 47.21 | 49.44 | 46.09 | 46.93 | 46.37 |
| MME-Reasoning | 26.03 | 27.09 | 24.66 | 30.16 | 29.10 | 28.68 | 31.64 | 30.79 | 30.79 |
| **macro** | 48.17 | 48.31 | 47.83 | 52.33 | 51.89 | 52.16 | 52.23 | 52.17 | 52.30 |

New runs:

| benchmark | V@P1 s42 | V@P1 s43 | V@P1 s44 | M050 s42 | M050 s43 | M050 s44 | CV050 s42 | CV050 s43 | CV050 s44 | CT050 s42 | CT050 s43 | CT050 s44 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ChartQAPro | 41.18 | 42.07 | 40.78 | 50.27 | 49.26 | 50.44 | 47.36 | 47.43 | 47.51 | 47.51 | 48.54 | 48.41 |
| CharXivReason | 45.75 | 45.88 | 45.75 | 47.38 | 46.38 | 48.88 | 44.13 | 46.63 | 44.25 | 43.00 | 44.50 | 43.75 |
| TableVQABench | 62.48 | 61.02 | 64.61 | 71.06 | 68.46 | 72.69 | 69.12 | 71.41 | 73.34 | 74.51 | 76.77 | 75.97 |
| EvoChart | 64.16 | 64.26 | 63.76 | 65.87 | 65.57 | 65.47 | 64.26 | 63.66 | 63.56 | 61.16 | 62.36 | 61.36 |
| MathVision | 26.19 | 27.26 | 26.48 | 28.78 | 27.80 | 27.92 | 27.67 | 27.14 | 28.29 | 26.69 | 26.73 | 26.32 |
| MathVista | 76.00 | 76.25 | 76.38 | 78.00 | 77.75 | 76.63 | 75.50 | 74.25 | 77.00 | 71.88 | 72.13 | 70.63 |
| MathVerse | 43.33 | 48.10 | 50.63 | 43.65 | 47.30 | 42.38 | 40.16 | 42.06 | 47.14 | 41.27 | 40.95 | 43.81 |
| WeMath | 45.22 | 42.70 | 45.10 | 47.25 | 45.93 | 45.57 | 45.10 | 43.06 | 43.18 | 40.67 | 42.22 | 40.91 |
| PhyX mini MC | 46.63 | 45.00 | 45.63 | 47.75 | 50.00 | 47.75 | 45.50 | 46.13 | 43.25 | 48.25 | 47.75 | 47.63 |
| MMMU-ProVis | 40.46 | 41.11 | 40.17 | 40.82 | 41.33 | 40.97 | 39.38 | 39.52 | 38.37 | 37.57 | 37.07 | 39.31 |
| RealWorldQA | 70.10 | 71.08 | 69.61 | 69.12 | 71.24 | 69.93 | 67.48 | 68.79 | 67.65 | 67.65 | 67.16 | 67.65 |
| MMStar | 66.00 | 67.58 | 67.33 | 67.67 | 68.50 | 68.17 | 66.75 | 66.33 | 66.08 | 65.67 | 66.25 | 65.08 |
| EmbSpatial | 71.26 | 71.15 | 71.02 | 71.91 | 72.22 | 71.39 | 71.70 | 71.39 | 70.98 | 70.47 | 70.02 | 69.44 |
| SpatialVizBench COT | 33.58 | 32.63 | 32.94 | 38.77 | 36.65 | 37.08 | 33.90 | 35.38 | 34.96 | 37.18 | 35.91 | 35.38 |
| CV-Bench 3D | 84.90 | 84.90 | 83.65 | 82.19 | 81.98 | 82.71 | 80.21 | 80.73 | 81.35 | 78.65 | 76.35 | 78.85 |
| ERQA | 41.51 | 40.88 | 40.88 | 41.19 | 40.57 | 39.62 | 40.57 | 40.25 | 39.94 | 40.57 | 40.57 | 38.68 |
| BLINK | 56.15 | 57.07 | 55.69 | 58.38 | 58.45 | 57.20 | 55.23 | 56.94 | 54.77 | 54.77 | 55.42 | 53.91 |
| CountBenchQA | 84.36 | 83.33 | 84.10 | 88.21 | 86.41 | 87.44 | 88.72 | 86.15 | 87.69 | 88.46 | 84.10 | 83.85 |
| CountQA | 24.24 | 23.51 | 24.57 | 24.16 | 23.91 | 24.49 | 24.24 | 23.91 | 24.08 | 22.44 | 22.93 | 23.42 |
| TreeBench | 38.27 | 38.58 | 36.11 | 37.04 | 38.27 | 39.20 | 38.27 | 37.96 | 38.58 | 38.89 | 38.27 | 41.05 |
| PuzzleVQA | 15.36 | 17.05 | 14.55 | 45.52 | 44.08 | 45.77 | 46.96 | 45.33 | 45.27 | 46.90 | 46.65 | 48.84 |
| VisualPuzzles | 24.87 | 22.72 | 25.62 | 33.55 | 31.51 | 34.62 | 31.40 | 32.05 | 32.15 | 33.87 | 31.19 | 33.65 |
| LogicVista | 49.44 | 49.44 | 48.04 | 50.84 | 51.68 | 49.44 | 47.21 | 51.68 | 50.28 | 47.77 | 49.16 | 45.81 |
| MME-Reasoning | 29.84 | 30.26 | 30.37 | 30.26 | 30.37 | 28.68 | 29.21 | 29.74 | 28.78 | 27.30 | 27.62 | 27.09 |
| **macro** | 49.22 | 49.33 | 49.32 | 52.48 | 52.32 | 52.27 | 50.83 | 51.16 | 51.19 | 50.54 | 50.44 | 50.45 |

### 7.5 Primary family (Holm-adjusted) and secondary contrasts (`confirmation/differences.csv`)

| comparison | macro [95% CI] | p | Holm p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|---|
| M050 − T | +0.23 [−0.19, +0.65] | 0.29 | 0.29 | −0.91 [−1.76, −0.07] | +0.39 [−0.70, +1.48] | +1.15 [+0.24, +2.04] | +1.13 [+0.14, +2.15] | +0.33 [−0.75, +1.38] | −0.72 [−1.97, +0.48] |
| M050 − V@P1 | +3.07 [+2.62, +3.51] | < 0.001 | < 0.001 | +5.00 [+4.02, +5.95] | +0.44 [−0.59, +1.47] | +1.05 [+0.12, +1.97] | +0.58 [−0.54, +1.65] | +1.43 [+0.26, +2.62] | +9.89 [+8.59, +11.19] |
| M050 − CV050 | +1.29 [+0.93, +1.64] | < 0.001 | < 0.001 | +1.59 [+0.83, +2.34] | +1.53 [+0.65, +2.45] | +2.33 [+1.52, +3.15] | +1.24 [+0.46, +2.05] | +0.55 [−0.21, +1.30] | +0.52 [−0.58, +1.59] |
| M050 − CT050 | +1.88 [+1.45, +2.31] | < 0.001 | < 0.001 | +1.15 [+0.26, +2.04] | +3.73 [+2.69, +4.78] | +2.19 [+1.25, +3.14] | +2.02 [+1.01, +3.08] | +1.30 [+0.29, +2.32] | +0.87 [−0.38, +2.11] |

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| T − B | +4.02 [+3.58, +4.45] | < 0.001 | +5.11 [+4.17, +6.05] | +5.76 [+4.66, +6.91] | +4.48 [+3.44, +5.50] | +1.95 [+0.93, +2.94] | +2.80 [+1.83, +3.81] | +4.03 [+2.75, +5.32] |
| V@P1 − B | +1.19 [+0.65, +1.72] | < 0.001 | −0.80 [−1.92, +0.32] | +5.71 [+4.49, +6.97] | +4.59 [+3.32, +5.89] | +2.50 [+1.11, +3.87] | +1.70 [+0.31, +3.09] | −6.58 [−8.05, −5.13] |
| V@P1 − V@P0 | −2.94 [−3.33, −2.56] | < 0.001 | −3.89 [−4.73, −3.04] | −1.77 [−2.68, −0.90] | +0.19 [−0.63, +1.02] | −0.15 [−1.13, +0.79] | −0.84 [−1.87, +0.19] | −11.20 [−12.38, −10.03] |
| M050 − B | +4.25 [+3.78, +4.72] | < 0.001 | +4.20 [+3.20, +5.23] | +6.16 [+5.01, +7.30] | +5.64 [+4.50, +6.78] | +3.08 [+1.98, +4.18] | +3.13 [+2.03, +4.26] | +3.31 [+1.92, +4.69] |
| CV050 − B | +2.96 [+2.52, +3.38] | < 0.001 | +2.62 [+1.77, +3.49] | +4.62 [+3.56, +5.68] | +3.30 [+2.28, +4.30] | +1.84 [+0.82, +2.82] | +2.58 [+1.56, +3.65] | +2.79 [+1.57, +4.06] |
| CT050 − B | +2.38 [+2.00, +2.75] | < 0.001 | +3.05 [+2.32, +3.79] | +2.42 [+1.41, +3.45] | +3.45 [+2.52, +4.38] | +1.06 [+0.19, +1.91] | +1.83 [+1.13, +2.54] | +2.44 [+1.26, +3.60] |

### 7.6 Per-benchmark contrasts (`confirmation/differences.csv`; exploratory, unadjusted)

| category | benchmark | M050 − T | M050 − V@P1 | M050 − CV050 | M050 − CT050 | V@P1 − V@P0 |
|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | +1.71 [+0.31, +3.11] | +8.64 [+6.46, +10.88] | +2.55 [+1.11, +4.03] | +1.83 [+0.42, +3.21] | +0.43 [−0.84, +1.71] |
| Charts & Tables | CharXivReason | +1.21 [−0.64, +3.07] | +1.75 [−0.25, +3.79] | +2.54 [+0.70, +4.44] | +3.79 [+1.73, +5.88] | −0.04 [−1.89, +1.79] |
| Charts & Tables | TableVQABench | −7.63 [−9.52, −5.66] | +8.03 [+6.14, +9.88] | −0.55 [−1.75, +0.68] | −5.02 [−6.86, −3.10] | −15.97 [−18.07, −13.89] |
| Charts & Tables | EvoChart | +1.07 [−0.52, +2.67] | +1.57 [−0.13, +3.25] | +1.80 [+0.38, +3.25] | +4.00 [+2.25, +5.77] | +0.03 [−1.34, +1.38] |
| Visual Math | MathVision | +1.49 [+0.38, +2.63] | +1.52 [+0.38, +2.69] | +0.47 [−0.57, +1.51] | +1.59 [+0.45, +2.72] | −1.01 [−2.00, −0.05] |
| Visual Math | MathVista | +2.96 [+1.23, +4.73] | +1.25 [−0.30, +2.92] | +1.88 [+0.27, +3.47] | +5.92 [+4.09, +7.75] | −1.75 [−3.10, −0.46] |
| Visual Math | MathVerse | −2.80 [−5.36, −0.35] | −2.91 [−5.25, −0.54] | +1.32 [−0.79, +3.51] | +2.43 [−0.11, +4.95] | −2.22 [−4.27, −0.16] |
| Visual Math | WeMath | −0.08 [−2.99, +2.78] | +1.91 [−0.87, +4.69] | +2.47 [+0.24, +4.69] | +4.98 [+2.33, +7.58] | −2.11 [−4.52, +0.32] |
| Science & General | PhyX mini MC | −0.21 [−2.46, +2.03] | +2.75 [+0.66, +4.91] | +3.54 [+1.24, +5.81] | +0.63 [−1.70, +2.93] | −0.21 [−1.98, +1.55] |
| Science & General | MMMU-ProVis | +1.73 [+0.20, +3.25] | +0.46 [−1.06, +2.01] | +1.95 [+0.56, +3.32] | +3.06 [+1.51, +4.61] | +0.84 [−0.40, +2.11] |
| Science & General | RealWorldQA | +1.14 [−0.82, +3.06] | −0.16 [−2.23, +1.95] | +2.12 [+0.54, +3.72] | +2.61 [+0.40, +4.77] | +0.05 [−2.04, +2.16] |
| Science & General | MMStar | +1.94 [+0.59, +3.34] | +1.14 [−0.35, +2.58] | +1.72 [+0.62, +2.85] | +2.44 [+1.07, +3.79] | +0.08 [−1.26, +1.41] |
| Spatial Reasoning | EmbSpatial | +1.22 [+0.46, +1.99] | +0.70 [−0.02, +1.44] | +0.48 [+0.09, +0.88] | +1.87 [+1.09, +2.65] | +0.72 [−0.01, +1.46] |
| Spatial Reasoning | SpatialVizBench COT | +2.19 [−0.27, +4.60] | +4.45 [+2.05, +6.90] | +2.75 [+0.54, +5.07] | +1.34 [−1.19, +3.83] | +1.13 [−0.46, +2.74] |
| Spatial Reasoning | CV-Bench 3D | +1.32 [−0.38, +3.01] | −2.19 [−3.97, −0.48] | +1.53 [+0.18, +2.89] | +4.34 [+2.58, +6.16] | +1.11 [−0.13, +2.37] |
| Spatial Reasoning | ERQA | −0.21 [−2.80, +2.37] | −0.63 [−3.79, +2.42] | +0.21 [−1.51, +1.91] | +0.52 [−2.08, +3.15] | −3.56 [−6.86, −0.49] |
| Perception & Counting | BLINK | +1.62 [+0.13, +3.13] | +1.71 [+0.02, +3.40] | +2.37 [+1.12, +3.61] | +3.31 [+1.63, +4.99] | −0.59 [−2.12, +0.92] |
| Perception & Counting | CountBenchQA | +0.94 [−1.31, +3.23] | +3.42 [+0.97, +5.94] | −0.17 [−1.65, +1.28] | +1.88 [−0.17, +3.96] | +0.34 [−1.46, +2.11] |
| Perception & Counting | CountQA | +1.04 [−0.46, +2.56] | +0.08 [−1.79, +1.95] | +0.11 [−1.55, +1.74] | +1.26 [−0.44, +2.93] | +0.19 [−1.19, +1.54] |
| Perception & Counting | TreeBench | −2.26 [−5.29, +0.64] | +0.51 [−2.64, +3.69] | −0.10 [−1.79, +1.52] | −1.23 [−3.84, +1.38] | −3.29 [−6.54, −0.11] |
| Puzzles & Logic | PuzzleVQA | −4.62 [−6.57, −2.70] | +29.47 [+27.25, +31.75] | −0.73 [−2.36, +0.88] | −2.34 [−4.44, −0.21] | −33.10 [−35.63, −30.61] |
| Puzzles & Logic | VisualPuzzles | −1.43 [−3.69, +0.84] | +8.82 [+6.41, +11.27] | +1.36 [−0.75, +3.41] | +0.32 [−1.98, +2.59] | −13.29 [−15.74, −10.84] |
| Puzzles & Logic | LogicVista | +2.70 [−0.71, +5.97] | +1.68 [−1.79, +5.09] | +0.93 [−2.03, +3.82] | +3.07 [−0.27, +6.51] | +2.51 [−0.27, +5.26] |
| Puzzles & Logic | MME-Reasoning | +0.46 [−1.44, +2.34] | −0.39 [−2.39, +1.68] | +0.53 [−1.26, +2.32] | +2.43 [+0.54, +4.33] | −0.92 [−2.41, +0.48] |

| category | benchmark | T − B | V@P1 − B | M050 − B | CV050 − B | CT050 − B | V@P0 − B (Stage B tables) |
|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | +2.57 [+0.92, +4.22] | −4.36 [−6.48, −2.32] | +4.29 [+2.60, +5.92] | +1.73 [+0.33, +3.12] | +2.45 [+1.05, +3.90] | −4.79 [−7.03, −2.64] |
| Charts & Tables | CharXivReason | +6.83 [+4.54, +9.21] | +6.29 [+3.72, +8.92] | +8.04 [+5.66, +10.49] | +5.50 [+3.50, +7.47] | +4.25 [+2.62, +5.95] | +6.33 [+3.61, +9.10] |
| Charts & Tables | TableVQABench | +3.37 [+2.00, +4.78] | −12.29 [−14.55, −10.01] | −4.26 [−6.13, −2.36] | −3.70 [−5.33, −2.04] | +0.76 [−0.06, +1.61] | +3.68 [+1.96, +5.39] |
| Charts & Tables | EvoChart | +7.67 [+5.62, +9.76] | +7.17 [+5.06, +9.30] | +8.74 [+6.67, +10.86] | +6.94 [+5.09, +8.84] | +4.74 [+2.98, +6.51] | +7.14 [+4.98, +9.32] |
| Visual Math | MathVision | +2.44 [+1.27, +3.62] | +2.41 [+1.08, +3.75] | +3.93 [+2.72, +5.18] | +3.47 [+2.29, +4.64] | +2.34 [+1.23, +3.48] | +3.43 [+2.04, +4.84] |
| Visual Math | MathVista | +5.17 [+3.17, +7.21] | +6.88 [+4.59, +9.12] | +8.13 [+6.14, +10.18] | +6.25 [+4.42, +8.05] | +2.21 [+0.44, +3.96] | +8.63 [+6.37, +10.91] |
| Visual Math | MathVerse | +4.44 [+2.03, +6.93] | +4.55 [+1.74, +7.47] | +1.64 [−0.93, +4.18] | +0.32 [−2.23, +2.92] | −0.79 [−3.17, +1.61] | +6.77 [+4.03, +9.59] |
| Visual Math | WeMath | +11.00 [+7.98, +14.04] | +9.01 [+5.99, +12.04] | +10.93 [+8.04, +13.83] | +8.45 [+5.86, +11.05] | +5.94 [+3.43, +8.50] | +11.12 [+7.97, +14.38] |
| Science & General | PhyX mini MC | +8.29 [+5.35, +11.22] | +5.33 [+2.07, +8.69] | +8.08 [+4.89, +11.34] | +4.54 [+1.71, +7.39] | +7.46 [+4.49, +10.36] | +5.54 [+2.24, +8.87] |
| Science & General | MMMU-ProVis | +3.42 [+1.78, +5.07] | +4.70 [+2.91, +6.54] | +5.15 [+3.53, +6.83] | +3.20 [+1.64, +4.85] | +2.10 [+0.57, +3.59] | +3.85 [+2.01, +5.71] |
| Science & General | RealWorldQA | +2.83 [+0.97, +4.75] | +4.14 [+1.36, +6.89] | +3.98 [+1.55, +6.38] | +1.85 [−0.06, +3.73] | +1.36 [+0.11, +2.63] | +4.08 [+1.08, +7.01] |
| Science & General | MMStar | +3.39 [+1.91, +4.93] | +4.19 [+2.26, +6.11] | +5.33 [+3.73, +6.97] | +3.61 [+2.17, +5.05] | +2.89 [+1.73, +4.08] | +4.11 [+2.11, +6.11] |
| Spatial Reasoning | EmbSpatial | +1.47 [+0.85, +2.10] | +1.99 [+0.92, +3.07] | +2.69 [+1.86, +3.53] | +2.21 [+1.46, +2.97] | +0.82 [+0.41, +1.24] | +1.27 [+0.01, +2.50] |
| Spatial Reasoning | SpatialVizBench COT | +0.25 [−2.27, +2.77] | −2.01 [−5.00, +1.06] | +2.44 [−0.11, +5.16] | −0.32 [−2.75, +2.13] | +1.09 [−1.24, +3.41] | −3.14 [−6.16, −0.10] |
| Spatial Reasoning | CV-Bench 3D | +4.41 [+2.56, +6.24] | +7.92 [+5.74, +10.09] | +5.73 [+3.80, +7.62] | +4.20 [+2.34, +6.09] | +1.39 [−0.33, +3.08] | +6.81 [+4.50, +9.13] |
| Spatial Reasoning | ERQA | +1.68 [−0.95, +4.25] | +2.10 [−1.61, +5.93] | +1.47 [−1.42, +4.31] | +1.26 [−1.20, +3.66] | +0.94 [−0.96, +2.82] | +5.66 [+0.75, +10.68] |
| Perception & Counting | BLINK | +3.44 [+1.99, +4.85] | +3.35 [+1.16, +5.50] | +5.06 [+3.17, +6.90] | +2.70 [+1.15, +4.24] | +1.75 [+0.69, +2.79] | +3.94 [+1.66, +6.26] |
| Perception & Counting | CountBenchQA | +2.74 [+0.97, +4.58] | +0.26 [−2.73, +3.27] | +3.68 [+1.47, +5.96] | +3.85 [+1.76, +6.01] | +1.79 [+0.55, +3.08] | −0.09 [−3.23, +3.04] |
| Perception & Counting | CountQA | +2.87 [+1.08, +4.66] | +3.82 [+1.82, +5.79] | +3.90 [+2.06, +5.75] | +3.79 [+2.06, +5.62] | +2.65 [+1.15, +4.15] | +3.63 [+1.46, +5.78] |
| Perception & Counting | TreeBench | +2.16 [−0.38, +4.81] | −0.62 [−4.40, +3.04] | −0.10 [−2.97, +2.80] | +0.00 [−2.68, +2.77] | +1.13 [−0.66, +2.93] | +2.67 [−1.58, +6.88] |
| Puzzles & Logic | PuzzleVQA | +5.83 [+3.81, +7.77] | −28.25 [−30.50, −25.97] | +1.21 [−1.24, +3.67] | +1.94 [−0.30, +4.20] | +3.55 [+1.55, +5.52] | +4.85 [+2.33, +7.42] |
| Puzzles & Logic | VisualPuzzles | +2.54 [+0.18, +4.83] | −7.72 [−10.34, −5.06] | +1.11 [−1.24, +3.48] | −0.25 [−2.40, +1.94] | +0.79 [−1.40, +2.95] | +5.57 [+2.86, +8.29] |
| Puzzles & Logic | LogicVista | +4.38 [+0.80, +8.06] | +5.40 [+1.46, +9.43] | +7.08 [+3.30, +10.91] | +6.15 [+2.79, +9.54] | +4.00 [+0.87, +7.23] | +2.89 [−1.45, +7.32] |
| Puzzles & Logic | MME-Reasoning | +3.39 [+1.33, +5.48] | +4.23 [+1.69, +6.70] | +3.84 [+1.80, +5.90] | +3.32 [+1.30, +5.24] | +1.41 [−0.42, +3.20] | +5.15 [+2.61, +7.66] |

The V@P0 − B column comes from the Stage B tables on the same partition and seeds (`stage_b/differences.confirmation.csv`), whose bootstrap uses the same resampling units and draws: its T − B column equals the one above exactly.

### 7.7 Capture, retention and damage (`confirmation/merge_capture_damage.base_reference.csv`)

Setup:
- 21 binary-eligible benchmarks, 22,078 rows per seed.
- Groups are defined by T and V@P1 on the confirmation partition.
- Rates are pooled over conditions, and group sizes are mean rows per condition.
- Intervals are for each rate on its own, not for differences.
- The B (reference) rows score the base model as if it were a merge. They were added post hoc.

- In the stable-other-seeds variant, rows whose T/V@P1 pattern differs between the two other seeds form a fifth group; its column is the merge's accuracy on those rows, and the five groups partition the rows.

**all 21 benchmarks**

| model (variant, group sizes) | TRACE-only capture | VERO-only capture | shared-success damage | joint-failure rescue | unstable parent pattern: merge accuracy |
|---|---|---|---|---|---|
| **independent repeat** (2,732 / 2,141 / 8,446 / 8,759) |  |  |  |  |  |
| M050 | 49.0 [47.7, 50.4] | 52.8 [51.4, 54.1] | 7.0 [6.6, 7.3] | 11.4 [10.9, 11.8] | – |
| CV050 | 46.1 [44.8, 47.3] | 49.7 [48.3, 51.1] | 9.6 [9.1, 10.0] | 12.5 [12.1, 13.0] | – |
| CT050 | 54.6 [53.4, 55.8] | 36.4 [35.2, 37.6] | 12.8 [12.3, 13.3] | 14.2 [13.7, 14.6] | – |
| B (reference, post hoc) | 47.6 [46.4, 48.8] | 36.7 [35.4, 37.9] | 17.7 [17.1, 18.3] | 15.4 [14.9, 15.9] | – |
| **contemporaneous** (2,732 / 2,141 / 8,446 / 8,759) |  |  |  |  |  |
| M050 | 50.6 [49.3, 52.0] | 51.5 [50.0, 52.9] | 5.8 [5.5, 6.2] | 10.1 [9.7, 10.6] | – |
| CV050 | 47.5 [46.2, 48.8] | 47.5 [46.0, 48.9] | 8.3 [7.8, 8.7] | 11.4 [11.0, 11.9] | – |
| CT050 | 58.5 [57.2, 59.7] | 30.3 [29.0, 31.6] | 10.7 [10.2, 11.1] | 12.4 [12.0, 12.9] | – |
| B (reference, post hoc) | 50.2 [49.0, 51.5] | 31.5 [30.2, 32.8] | 15.9 [15.3, 16.4] | 14.1 [13.6, 14.6] | – |
| **stable other seeds** (1,316 / 978 / 7,304 / 7,229 / 5,251) |  |  |  |  |  |
| M050 | 55.5 [53.5, 57.6] | 50.9 [48.6, 53.1] | 3.6 [3.4, 3.9] | 7.0 [6.6, 7.4] | 48.5 [47.6, 49.5] |
| CV050 | 52.1 [50.1, 54.1] | 47.5 [45.3, 49.8] | 5.7 [5.3, 6.1] | 8.6 [8.2, 9.0] | 45.7 [44.7, 46.6] |
| CT050 | 63.9 [62.1, 65.6] | 29.9 [28.0, 31.8] | 8.4 [8.0, 8.8] | 10.0 [9.6, 10.5] | 44.3 [43.4, 45.3] |
| B (reference, post hoc) | 54.9 [53.1, 56.8] | 32.8 [30.7, 34.7] | 13.0 [12.4, 13.6] | 12.0 [11.5, 12.5] | 40.4 [39.5, 41.4] |

**without PuzzleVQA and VisualPuzzles**

| model (variant, group sizes) | TRACE-only capture | VERO-only capture | shared-success damage | joint-failure rescue | unstable parent pattern: merge accuracy |
|---|---|---|---|---|---|
| **independent repeat** (1,922 / 1,971 / 8,139 / 7,518) |  |  |  |  |  |
| M050 | 45.3 [43.9, 46.7] | 53.0 [51.5, 54.4] | 6.6 [6.3, 6.9] | 10.3 [9.9, 10.8] | – |
| CV050 | 42.3 [40.9, 43.7] | 50.1 [48.7, 51.4] | 9.1 [8.7, 9.6] | 11.1 [10.6, 11.6] | – |
| CT050 | 52.5 [51.1, 53.9] | 36.8 [35.5, 38.1] | 12.2 [11.7, 12.7] | 12.4 [12.0, 12.9] | – |
| B (reference, post hoc) | 45.4 [44.0, 46.7] | 36.6 [35.3, 37.8] | 16.9 [16.3, 17.5] | 13.7 [13.2, 14.2] | – |
| **contemporaneous** (1,922 / 1,971 / 8,139 / 7,518) |  |  |  |  |  |
| M050 | 46.9 [45.4, 48.5] | 51.5 [50.0, 53.0] | 5.4 [5.1, 5.7] | 9.0 [8.6, 9.4] | – |
| CV050 | 44.2 [42.7, 45.7] | 47.7 [46.2, 49.2] | 7.7 [7.3, 8.1] | 9.7 [9.3, 10.2] | – |
| CT050 | 57.8 [56.3, 59.3] | 30.0 [28.7, 31.4] | 10.1 [9.6, 10.5] | 10.5 [10.1, 11.0] | – |
| B (reference, post hoc) | 49.2 [47.7, 50.7] | 31.1 [29.8, 32.5] | 15.1 [14.5, 15.6] | 12.1 [11.6, 12.6] | – |
| **stable other seeds** (863 / 915 / 7,138 / 6,318 / 4,316) |  |  |  |  |  |
| M050 | 49.4 [47.0, 51.9] | 50.9 [48.5, 53.2] | 3.5 [3.2, 3.8] | 6.4 [6.0, 6.8] | 48.8 [47.7, 49.9] |
| CV050 | 46.2 [43.8, 48.6] | 47.8 [45.5, 50.1] | 5.5 [5.1, 5.9] | 7.5 [7.0, 7.9] | 45.8 [44.7, 46.9] |
| CT050 | 62.0 [59.8, 64.2] | 30.3 [28.4, 32.3] | 8.1 [7.7, 8.5] | 8.7 [8.3, 9.2] | 43.8 [42.8, 44.9] |
| B (reference, post hoc) | 52.2 [49.9, 54.5] | 32.5 [30.5, 34.6] | 12.6 [12.0, 13.2] | 10.6 [10.1, 11.1] | 40.0 [39.0, 41.1] |

**Where pooled binary accuracy comes from** (independent repeat). Each group's contribution is its share of rows times the model's rate of being correct on it, in percentage points of all rows; before rounding, the four contributions add up to the model's pooled accuracy on these benchmarks, which pools rows and is not the 24-benchmark macro.

*all 21 benchmarks* (group shares: 12.37% / 9.70% / 38.25% / 39.67%):

| model | TRACE-only kept | VERO-only kept | shared successes kept | joint failures rescued | pooled accuracy [95% CI] |
|---|---|---|---|---|---|
| M050 | 6.07 | 5.12 | 35.58 | 4.51 | 51.28 [50.61, 51.95] |
| CV050 | 5.70 | 4.82 | 34.59 | 4.98 | 50.09 [49.44, 50.77] |
| CT050 | 6.76 | 3.53 | 33.37 | 5.62 | 49.27 [48.63, 49.92] |
| B (reference, post hoc) | 5.89 | 3.56 | 31.49 | 6.10 | 47.03 [46.41, 47.67] |
| T (same rows) |  |  |  |  | 50.63 |
| V@P1 (same rows) |  |  |  |  | 47.95 |

Differences, arithmetic on the rows above (no paired interval):

| comparison | TRACE-only kept | VERO-only kept | shared successes kept | joint failures rescued | pooled accuracy |
|---|---|---|---|---|---|
| M050 − CV050 | +0.37 | +0.30 | +0.99 | −0.47 | +1.19 |
| M050 − CT050 | −0.69 | +1.59 | +2.21 | −1.10 | +2.01 |
| M050 − B (reference, post hoc) | +0.18 | +1.56 | +4.09 | −1.58 | +4.25 |

*without PuzzleVQA and VisualPuzzles* (group shares: 9.83% / 10.08% / 41.63% / 38.46%):

| model | TRACE-only kept | VERO-only kept | shared successes kept | joint failures rescued | pooled accuracy [95% CI] |
|---|---|---|---|---|---|
| M050 | 4.45 | 5.34 | 38.88 | 3.97 | 52.65 [51.94, 53.34] |
| CV050 | 4.16 | 5.05 | 37.83 | 4.27 | 51.31 [50.59, 51.99] |
| CT050 | 5.17 | 3.71 | 36.55 | 4.78 | 50.20 [49.52, 50.89] |
| B (reference, post hoc) | 4.46 | 3.69 | 34.58 | 5.27 | 48.00 [47.33, 48.67] |
| T (same rows) |  |  |  |  | 51.46 |
| V@P1 (same rows) |  |  |  |  | 51.71 |

Differences, arithmetic on the rows above (no paired interval):

| comparison | TRACE-only kept | VERO-only kept | shared successes kept | joint failures rescued | pooled accuracy |
|---|---|---|---|---|---|
| M050 − CV050 | +0.29 | +0.29 | +1.05 | −0.30 | +1.34 |
| M050 − CT050 | −0.71 | +1.63 | +2.34 | −0.82 | +2.44 |
| M050 − B (reference, post hoc) | −0.01 | +1.65 | +4.31 | −1.30 | +4.65 |

### 7.8 Capture, retention and damage by category (`confirmation/merge_capture_damage.base_reference.csv`; independent repeat, descriptive)

Same setup as §7.7, one table per outcome group. A category marked partial has fewer than four binary-eligible benchmarks.

**TRACE-only capture**

| category | mean group size per condition | M050 | CV050 | CT050 | B (reference, post hoc) |
|---|---|---|---|---|---|
| Charts & Tables (partial: 2 of 4 benchmarks) | 164 | 52.7 [47.8, 57.6] | 43.5 [38.3, 48.6] | 53.6 [48.6, 58.3] | 37.7 [32.9, 42.5] |
| Visual Math (partial: 3 of 4 benchmarks) | 388 | 44.5 [41.6, 47.5] | 42.8 [40.0, 45.7] | 49.4 [46.5, 52.2] | 42.9 [40.2, 45.7] |
| Science & General | 407 | 47.0 [43.9, 50.2] | 41.2 [38.1, 44.2] | 53.2 [50.4, 56.1] | 45.1 [42.2, 48.0] |
| Spatial Reasoning | 442 | 43.5 [40.4, 46.6] | 42.4 [39.4, 45.4] | 57.1 [54.4, 59.9] | 52.8 [49.9, 55.6] |
| Perception & Counting | 363 | 46.0 [42.4, 49.6] | 44.1 [40.6, 47.5] | 53.3 [50.1, 56.6] | 47.2 [43.8, 50.6] |
| Puzzles & Logic | 968 | 54.8 [52.3, 57.2] | 52.3 [49.9, 54.5] | 56.8 [54.7, 58.9] | 49.9 [47.9, 51.9] |

**VERO-only capture**

| category | mean group size per condition | M050 | CV050 | CT050 | B (reference, post hoc) |
|---|---|---|---|---|---|
| Charts & Tables (partial: 2 of 4 benchmarks) | 154 | 56.0 [50.7, 61.2] | 58.5 [53.7, 63.3] | 41.1 [36.9, 45.6] | 41.5 [37.0, 45.9] |
| Visual Math (partial: 3 of 4 benchmarks) | 401 | 49.8 [46.7, 52.8] | 48.5 [45.6, 51.5] | 38.1 [35.4, 40.9] | 33.7 [31.1, 36.3] |
| Science & General | 419 | 54.2 [51.2, 57.2] | 48.2 [45.3, 51.2] | 37.1 [34.4, 39.9] | 35.9 [33.2, 38.7] |
| Spatial Reasoning | 471 | 60.5 [57.7, 63.3] | 55.9 [52.9, 58.8] | 39.0 [36.4, 41.8] | 43.4 [40.6, 46.2] |
| Perception & Counting | 355 | 46.7 [43.0, 50.6] | 45.4 [41.9, 49.1] | 31.2 [28.3, 34.3] | 30.8 [27.7, 34.0] |
| Puzzles & Logic | 341 | 48.8 [45.7, 51.8] | 44.8 [41.8, 47.9] | 33.2 [30.4, 36.0] | 35.9 [33.2, 38.6] |

**shared-success damage**

| category | mean group size per condition | M050 | CV050 | CT050 | B (reference, post hoc) |
|---|---|---|---|---|---|
| Charts & Tables (partial: 2 of 4 benchmarks) | 852 | 4.8 [3.9, 5.8] | 8.3 [7.0, 9.7] | 11.9 [10.3, 13.4] | 19.4 [17.4, 21.4] |
| Visual Math (partial: 3 of 4 benchmarks) | 1,155 | 11.0 [9.9, 12.1] | 14.4 [13.1, 15.7] | 18.2 [16.7, 19.6] | 22.4 [20.8, 24.0] |
| Science & General | 1,742 | 6.5 [5.8, 7.3] | 9.4 [8.5, 10.3] | 11.1 [10.1, 12.0] | 18.5 [17.3, 19.9] |
| Spatial Reasoning | 2,855 | 4.7 [4.2, 5.1] | 6.6 [6.0, 7.2] | 9.6 [8.9, 10.3] | 12.1 [11.3, 12.8] |
| Perception & Counting | 1,245 | 6.4 [5.5, 7.4] | 8.1 [7.1, 9.2] | 11.5 [10.3, 12.7] | 15.7 [14.3, 17.2] |
| Puzzles & Logic | 597 | 16.1 [14.3, 18.0] | 19.7 [17.7, 21.9] | 26.4 [24.1, 28.9] | 34.6 [32.3, 37.1] |

**joint-failure rescue**

| category | mean group size per condition | M050 | CV050 | CT050 | B (reference, post hoc) |
|---|---|---|---|---|---|
| Charts & Tables (partial: 2 of 4 benchmarks) | 629 | 8.3 [6.9, 9.8] | 8.7 [7.3, 10.2] | 10.1 [8.7, 11.7] | 11.4 [9.8, 13.1] |
| Visual Math (partial: 3 of 4 benchmarks) | 1,918 | 9.6 [8.8, 10.4] | 10.5 [9.6, 11.3] | 10.1 [9.3, 10.9] | 11.2 [10.3, 12.1] |
| Science & General | 1,427 | 10.8 [9.9, 11.8] | 11.6 [10.6, 12.6] | 13.2 [12.1, 14.3] | 15.7 [14.5, 16.9] |
| Spatial Reasoning | 1,367 | 12.2 [11.1, 13.3] | 14.1 [12.9, 15.4] | 17.4 [16.1, 18.7] | 18.7 [17.4, 20.1] |
| Perception & Counting | 1,493 | 9.7 [8.6, 10.8] | 9.4 [8.4, 10.4] | 11.1 [10.1, 12.2] | 11.6 [10.6, 12.7] |
| Puzzles & Logic | 1,925 | 15.3 [14.2, 16.4] | 17.9 [16.8, 19.1] | 20.3 [19.1, 21.5] | 21.1 [19.9, 22.2] |

**By benchmark** (`confirmation/merge_capture_damage.csv`; independent repeat; the base reference is not computed per benchmark):

*TRACE-only capture*

| category | benchmark | mean group size per condition | M050 | CV050 | CT050 |
|---|---|---|---|---|---|
| Charts & Tables | CharXivReason | 82 | 47.2 [39.8, 54.1] | 38.8 [32.0, 45.7] | 46.3 [39.7, 53.2] |
| Charts & Tables | EvoChart | 82 | 58.4 [51.7, 65.0] | 48.2 [40.3, 55.7] | 60.8 [53.6, 67.3] |
| Visual Math | MathVision | 255 | 40.4 [36.8, 43.9] | 39.7 [36.1, 43.1] | 44.8 [41.2, 48.3] |
| Visual Math | MathVista | 56 | 58.6 [50.9, 66.3] | 53.9 [45.7, 61.7] | 60.4 [52.8, 67.8] |
| Visual Math | MathVerse | 77 | 48.1 [41.6, 54.7] | 45.2 [39.5, 51.1] | 56.5 [50.2, 62.4] |
| Science & General | PhyX mini MC | 131 | 45.0 [40.0, 50.2] | 38.0 [33.3, 42.8] | 52.4 [48.0, 56.9] |
| Science & General | MMMU-ProVis | 147 | 42.8 [38.2, 47.4] | 37.2 [32.9, 41.6] | 44.7 [40.2, 49.1] |
| Science & General | RealWorldQA | 39 | 56.5 [43.2, 69.4] | 48.7 [36.2, 60.9] | 68.1 [57.4, 77.9] |
| Science & General | MMStar | 90 | 52.8 [45.7, 59.8] | 48.9 [41.5, 56.5] | 62.0 [55.6, 68.0] |
| Spatial Reasoning | EmbSpatial | 167 | 45.2 [39.8, 50.7] | 44.9 [40.0, 49.9] | 66.9 [62.6, 71.1] |
| Spatial Reasoning | SpatialVizBench COT | 178 | 39.4 [34.9, 43.9] | 39.4 [35.0, 43.8] | 46.7 [42.5, 51.0] |
| Spatial Reasoning | CV-Bench 3D | 66 | 50.5 [42.2, 58.7] | 43.9 [35.6, 52.2] | 58.8 [51.6, 65.8] |
| Spatial Reasoning | ERQA | 31 | 42.5 [31.4, 53.5] | 43.0 [31.3, 54.4] | 60.8 [51.3, 69.9] |
| Perception & Counting | BLINK | 177 | 46.4 [41.1, 51.7] | 48.0 [43.1, 52.9] | 57.8 [53.2, 62.2] |
| Perception & Counting | CountBenchQA | 30 | 60.1 [46.8, 72.4] | 59.0 [44.7, 72.4] | 72.5 [61.1, 82.3] |
| Perception & Counting | CountQA | 119 | 44.3 [38.7, 50.0] | 36.6 [31.2, 42.1] | 41.5 [35.5, 47.4] |
| Perception & Counting | TreeBench | 37 | 37.9 [26.7, 49.6] | 37.5 [26.4, 49.1] | 54.9 [44.0, 65.2] |
| Puzzles & Logic | PuzzleVQA | 601 | 61.6 [58.2, 64.9] | 60.1 [56.9, 63.1] | 62.8 [60.2, 65.4] |
| Puzzles & Logic | VisualPuzzles | 209 | 47.6 [43.1, 52.1] | 40.5 [36.3, 44.8] | 50.2 [45.9, 54.4] |
| Puzzles & Logic | LogicVista | 50 | 42.6 [34.2, 51.0] | 39.9 [32.6, 47.4] | 48.3 [40.0, 56.7] |
| Puzzles & Logic | MME-Reasoning | 109 | 36.5 [31.3, 42.0] | 37.5 [32.3, 42.8] | 40.2 [35.1, 45.4] |

*VERO-only capture*

| category | benchmark | mean group size per condition | M050 | CV050 | CT050 |
|---|---|---|---|---|---|
| Charts & Tables | CharXivReason | 78 | 52.6 [45.5, 59.6] | 51.9 [45.4, 58.6] | 41.8 [35.9, 48.0] |
| Charts & Tables | EvoChart | 77 | 59.6 [52.1, 67.2] | 65.2 [58.6, 71.7] | 40.4 [34.3, 47.0] |
| Visual Math | MathVision | 254 | 44.5 [40.6, 48.3] | 43.7 [40.1, 47.4] | 36.5 [33.1, 40.0] |
| Visual Math | MathVista | 70 | 71.3 [65.0, 77.4] | 67.5 [60.6, 74.1] | 49.0 [42.0, 56.3] |
| Visual Math | MathVerse | 78 | 47.6 [40.8, 54.4] | 47.2 [40.8, 53.5] | 33.7 [28.0, 39.7] |
| Science & General | PhyX mini MC | 107 | 54.0 [49.0, 59.1] | 50.5 [45.4, 55.5] | 46.0 [40.9, 51.1] |
| Science & General | MMMU-ProVis | 165 | 50.9 [46.3, 55.6] | 46.2 [41.7, 50.6] | 34.0 [30.1, 38.1] |
| Science & General | RealWorldQA | 47 | 53.9 [43.5, 64.4] | 41.8 [31.7, 52.6] | 27.9 [20.5, 36.3] |
| Science & General | MMStar | 100 | 60.0 [53.5, 66.5] | 52.2 [45.4, 59.0] | 36.8 [31.0, 43.1] |
| Spatial Reasoning | EmbSpatial | 182 | 66.9 [62.6, 71.3] | 65.1 [60.3, 70.0] | 41.3 [36.9, 46.0] |
| Spatial Reasoning | SpatialVizBench COT | 156 | 54.1 [49.1, 58.9] | 44.0 [39.5, 48.7] | 35.0 [30.8, 39.2] |
| Spatial Reasoning | CV-Bench 3D | 100 | 64.7 [58.0, 71.0] | 62.0 [55.7, 68.3] | 44.1 [38.1, 50.4] |
| Spatial Reasoning | ERQA | 32 | 42.3 [32.4, 53.1] | 42.3 [32.4, 53.0] | 29.9 [20.8, 40.5] |
| Perception & Counting | BLINK | 176 | 50.9 [45.7, 56.3] | 43.5 [38.4, 48.7] | 28.5 [24.5, 32.7] |
| Perception & Counting | CountBenchQA | 20 | 80.8 [68.0, 92.3] | 79.2 [69.5, 89.1] | 57.5 [45.3, 70.1] |
| Perception & Counting | CountQA | 131 | 36.5 [30.8, 42.4] | 41.6 [36.1, 47.4] | 30.5 [25.5, 35.6] |
| Perception & Counting | TreeBench | 28 | 44.1 [31.5, 57.3] | 51.8 [39.0, 64.7] | 32.9 [24.0, 42.6] |
| Puzzles & Logic | PuzzleVQA | 57 | 59.9 [53.7, 66.2] | 56.1 [49.3, 63.0] | 44.7 [37.9, 51.7] |
| Puzzles & Logic | VisualPuzzles | 113 | 45.9 [40.2, 51.5] | 40.1 [34.8, 45.4] | 25.9 [21.8, 30.0] |
| Puzzles & Logic | LogicVista | 53 | 51.9 [44.0, 59.7] | 49.7 [42.4, 57.2] | 43.1 [35.3, 51.0] |
| Puzzles & Logic | MME-Reasoning | 117 | 44.7 [39.4, 50.0] | 41.6 [36.2, 47.0] | 30.1 [25.4, 35.1] |

*shared-success damage*

| category | benchmark | mean group size per condition | M050 | CV050 | CT050 |
|---|---|---|---|---|---|
| Charts & Tables | CharXivReason | 289 | 6.2 [4.6, 8.0] | 12.8 [10.2, 15.6] | 17.6 [14.6, 20.8] |
| Charts & Tables | EvoChart | 563 | 4.1 [3.1, 5.2] | 6.0 [4.6, 7.5] | 9.0 [7.3, 10.7] |
| Visual Math | MathVision | 394 | 17.1 [14.9, 19.5] | 22.3 [19.9, 24.8] | 27.1 [24.4, 29.8] |
| Visual Math | MathVista | 540 | 3.6 [2.7, 4.5] | 6.2 [4.9, 7.7] | 9.6 [8.0, 11.3] |
| Visual Math | MathVerse | 221 | 18.0 [15.0, 21.2] | 20.3 [17.1, 23.7] | 23.2 [19.7, 26.8] |
| Science & General | PhyX mini MC | 259 | 14.4 [12.0, 17.1] | 22.9 [19.9, 26.2] | 22.3 [19.5, 25.4] |
| Science & General | MMMU-ProVis | 397 | 11.6 [9.7, 13.5] | 15.0 [12.8, 17.3] | 17.8 [15.5, 20.3] |
| Science & General | RealWorldQA | 383 | 2.6 [1.5, 3.8] | 3.8 [2.5, 5.2] | 5.5 [3.8, 7.2] |
| Science & General | MMStar | 704 | 2.9 [2.0, 3.8] | 4.3 [3.3, 5.4] | 6.2 [5.1, 7.3] |
| Spatial Reasoning | EmbSpatial | 1,889 | 2.9 [2.5, 3.4] | 3.9 [3.4, 4.5] | 6.7 [6.0, 7.4] |
| Spatial Reasoning | SpatialVizBench COT | 156 | 20.3 [16.7, 24.4] | 35.5 [30.6, 40.8] | 37.8 [33.4, 42.3] |
| Spatial Reasoning | CV-Bench 3D | 711 | 4.7 [3.7, 5.8] | 6.1 [4.9, 7.3] | 9.8 [8.4, 11.3] |
| Spatial Reasoning | ERQA | 98 | 12.9 [9.1, 16.9] | 16.9 [12.7, 21.6] | 20.2 [15.7, 24.8] |
| Perception & Counting | BLINK | 681 | 5.6 [4.5, 6.7] | 7.8 [6.4, 9.3] | 11.7 [10.1, 13.3] |
| Perception & Counting | CountBenchQA | 307 | 2.5 [1.4, 4.0] | 2.6 [1.5, 3.9] | 5.3 [3.8, 7.0] |
| Perception & Counting | CountQA | 163 | 17.6 [13.5, 21.9] | 19.9 [15.8, 24.5] | 21.9 [17.7, 26.5] |
| Perception & Counting | TreeBench | 94 | 6.0 [3.3, 9.3] | 8.0 [4.8, 11.7] | 12.1 [8.4, 16.5] |
| Puzzles & Logic | PuzzleVQA | 193 | 12.6 [9.7, 15.9] | 15.6 [12.3, 19.2] | 24.3 [20.0, 28.7] |
| Puzzles & Logic | VisualPuzzles | 114 | 25.4 [20.8, 30.3] | 31.3 [26.5, 36.3] | 33.2 [28.0, 38.6] |
| Puzzles & Logic | LogicVista | 122 | 10.1 [7.2, 13.5] | 14.2 [10.6, 18.2] | 19.1 [14.5, 24.3] |
| Puzzles & Logic | MME-Reasoning | 168 | 18.1 [14.6, 21.7] | 20.6 [16.8, 24.8] | 29.6 [25.2, 34.1] |

*joint-failure rescue*

| category | benchmark | mean group size per condition | M050 | CV050 | CT050 |
|---|---|---|---|---|---|
| Charts & Tables | CharXivReason | 352 | 8.6 [6.7, 10.6] | 10.3 [8.3, 12.4] | 11.8 [9.7, 14.1] |
| Charts & Tables | EvoChart | 277 | 7.9 [5.7, 10.3] | 6.8 [4.9, 8.9] | 8.0 [6.1, 10.0] |
| Visual Math | MathVision | 1,529 | 9.3 [8.4, 10.3] | 10.2 [9.2, 11.1] | 9.9 [9.1, 10.9] |
| Visual Math | MathVista | 134 | 12.3 [9.0, 16.0] | 15.8 [12.0, 19.8] | 12.0 [8.7, 15.7] |
| Visual Math | MathVerse | 255 | 9.8 [7.8, 12.0] | 9.6 [7.5, 11.8] | 10.0 [7.9, 12.3] |
| Science & General | PhyX mini MC | 303 | 16.4 [14.0, 18.9] | 18.6 [16.0, 21.4] | 21.1 [18.5, 23.8] |
| Science & General | MMMU-ProVis | 675 | 10.4 [9.0, 11.9] | 10.8 [9.5, 12.2] | 11.5 [10.1, 13.0] |
| Science & General | RealWorldQA | 143 | 5.9 [3.6, 8.5] | 6.2 [3.6, 9.1] | 7.9 [5.0, 11.2] |
| Science & General | MMStar | 306 | 8.6 [6.7, 10.7] | 8.9 [7.0, 11.0] | 11.4 [9.3, 13.6] |
| Spatial Reasoning | EmbSpatial | 673 | 9.0 [7.7, 10.3] | 10.2 [8.8, 11.6] | 12.9 [11.4, 14.5] |
| Spatial Reasoning | SpatialVizBench COT | 454 | 16.6 [14.5, 18.9] | 19.6 [17.2, 22.0] | 23.5 [21.1, 26.0] |
| Spatial Reasoning | CV-Bench 3D | 83 | 17.3 [12.6, 22.4] | 19.7 [13.9, 26.1] | 28.7 [22.8, 34.9] |
| Spatial Reasoning | ERQA | 156 | 10.3 [7.5, 13.7] | 12.4 [9.2, 16.0] | 12.8 [9.8, 16.2] |
| Perception & Counting | BLINK | 488 | 14.0 [11.9, 16.1] | 11.8 [10.0, 13.6] | 16.1 [14.0, 18.2] |
| Perception & Counting | CountBenchQA | 33 | 21.7 [12.7, 33.0] | 26.3 [16.0, 38.8] | 28.3 [19.0, 39.1] |
| Perception & Counting | CountQA | 807 | 7.4 [6.1, 8.8] | 8.1 [6.8, 9.4] | 7.8 [6.6, 9.1] |
| Perception & Counting | TreeBench | 165 | 5.5 [3.5, 7.7] | 5.6 [3.6, 7.9] | 9.4 [6.9, 12.3] |
| Puzzles & Logic | PuzzleVQA | 745 | 19.8 [17.6, 22.1] | 23.6 [21.4, 26.0] | 28.0 [25.7, 30.4] |
| Puzzles & Logic | VisualPuzzles | 496 | 14.7 [12.9, 16.7] | 17.9 [15.9, 19.9] | 19.4 [17.4, 21.5] |
| Puzzles & Logic | LogicVista | 133 | 17.2 [13.3, 21.4] | 20.3 [16.5, 24.4] | 18.5 [14.9, 22.5] |
| Puzzles & Logic | MME-Reasoning | 551 | 9.3 [7.9, 10.9] | 9.7 [8.2, 11.3] | 11.1 [9.5, 12.7] |

Across the 21 benchmarks, M050 has the lowest shared-success damage of the three merges on 21, keeps more VERO-only successes than CT050 on 21, keeps fewer TRACE-only successes than CT050 on 19, and has the lowest joint-failure rescue on 17.

### 7.9 Answer format (`confirmation/answer_format_sensitivity.csv`; mean over seeds, % of rows)

| model | PuzzleVQA accuracy | PuzzleVQA: ends in \( X \) | PuzzleVQA: scored 0, final letter correct | VisualPuzzles accuracy | VisualPuzzles: ends in \( X \) | VisualPuzzles: scored 0, final letter correct |
|---|---|---|---|---|---|---|
| B | 43.9 | 0.0 | 2.7 | 32.1 | 0.4 | 1.1 |
| T | 49.7 | 0.0 | 1.9 | 34.7 | 0.5 | 0.5 |
| V@P0 | 48.8 | 0.0 | 0.0 | 37.7 | 0.0 | 0.0 |
| V@P1 | 15.7 | 27.6 | 32.0 | 24.4 | 24.6 | 12.6 |
| M050 | 45.1 | 0.9 | 5.7 | 33.2 | 7.7 | 3.6 |
| CV050 | 45.9 | 0.5 | 5.5 | 31.9 | 4.3 | 4.0 |
| CT050 | 47.5 | 0.0 | 1.7 | 32.9 | 0.2 | 0.5 |

### 7.10 Length, finish reasons and extraction (`length_format_judge_audit.csv`)

B, T and V@P0 come from the archive (full split, 98,415 responses each). The new models come from the confirmation runs (78,684 responses each). Rates are row-weighted.

| model | responses | length stops | length stops with a row score / scored 0 | median of per-benchmark median completion tokens | with </think> | with \boxed{} | with "The answer is" | with any final-answer marker | extraction not resolved | judge "incomplete" sentinels |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 98,415 | 313 | 121 / 118 | 96 | 0.9% | 19.4% | 22.4% | 53.3% | 0.006% | 16 |
| T | 98,415 | 14 | 9 / 8 | 179 | 2.4% | 25.5% | 20.9% | 61.9% | 0.005% | 0 |
| V@P0 | 98,415 | 8 | 3 / 3 | 228 | 100.0% | 67.8% | 31.4% | 100.0% | 0.000% | 0 |
| V@P1 | 78,684 | 8 | 0 / 0 | 127 | 3.6% | 39.0% | 41.2% | 68.4% | 0.000% | 0 |
| M050 | 78,684 | 9 | 2 / 2 | 170 | 3.6% | 30.1% | 34.8% | 65.1% | 0.003% | 0 |
| CV050 | 78,684 | 15 | 2 / 2 | 125 | 3.6% | 26.3% | 35.9% | 61.3% | 0.003% | 1 |
| CT050 | 78,684 | 8 | 2 / 2 | 149 | 1.3% | 23.1% | 22.5% | 58.5% | 0.004% | 2 |

A judge "incomplete" sentinel is the judge's own statement that a response gives no answer. Length stops without a row score are on benchmarks whose rows are re-derived (§2.7).

Median completion tokens per benchmark (mean of the three seeds' medians; archived models on the full split, new models on the confirmation partition):

| category | benchmark | B | T | V@P0 | V@P1 | M050 | CV050 | CT050 |
|---|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 154 | 227 | 25 | 8 | 211 | 132 | 202 |
| Charts & Tables | CharXivReason | 20 | 92 | 193 | 129 | 89 | 44 | 27 |
| Charts & Tables | TableVQABench | 3 | 5 | 21 | 5 | 5 | 4 | 3 |
| Charts & Tables | EvoChart | 37 | 144 | 244 | 126 | 123 | 101 | 98 |
| Visual Math | MathVision | 419 | 438 | 640 | 456 | 477 | 462 | 440 |
| Visual Math | MathVista | 142 | 205 | 304 | 201 | 207 | 180 | 174 |
| Visual Math | MathVerse | 363 | 365 | 544 | 347 | 372 | 368 | 363 |
| Visual Math | WeMath | 156 | 297 | 368 | 207 | 218 | 198 | 177 |
| Science & General | PhyX mini MC | 2 | 547 | 547 | 571 | 578 | 550 | 573 |
| Science & General | MMMU-ProVis | 381 | 393 | 409 | 340 | 379 | 371 | 392 |
| Science & General | RealWorldQA | 5 | 6 | 122 | 11 | 7 | 5 | 5 |
| Science & General | MMStar | 5 | 7 | 206 | 82 | 11 | 6 | 6 |
| Spatial Reasoning | EmbSpatial | 2 | 2 | 15 | 2 | 2 | 2 | 2 |
| Spatial Reasoning | SpatialVizBench COT | 303 | 333 | 378 | 342 | 311 | 304 | 315 |
| Spatial Reasoning | CV-Bench 3D | 120 | 151 | 214 | 106 | 133 | 124 | 123 |
| Spatial Reasoning | ERQA | 2 | 2 | 15 | 2 | 2 | 2 | 2 |
| Perception & Counting | BLINK | 5 | 6 | 196 | 54 | 6 | 6 | 6 |
| Perception & Counting | CountBenchQA | 2 | 2 | 15 | 2 | 2 | 2 | 2 |
| Perception & Counting | CountQA | 40 | 124 | 182 | 84 | 119 | 93 | 103 |
| Perception & Counting | TreeBench | 2 | 2 | 15 | 2 | 2 | 2 | 2 |
| Puzzles & Logic | PuzzleVQA | 200 | 236 | 247 | 246 | 251 | 234 | 218 |
| Puzzles & Logic | VisualPuzzles | 292 | 279 | 341 | 286 | 314 | 314 | 276 |
| Puzzles & Logic | LogicVista | 253 | 301 | 394 | 294 | 307 | 276 | 275 |
| Puzzles & Logic | MME-Reasoning | 326 | 346 | 425 | 359 | 371 | 357 | 341 |

Responses containing `\boxed{}`, per benchmark (% of rows, row-weighted over seeds):

| category | benchmark | B | T | V@P0 | V@P1 | M050 | CV050 | CT050 |
|---|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 0.0 | 0.0 | 1.5 | 0.1 | 0.0 | 0.0 | 0.0 |
| Charts & Tables | CharXivReason | 0.7 | 25.0 | 96.5 | 29.7 | 26.2 | 5.8 | 4.7 |
| Charts & Tables | TableVQABench | 0.0 | 0.1 | 13.3 | 0.2 | 0.1 | 0.0 | 0.0 |
| Charts & Tables | EvoChart | 97.3 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| Visual Math | MathVision | 69.6 | 85.0 | 100.0 | 98.1 | 97.5 | 92.3 | 79.1 |
| Visual Math | MathVista | 11.2 | 26.5 | 100.0 | 89.7 | 57.2 | 36.3 | 17.7 |
| Visual Math | MathVerse | 97.8 | 98.9 | 100.0 | 99.9 | 100.0 | 99.9 | 99.3 |
| Visual Math | WeMath | 0.0 | 0.2 | 88.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| Science & General | PhyX mini MC | 11.9 | 68.5 | 95.8 | 96.4 | 87.5 | 68.0 | 62.8 |
| Science & General | MMMU-ProVis | 1.3 | 4.6 | 77.3 | 1.9 | 3.8 | 1.5 | 2.7 |
| Science & General | RealWorldQA | 0.0 | 0.0 | 98.4 | 1.1 | 0.0 | 0.0 | 0.0 |
| Science & General | MMStar | 1.4 | 2.7 | 100.0 | 27.8 | 10.4 | 5.3 | 1.6 |
| Spatial Reasoning | EmbSpatial | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Spatial Reasoning | SpatialVizBench COT | 0.0 | 0.0 | 100.0 | 89.0 | 0.0 | 0.1 | 0.0 |
| Spatial Reasoning | CV-Bench 3D | 0.0 | 0.0 | 100.0 | 57.3 | 0.3 | 0.0 | 0.0 |
| Spatial Reasoning | ERQA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Perception & Counting | BLINK | 0.0 | 0.4 | 99.1 | 16.1 | 7.9 | 4.2 | 0.0 |
| Perception & Counting | CountBenchQA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Perception & Counting | CountQA | 95.7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.9 |
| Perception & Counting | TreeBench | 0.0 | 0.0 | 16.8 | 4.4 | 0.2 | 0.0 | 0.0 |
| Puzzles & Logic | PuzzleVQA | 0.0 | 0.0 | 33.3 | 0.5 | 0.0 | 0.0 | 0.0 |
| Puzzles & Logic | VisualPuzzles | 1.1 | 0.4 | 64.6 | 7.2 | 1.4 | 1.5 | 0.8 |
| Puzzles & Logic | LogicVista | 4.8 | 31.8 | 99.9 | 96.9 | 70.4 | 36.4 | 9.3 |
| Puzzles & Logic | MME-Reasoning | 40.1 | 63.5 | 98.6 | 86.0 | 79.8 | 66.6 | 53.5 |

### 7.11 Sensitivity analyses

Without the 130 confirmation rows that have a near-duplicate development twin (`confirmation/sensitivity_without_dev_twins.csv`):

| comparison | all rows | without twins |
|---|---|---|
| M050 − T | +0.23 [−0.19, +0.65] | +0.23 [−0.19, +0.66] |
| M050 − V@P1 | +3.07 [+2.62, +3.51] | +3.03 [+2.60, +3.48] |
| M050 − CV050 | +1.29 [+0.93, +1.64] | +1.28 [+0.93, +1.62] |
| M050 − CT050 | +1.88 [+1.45, +2.31] | +1.85 [+1.44, +2.28] |
| T − B | +4.02 [+3.58, +4.45] | +3.97 [+3.54, +4.41] |
| V@P1 − B | +1.19 [+0.65, +1.72] | +1.17 [+0.63, +1.70] |
| V@P1 − V@P0 | −2.94 [−3.33, −2.56] | −2.91 [−3.30, −2.52] |
| M050 − B | +4.25 [+3.78, +4.72] | +4.20 [+3.74, +4.67] |
| CV050 − B | +2.96 [+2.52, +3.38] | +2.92 [+2.51, +3.34] |
| CT050 − B | +2.38 [+2.00, +2.75] | +2.35 [+1.99, +2.72] |

Seed 42 only, with the archived B/T or with local reruns of B/T made in the new environment (`confirmation/sensitivity_seed42.*.csv`):

| comparison | archived B/T | local-rerun B/T |
|---|---|---|
| M050 − T | +0.15 [−0.46, +0.77] | +0.27 [−0.34, +0.88] |
| M050 − V@P1 | +3.26 [+2.67, +3.86] | +3.26 [+2.67, +3.86] |
| M050 − CV050 | +1.65 [+1.11, +2.18] | +1.65 [+1.11, +2.18] |
| M050 − CT050 | +1.94 [+1.32, +2.56] | +1.94 [+1.32, +2.56] |
| CV050 − B | +2.67 [+2.02, +3.30] | +2.92 [+2.29, +3.57] |
| CT050 − B | +2.38 [+1.76, +3.00] | +2.63 [+2.04, +3.24] |
| V@P1 − B | +1.05 [+0.34, +1.77] | +1.31 [+0.60, +2.02] |
| T − B | +4.16 [+3.50, +4.82] | +4.31 [+3.66, +4.97] |
| M050 − B | +4.32 [+3.64, +4.99] | +4.57 [+3.91, +5.26] |

By category, seed 42 (only the comparisons that involve B or T change; the others are identical in both files):

| comparison | category | archived B/T | local-rerun B/T | shift |
|---|---|---|---|---|
| M050 − T | Charts & Tables | −0.92 [−2.09, +0.24] | −0.48 [−1.66, +0.69] | +0.45 |
| M050 − T | Visual Math | +0.39 [−1.21, +1.99] | +0.79 [−0.80, +2.33] | +0.40 |
| M050 − T | Science & General | +0.61 [−0.71, +1.96] | +0.44 [−0.88, +1.76] | −0.17 |
| M050 − T | Spatial Reasoning | +1.58 [+0.17, +3.04] | +1.06 [−0.39, +2.56] | −0.51 |
| M050 − T | Perception & Counting | +0.00 [−1.46, +1.46] | +0.80 [−0.63, +2.21] | +0.80 |
| M050 − T | Puzzles & Logic | −0.73 [−2.62, +1.14] | −1.02 [−2.99, +0.92] | −0.29 |
| CV050 − B | Charts & Tables | +1.99 [+0.72, +3.26] | +2.30 [+1.04, +3.59] | +0.32 |
| CV050 − B | Visual Math | +3.73 [+2.11, +5.36] | +4.54 [+2.89, +6.20] | +0.81 |
| CV050 − B | Science & General | +3.92 [+2.46, +5.39] | +3.60 [+2.16, +5.05] | −0.32 |
| CV050 − B | Spatial Reasoning | +1.55 [+0.04, +3.06] | +1.66 [+0.15, +3.13] | +0.11 |
| CV050 − B | Perception & Counting | +3.11 [+1.72, +4.52] | +2.81 [+1.40, +4.27] | −0.30 |
| CV050 − B | Puzzles & Logic | +1.70 [−0.40, +3.74] | +2.64 [+0.62, +4.67] | +0.94 |
| CT050 − B | Charts & Tables | +2.31 [+1.19, +3.44] | +2.63 [+1.51, +3.75] | +0.32 |
| CT050 − B | Visual Math | +1.74 [+0.06, +3.40] | +2.55 [+0.82, +4.25] | +0.81 |
| CT050 − B | Science & General | +3.93 [+2.56, +5.33] | +3.61 [+2.24, +5.00] | −0.32 |
| CT050 − B | Spatial Reasoning | +1.67 [+0.27, +3.02] | +1.78 [+0.37, +3.17] | +0.11 |
| CT050 − B | Perception & Counting | +2.64 [+1.56, +3.76] | +2.34 [+1.22, +3.48] | −0.30 |
| CT050 − B | Puzzles & Logic | +1.97 [−0.04, +3.92] | +2.90 [+0.98, +4.77] | +0.94 |
| V@P1 − B | Charts & Tables | −0.84 [−2.28, +0.58] | −0.52 [−1.94, +0.92] | +0.32 |
| V@P1 − B | Visual Math | +4.30 [+2.55, +6.07] | +5.11 [+3.39, +6.87] | +0.81 |
| V@P1 − B | Science & General | +4.94 [+3.30, +6.59] | +4.62 [+2.96, +6.25] | −0.32 |
| V@P1 − B | Spatial Reasoning | +2.77 [+0.95, +4.57] | +2.87 [+1.03, +4.73] | +0.11 |
| V@P1 − B | Perception & Counting | +2.25 [+0.58, +3.96] | +1.95 [+0.26, +3.67] | −0.30 |
| V@P1 − B | Puzzles & Logic | −7.11 [−9.15, −5.06] | −6.18 [−8.31, −4.12] | +0.94 |
| T − B | Charts & Tables | +5.33 [+4.03, +6.63] | +5.20 [+3.86, +6.52] | −0.13 |
| T − B | Visual Math | +5.65 [+3.91, +7.38] | +6.06 [+4.35, +7.78] | +0.41 |
| T − B | Science & General | +4.87 [+3.34, +6.38] | +4.72 [+3.20, +6.23] | −0.15 |
| T − B | Spatial Reasoning | +1.90 [+0.37, +3.40] | +2.52 [+1.03, +4.01] | +0.62 |
| T − B | Perception & Counting | +3.45 [+2.13, +4.77] | +2.34 [+0.97, +3.71] | −1.10 |
| T − B | Puzzles & Logic | +3.78 [+1.70, +5.87] | +5.01 [+2.95, +7.03] | +1.23 |
| M050 − B | Charts & Tables | +4.41 [+3.06, +5.77] | +4.73 [+3.37, +6.10] | +0.32 |
| M050 − B | Visual Math | +6.04 [+4.38, +7.71] | +6.85 [+5.14, +8.55] | +0.81 |
| M050 − B | Science & General | +5.49 [+3.92, +7.03] | +5.16 [+3.59, +6.67] | −0.32 |
| M050 − B | Spatial Reasoning | +3.47 [+1.85, +5.14] | +3.58 [+1.96, +5.22] | +0.11 |
| M050 − B | Perception & Counting | +3.45 [+2.02, +4.94] | +3.14 [+1.67, +4.65] | −0.30 |
| M050 − B | Puzzles & Logic | +3.05 [+0.89, +5.16] | +3.99 [+1.83, +6.10] | +0.94 |

- Category intervals that change side of 0 (a bound within 1e-9 of 0 counts as touching it): 3 (M050 − T in Spatial Reasoning; CV050 − B in Puzzles & Logic; CT050 − B in Puzzles & Logic).
- Benchmark intervals that change side of 0: 10 (M050 − T on MMStar; M050 − T on BLINK; CV050 − B on PuzzleVQA; CT050 − B on MathVision; CT050 − B on MathVista; CT050 − B on PuzzleVQA; V@P1 − B on MMStar (a bound at exactly 0); T − B on MathVision; T − B on VisualPuzzles (a bound at exactly 0); M050 − B on LogicVista).
- Without the development twins, category contrasts move by at most 0.19 pp; benchmark intervals that change side of 0: 1 (M050 − CV050 on MathVista, +1.88 [+0.27, +3.47] → +1.59 [−0.05, +3.21]).

## 8. Technical record

### 8.1 Judge health (`judge_extraction_agreement.csv`)

For each run's MathVerse slice: the share of rows whose judge-extracted answer equals the response's own final `\boxed{}` answer. The floor is 30% (DESIGN.md A8).

| run | tier | seed | quarantined | rows with a boxed answer | extracted = boxed (%) | below floor |
|---|---|---|---|---|---|---|
| CT050 | confirmation | 42 |  | 561 | 63.3 |  |
| CT050 | confirmation | 43 |  | 573 | 67.4 |  |
| CT050 | confirmation | 44 |  | 566 | 71.7 |  |
| CV050 | confirmation | 42 |  | 583 | 66.6 |  |
| CV050 | confirmation | 43 |  | 574 | 69.3 |  |
| CV050 | confirmation | 44 |  | 586 | 69.8 |  |
| M050 | confirmation | 42 |  | 586 | 66.0 |  |
| M050 | confirmation | 43 |  | 587 | 79.7 |  |
| M050 | confirmation | 44 |  | 582 | 69.2 |  |
| M050 | confirmation | 44 | yes | 582 | 14.8 | yes |
| V@P1 | confirmation | 42 |  | 589 | 69.1 |  |
| V@P1 | confirmation | 43 |  | 589 | 72.0 |  |
| V@P1 | confirmation | 44 |  | 587 | 82.5 |  |
| CT050 | development screen | 42 |  | 120 | 60.0 |  |
| CT050@P2 | development screen | 42 |  | 123 | 57.7 |  |
| CV050 | development screen | 42 |  | 122 | 61.5 |  |
| CV050@P2 | development screen | 42 |  | 121 | 81.8 |  |
| M050 | development screen | 42 |  | 124 | 70.2 |  |
| M050@P2 | development screen | 42 |  | 121 | 59.5 |  |
| B@P2 | development screen | 42 |  | 114 | 43.0 |  |
| T@P2 | development screen | 42 |  | 116 | 56.9 |  |
| V@P1 | development screen | 42 |  | 120 | 87.5 |  |
| CT050 | smoke | 42 |  | 3 | 66.7 |  |
| CV050 | smoke | 42 |  | 4 | 75.0 |  |
| M050 | smoke | 42 |  | 4 | 50.0 |  |
| V@P1 | smoke | 42 |  | 4 | 75.0 |  |

### 8.2 The A8 repair: M050 confirmation seed 44 before and after rescoring

The fault's signature (DESIGN.md A9): the phrase "They give the solutions to the equation", from one of the judge prompt's few-shot examples, appears in 133 of the 630 quarantined MathVerse extraction outputs and in 0 of 630 after the rescore; 297 against 540 are short single-line answers. The cause inside the serving stack was not isolated.

The quarantined first scoring was done with the judge at tensor-parallel 4 on A800 cards; the rescore used tensor-parallel 1 on RTX PRO 6000 cards. The values are the score campaign's own per-benchmark aggregates from the runner summaries. WeMath is the campaign's fixed-525 value; the study's WeMath metric differs (DESIGN.md A2).

| benchmark | seed 44 before | seed 44 after | change | seed 42 (study metric) | seed 43 (study metric) |
|---|---|---|---|---|---|
| MathVision | 27.18 | 27.92 | +0.74 | 28.78 | 27.80 |
| MathVista | 67.25 | 76.63 | +9.38 | 78.00 | 77.75 |
| MathVerse | 16.98 | 42.38 | +25.40 | 43.65 | 47.30 |
| MMStar | 68.08 | 68.17 | +0.08 | 67.67 | 68.50 |
| SpatialVizBench COT | 37.50 | 37.08 | −0.42 | 38.77 | 36.65 |
| LogicVista | 47.21 | 49.44 | +2.23 | 50.84 | 51.68 |

Only benchmarks whose campaign value changed are listed. Every other benchmark is identical before and after.

Effect on the primary contrast:
- Before the repair, M050 − T was −0.29: the point estimate in the superseded tables, as quoted in `analysis_report.md` and `decision_memo.md`.
- After it, M050 − T is +0.23 [−0.19, +0.65].
- The old confirmation tables are superseded. `analysis_report.md` §4 and `decision_memo.md` quote only this pre-repair estimate and the Visual Math shortfall it showed, to describe the fault.

## 9. Where the data lives (`experiments/trace_merge/results/`)

| path | contents |
|---|---|
| `reports/analysis_report.md`, `reports/decision_memo.md` | interpretation, caveats, decision |
| `reports/behavior_report.md`, `reports/merge_screen.md` | frozen Stage B report; development-screen report |
| `reports/selection_lock.json` | hashes frozen before confirmation |
| `tables/stage_b/` | Stage B scores, contrasts, corrections/regressions, joint cells, overlap, persistence |
| `tables/screen/` | development screen: P1/P2 scores, contrasts, seed scores, capture/damage, answer format |
| `tables/confirmation/` | confirmation: scores, seed scores, contrasts, capture/damage (+ base reference), answer format, sensitivities |
| `tables/length_format_judge_audit.csv` | lengths, finish reasons, extraction and judge use per model, seed and benchmark |
| `tables/judge_extraction_agreement.csv` | judge-health check per run |
| `tables/evaluation_costs.csv` | GPU-hours, attempts, tokens per run |
| `tables/evaluation_status.json`, `tables/environment_drift.csv` | run completeness; local reruns against the archive per benchmark |
| `audit/compatibility.json`, `audit/parent_update_statistics.json` | parent compatibility; update norms and cosines |
| `audit/protocol_comparison.json`, `audit/missingness.csv` | prompt conformance, settings, environment drift; row completeness |
| `audit/metric_reproduction.json` | metric adapters against the archived aggregates |
| `audit/` | asset, archive, metric-reproduction, protocol and environment audits |
| `manifests/` | model roles, inference views, merges, metric registry, input manifest |
| `splits/` | partition and bootstrap-unit manifests |
| `figures/` | Stage B, screen and confirmation figures |
| `merges/*/merge_manifest.json` | merge recipes and hashes |
| `tests/test_report.txt` | test-suite run |
