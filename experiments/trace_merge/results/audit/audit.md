# Stage A audit: assets, identity, metrics, protocol

Study `trace_vero_qwen25_v1`. Machine-readable evidence for every line is listed below each section. Paths are relative
to the local work root.

## 1. Release and code

- `rlvr/evaluation/scripts/validate_release_inputs.py` passes on the unmodified `rlvr` tree (af32fe0), both before and
  after the study was added. The study lives in `experiments/trace_merge/` because the validator's source-map check and
  `rlvr/train.py` reject any unmapped file under `rlvr/`.
- New generations used the prepared protocol clone at af32fe0 with VLMEvalKit `a8b12bf1` and the Qwen3-32B judge
  `9216db57`, plus three fail-closed scorer fixes:
  - the CharXiv judge parser that the public release omits;
  - MME-Reasoning official-function exceptions are scored incorrect;
  - a judge output still unusable after the official five attempts is scored incorrect.

  Each spool descriptor of a new run records the pinned commits and code hash, and ingestion checks them.

## 2. Archive integrity (`audit/archive_ingest.json`, `manifests/input_manifest.json`)

- Archive: `maveryn/trace-eval-runs@4178a839`, run `qwen2.5-vl-7b-comparison-temp06-seeds42-44-v1`.
  - All 1,303 files (375 MB) match their sha256 and size in the archive manifest.
  - The manifest's sha256 (`0334afa7…`) and the results file's (`578e574d…`, `metadata/results/benchmark_scores.json`)
    equal the release receipts.
  - All 24 licensed TSVs of the prepared data match the dataset manifest.
- Content: 295,245 rows in 216 slices (3 models × 3 seeds × 24 benchmarks).
  - Every content key (benchmark, ordinal, index, row hash, record hash) appears exactly once per slice.
  - The 9 slices of each benchmark have identical key sets.
  - Response/extraction/score joins are one-to-one.
  - Every row matches exactly one prepared-dataset row, including its media-set hash.
- No missing rows. `audit/missingness.csv` records zero gaps for the archive.
- One verified alias, for CountQA (DESIGN amendment A1):
  - The canonical producer hashed CountQA's cached integer answer before the public
    `countqa-cached-answer-normalization` patch.
  - Recomputing both hashes from the licensed TSV reproduces the archived and the prepared value on all 1,528 rows.
  - Archive rows keep their original hashes in separate columns.
- Generation settings: one normalized setting across all 216 slices, equal to the suite (temperature 0.6, top-p 1,
  top-k −1, max 4,096 tokens, file-url media, 3,136 to 12,845,056 pixels). Seeds 42/43/44.
- Not reproducible from the public data: the archive's whole-dataset digest (`50e67749…`); its construction is not
  published. The per-row identity checks above cover the same content.

## 3. Models (`audit/compatibility.json`, `audit/parent_weight_sha256.json`, `audit/parent_update_statistics.json`)

- B `Qwen/Qwen2.5-VL-7B-Instruct@cc594898`, T `maveryn/trace-qwen2.5-vl-7b@4d0f1ae8` and
  V `zlab-princeton/Vero-Qwen25-7B@180e84be`: every weight file's full sha256 equals its LFS object id.
- T's archived runtime view `sha256set:8396b0be…` recomputes from the 10 files listed in T's own repository marker, all
  identical to the pinned snapshot. That view shares its weights with the HF commit.
- Compatibility: **compatible**, with no hard stops.
  - Identical architecture: 28 language layers, 32 vision blocks, untied embeddings.
  - Identical 729 tensor names, shapes and dtypes (all BF16, 8.29B parameters).
  - Identical token-to-id maps, 22 added tokens and special tokens.
  - Identical image-processor bounds and normalization.
- Recorded differences:
  - B declares the slow Qwen2-VL image processor and T/V the fast one. The pinned transformers loads the fast one for
    all three (vLLM log).
  - T's generation config lacks `<|endoftext|>` as an extra stop id.
  - V's config declares float32; vLLM downcasts it to bf16.
- Lineage:
  - T declares `Qwen/Qwen2.5-VL-7B-Instruct@cc594898` as its base.
  - V's model card names `Qwen/Qwen2.5-VL-7B-Instruct` without a revision.

  Task-vector readings of V − B therefore assume, without proof, that V started from this revision.
- Update diagnostic, relative to B (no claim about skills):
  - V's relative update is 0.34–0.67% per parameter group; T's is 0.05–0.18%.
  - The cosine between T − B and V − B is 0.0002–0.008 in every group (0.06 for the patch embedding).
  - Both recipes changed the vision tower. 58–62 of its 384 tensors are bitwise unchanged, and the final language
    norm is unchanged in both.

## 4. Protocol (`audit/protocol_comparison.json`)

- `prompt_sha256` (message payload) agrees on every row across B, T and V, and so does the model-neutral request digest.
  The rendered prompt is not the same:
  - B and T have identical prompt-token counts on every row (median 674).
  - V's native template adds a fixed think/answer system prompt and a `<think>` prefill: +486 tokens at the paired
    median.
  - V wraps 100% of its responses in `<think>`/`<answer>`.

  This is why the merge study fixes one interface: P1, the backbone's files. V is re-run under P1; B's and T's archived
  outputs are already P1 outputs.
- Scorer ids and archived score-part code digests are identical across models.
- Length stops: B 313, T 14, V 8 over 98,415 responses each.

## 5. Environment drift (`audit/environment_drift_*.json`, `tables/environment_drift.csv`)

Existing local full-suite seed-42 reruns of B and T were made earlier for another study, in the same pinned
environment on different hardware, and were reused rather than repeated. Their requests are byte-identical to the
archive on 32,805 of 32,805 rows: same prompt digest, same neutral request digest, same prompt-token counts.

| | B rerun vs archive | B, archived seed pairs | T rerun vs archive | T, archived seed pairs |
|---|---|---|---|---|
| 24-benchmark macro delta (pp) | −0.25 | −0.53 to −0.04 | −0.02 | −0.28 to +0.32 |
| mean / max absolute benchmark delta (pp) | 0.56 / 1.52 | 1.40–1.56 / 5.3–6.8 | 0.81 / 2.46 | 0.86–0.95 / 2.47–4.20 |
| row-score agreement | 0.926 | 0.782–0.819 | 0.924 | 0.835–0.853 |
| length stops | 11 | 107 / 112 / 94 | 0 | 4 / 7 / 3 |

The rerun differs from the archive by less than two archived seeds differ from each other. Part of the higher
agreement comes from sharing seed 42. Archived and new runs are treated as exchangeable draws of the same protocol;
the confirmation report repeats its seed-42 comparisons with the local B/T reruns substituted.

## 6. Metrics (`audit/metric_reproduction.json`, `manifests/metric_registry.json`)

- All 24 metric adapters were accepted. Each reproduces the archived aggregate of all 9 slices of its benchmark,
  maximum error 4.5e-11 pp (archived rounding). They cover:
  - 16 pooled 0/1 row means;
  - WeMath's strict group metric;
  - TableVQABench's five-split mean;
  - five pooled metrics (MathVision, MathVista, CountBenchQA, PuzzleVQA, VisualPuzzles) and ChartQAPro, re-derived
    from archived extractions with the pinned scorer functions and no new judge calls.
- `results.json` per-benchmark values equal the archived aggregates exactly.
- On the existing local reruns, every adapter equals that run's own `scores.json`: 12,883 rows compared row by row,
  0 mismatches, 0 per-stratum mismatches.
- Subset semantics: pooled means and the TableVQABench split mean are computed on the subset's rows.
  - WeMath uses the G complete problem groups present (amendment A2). The pinned scorer's fixed 525 would score absent
    groups as mastered; a new subset run is checked against that pinned value separately.
  - Binary diagnostics use 21 benchmarks (amendment A3).

## 7. Splits (`splits/split_manifest.json`, frozen 2026-11-02T05:16Z before any merged-model output)

| item | value |
|---|---|
| connected image/problem groups | 26,422 |
| groups spanning several benchmarks | 179 |
| largest group | 28 rows |
| development partition | 6,577 rows (20.0%; each benchmark 19.9–20.5%) |
| confirmation partition | 26,228 rows |
| development screen | 2,910 rows (81–129 per benchmark; strata covered) |
| smoke list | 135 rows |

Groups join rows that share any image (sha256, across all benchmarks) or a WeMath problem `ID`. All seeds and
variants of a group stay in one partition.

## 8. What was not verified

- Bitwise determinism across GPU types. Drift was measured instead (Section 5).
- V's base-model revision.
- Trainable-module histories of either recipe. Both changed every parameter group, so neither froze a component outright.
- Benchmark contents beyond the pinned digests. The suite was examined in earlier work, so no split is pristine.
