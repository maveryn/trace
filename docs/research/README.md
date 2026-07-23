---
description: Paper, models, dataset, evaluation results, and reproduction guides for Trace.
---

# Research artifacts

This page collects the paper, released checkpoints, training dataset, TRACE
validation results, and `trace_eval_v1` external evaluation. The
[Trace Hugging Face collection](https://huggingface.co/collections/maveryn/trace-6a604291b4be4ed6399b9f24)
groups the dataset, both checkpoints, and evaluation runs.

## Paper

[Read the Trace paper](https://arxiv.org/abs/2607.19790){ .md-button .md-button--primary }

## Released models

Both checkpoints were trained on the 64,000-example
[`maveryn/trace@dataset-v1`](https://huggingface.co/datasets/maveryn/trace/tree/dataset-v1)
training split.

| Released checkpoint | Base checkpoint |
| --- | --- |
| [TRACE Qwen2.5-VL 3B](https://huggingface.co/maveryn/trace-qwen2.5-vl-3b) | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| [TRACE Qwen2.5-VL 7B](https://huggingface.co/maveryn/trace-qwen2.5-vl-7b) | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

Training configurations are available on the
[`rlvr` branch](https://github.com/maveryn/trace/tree/rlvr/rlvr).

## Artifacts

| Artifact | Contents | Reference |
| --- | --- | --- |
| TRACE dataset | 64,000 training and 2,000 validation examples with typed targets, verifier payloads, and execution-trace references | [`dataset-v1`](https://huggingface.co/datasets/maveryn/trace/tree/dataset-v1) |
| Evaluation run archive | Responses, extractions, scores, and receipts for `trace_eval_v1` | [`trace-eval-runs`](https://huggingface.co/datasets/maveryn/trace-eval-runs) |
| TRACE validation run | Responses, extraction outputs, scores, and metadata for the 2,000-instance validation campaign | [TRACE validation artifacts](https://huggingface.co/datasets/maveryn/trace-eval-runs/tree/main/runs/trace-iid-validation-2000-answer-seed42-8models-v1) |
| RLVR workflows | Training profiles, evaluation interfaces, result source, benchmark metadata, and compatibility notes | [`rlvr/`](https://github.com/maveryn/trace/tree/rlvr/rlvr) |

## TRACE validation

The validation set contains two previously unseen instances from each of the
1,000 task programs used to generate the training set. These results measure
generalization to new semantic and visual realizations within the TRACE task
distributions; they are separate from the external-transfer results below.
Every model was evaluated once with decoding seed 42, and every percentage
uses all 2,000 rows as its denominator.

| Scale | Model | Accuracy | Change from base |
| --- | --- | ---: | ---: |
| 3B | Qwen2.5-VL-3B Base | 24.45 | — |
| 3B | TRACE Qwen2.5-VL 3B | **41.05** | **+16.60** |
| 7B | Qwen2.5-VL-7B Base | 34.25 | — |
| 7B | TRACE Qwen2.5-VL 7B | **51.55** | **+17.30** |

The same evaluation includes VERO, Game-RL, Sphinx, and PCGRPO at 7B. Responses
and scores for all eight models are available in the evaluation run archive.

## External benchmark results

`trace_eval_v1` evaluates 32,805 examples per model and decoding seed. The
overall score is the unweighted macro mean of 24 benchmark percentages. Values
below are mean ± sample standard deviation across seeds 42, 43, and 44; paired
changes compare each trained model with the same-scale Qwen2.5-VL base on
matched benchmark and seed results.

| Scale | Model | Post-training data | Overall score | Paired change |
| --- | --- | --- | ---: | ---: |
| 3B | [Qwen2.5-VL-3B Base](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3) | — | 39.34 ± 0.63 | — |
| 3B | [TRACE Qwen2.5-VL 3B](https://huggingface.co/maveryn/trace-qwen2.5-vl-3b/tree/2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5) | TRACE (synthetic) | **42.85 ± 0.39** | **+3.51 ± 0.25** |
| 7B | [Qwen2.5-VL-7B Base](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5) | — | 47.93 ± 0.30 | — |
| 7B | [TRACE Qwen2.5-VL 7B](https://huggingface.co/maveryn/trace-qwen2.5-vl-7b/tree/4d0f1ae8ee25022058090dbdbff61957ece7331d) | TRACE (synthetic) | **51.99 ± 0.17** | **+4.06 ± 0.41** |
| 7B | [VERO Qwen2.5-VL 7B](https://huggingface.co/zlab-princeton/Vero-Qwen25-7B/tree/180e84be5acb2aa887cf51015b84b6a6e453ee90) | VERO (real-image) | 52.22 ± 0.07 | +4.30 ± 0.26 |
| 7B | [Game-RL Qwen2.5-VL 7B](https://huggingface.co/OpenMOSS-Team/Game-RL-Qwen2.5-VL-7B/tree/205b5934ce70504cfd6ae26b16f705d0b98b9306) | Game-RL (synthetic) | 48.05 ± 0.45 | +0.12 ± 0.18 |
| 7B | [Sphinx Qwen2.5-VL 7B](https://huggingface.co/xashru/sphinx_qwen7b_500/tree/6ffefb03d5cb0767683bfb42a084ea86b707ef9a) | Sphinx (synthetic) | 49.34 ± 0.15 | +1.41 ± 0.17 |
| 7B | [PCGRPO Qwen2.5-VL 7B](https://huggingface.co/armenjeddi/PCGRPO-Qwen2.5-VL-7B-Jigsaw-with-curriculum-with-grpo-care/tree/921bbced4176f5d362e98c843a57656c5d78dad7) | PCGRPO (synthetic) | 48.80 ± 0.28 | +0.87 ± 0.46 |

### Base versus TRACE by benchmark

The table below reports every benchmark for the matched 3B and 7B
Base-versus-TRACE comparisons. Model scores are mean ± sample standard
deviation across seeds 42, 43, and 44, in percent. The parenthesized Δ is the
mean seed-paired TRACE-minus-Base change in percentage points; paired
dispersion remains available in the full-precision result source.

<!-- trace-eval-v1-base-trace-table:start -->
| Benchmark | 3B Base | 3B TRACE (Δ) | 7B Base | 7B TRACE (Δ) |
| --- | ---: | ---: | ---: | ---: |
| **Charts & Tables** |  |  |  |  |
| ChartQAPro | 31.57 ± 0.66 | 31.43 ± 1.33 (-0.14) | 45.81 ± 0.24 | 48.05 ± 0.40 (+2.24) |
| CharXivReason | 28.90 ± 1.11 | 34.67 ± 1.47 (+5.77) | 39.73 ± 1.17 | 47.13 ± 0.35 (+7.40) |
| TableVQABench | 69.27 ± 0.91 | 71.99 ± 0.28 (+2.72) | 75.20 ± 0.90 | 78.31 ± 0.17 (+3.11) |
| EvoChart | 48.51 ± 0.45 | 46.83 ± 1.36 (-1.68) | 57.07 ± 0.72 | 64.91 ± 0.05 (+7.84) |
| **Visual Math** |  |  |  |  |
| MathVision | 19.25 ± 0.65 | 25.35 ± 0.86 (+6.10) | 24.81 ± 0.75 | 27.42 ± 0.62 (+2.61) |
| MathVista | 58.13 ± 3.74 | 64.43 ± 2.12 (+6.30) | 68.67 ± 0.40 | 73.37 ± 0.45 (+4.70) |
| MathVerse | 33.59 ± 1.95 | 40.02 ± 1.52 (+6.43) | 43.44 ± 0.92 | 47.76 ± 0.51 (+4.31) |
| WeMath | 17.87 ± 1.24 | 28.82 ± 0.40 (+10.95) | 35.20 ± 2.92 | 46.16 ± 1.32 (+10.96) |
| **Science & General** |  |  |  |  |
| PhyX mini MC | 32.80 ± 9.96 | 37.47 ± 5.58 (+4.67) | 40.97 ± 3.57 | 48.70 ± 0.82 (+7.73) |
| MMMU-ProVis | 26.59 ± 0.32 | 31.16 ± 1.20 (+4.57) | 35.70 ± 0.39 | 39.36 ± 0.71 (+3.66) |
| RealWorldQA | 60.35 ± 0.42 | 62.14 ± 1.18 (+1.79) | 65.45 ± 0.87 | 68.50 ± 0.68 (+3.05) |
| MMStar | 52.27 ± 0.52 | 55.24 ± 1.02 (+2.98) | 61.89 ± 0.87 | 65.64 ± 0.34 (+3.76) |
| **Spatial Reasoning** |  |  |  |  |
| EmbSpatial | 59.07 ± 1.04 | 60.88 ± 1.13 (+1.81) | 69.42 ± 0.90 | 70.95 ± 0.69 (+1.53) |
| SpatialVizBench COT | 30.08 ± 1.22 | 31.84 ± 1.52 (+1.75) | 35.14 ± 0.10 | 35.68 ± 0.34 (+0.54) |
| CV-Bench 3D | 58.67 ± 8.32 | 66.97 ± 3.68 (+8.31) | 76.25 ± 1.96 | 81.00 ± 0.43 (+4.75) |
| ERQA | 35.33 ± 0.80 | 36.42 ± 0.52 (+1.08) | 39.00 ± 1.56 | 41.17 ± 1.28 (+2.17) |
| **Perception & Counting** |  |  |  |  |
| BLINK | 44.52 ± 2.10 | 47.13 ± 0.70 (+2.61) | 53.31 ± 1.23 | 56.57 ± 0.73 (+3.26) |
| CountBenchQA | 65.43 ± 1.59 | 68.65 ± 0.83 (+3.22) | 82.14 ± 1.48 | 84.80 ± 1.44 (+2.67) |
| CountQA | 14.88 ± 1.80 | 15.64 ± 0.57 (+0.76) | 19.59 ± 1.17 | 22.51 ± 0.91 (+2.92) |
| TreeBench | 39.26 ± 1.96 | 38.60 ± 1.17 (-0.66) | 38.02 ± 1.37 | 40.25 ± 2.11 (+2.22) |
| **Puzzles & Logic** |  |  |  |  |
| PuzzleVQA | 32.55 ± 1.21 | 39.13 ± 1.13 (+6.58) | 44.57 ± 0.64 | 50.02 ± 0.50 (+5.45) |
| VisualPuzzles | 26.20 ± 2.74 | 28.51 ± 0.70 (+2.31) | 31.08 ± 0.15 | 34.02 ± 0.77 (+2.94) |
| LogicVista | 36.47 ± 1.25 | 40.12 ± 1.71 (+3.65) | 42.65 ± 2.24 | 47.35 ± 0.68 (+4.70) |
| MME-Reasoning | 22.62 ± 1.81 | 25.06 ± 1.53 (+2.44) | 25.17 ± 1.18 | 28.03 ± 0.89 (+2.86) |
<!-- trace-eval-v1-base-trace-table:end -->

Machine-readable scores, benchmark metadata, and verification receipts are
available in the
[`trace_eval_v1` directory](https://github.com/maveryn/trace/tree/rlvr/rlvr/evaluation/trace_eval).

## Benchmark index

| Category | Benchmarks |
| --- | --- |
| Charts & Tables | ChartQAPro, CharXivReason, TableVQABench, EvoChart |
| Visual Math | MathVision, MathVista, MathVerse, WeMath |
| Science & General | PhyX mini MC, MMMU-ProVis, RealWorldQA, MMStar |
| Spatial Reasoning | EmbSpatial, SpatialVizBench COT, CV-Bench 3D, ERQA |
| Perception & Counting | BLINK, CountBenchQA, CountQA, TreeBench |
| Puzzles & Logic | PuzzleVQA, VisualPuzzles, LogicVista, MME-Reasoning |

The suite manifest defines order, row counts, generation settings, and scoring
routes. The provenance matrix supplies the source revision, split, license or
terms, citation, prompt route, scorer, and scoped adapter for every benchmark.

## Reproduce the results

[Open the reproduction guide](REPRODUCING_RESULTS.md){ .md-button .md-button--primary }

The guide covers environment setup, a one-step training check, TRACE
validation, and the complete `trace_eval_v1` workflow.

## Licensing

- Trace source code and documentation use
  [Apache-2.0](https://github.com/maveryn/trace/blob/main/LICENSE). The paper
  PDF has no separate public license declaration; the source-code license does
  not automatically license the paper, model checkpoints, datasets, or
  third-party evaluation sources.
- The TRACE dataset uses
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The 3B checkpoint remains subject to the upstream
  [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE);
  the 7B checkpoint uses
  [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/LICENSE).
- The `trace-eval-runs` dataset card declares `license: other`. Benchmark
  inputs, generated responses, and metrics remain subject to their benchmark,
  source-model, and per-source terms recorded in the provenance matrix.
  Benchmark datasets are not redistributed in the evaluation archive.
- External comparison checkpoints retain the licenses of their linked model
  repositories. Third-party assets used by Trace are listed in
  [`THIRD_PARTY_NOTICES.md`](https://github.com/maveryn/trace/blob/main/THIRD_PARTY_NOTICES.md).

## Citation

Repository citation metadata is maintained in
[`CITATION.cff`](https://github.com/maveryn/trace/blob/main/CITATION.cff).
GitHub's **Cite this repository** control can export the same metadata in
common bibliography formats.
