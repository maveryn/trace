# RLVR Training And Evaluation

Trace separates task generation from model training and external evaluation.
The `main` branch provides deterministic task generation, metadata-backed
verification, packaged resources, dataset validation, and export. The
[`rlvr` branch](https://github.com/maveryn/trace/tree/rlvr) provides the
Qwen2.5-VL 3B/7B training workflow, TRACE validation, and the 24-benchmark
`trace_eval_v1` external evaluation used in the TRACE paper.

## Training Contract

The two reproducible training profiles are:

- [`trace-qwen2.5-vl-3b`](https://github.com/maveryn/trace/blob/rlvr/rlvr/configs/trace-qwen2.5-vl-3b.yaml),
  based on `Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`;
- [`trace-qwen2.5-vl-7b`](https://github.com/maveryn/trace/blob/rlvr/rlvr/configs/trace-qwen2.5-vl-7b.yaml),
  based on `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`.

Both profiles use
[`maveryn/trace@dataset-v1`](https://huggingface.co/datasets/maveryn/trace/tree/dataset-v1),
select `prompt_answer`, score `answer_gt`, run GRPO for 500 steps, and verify
the training prompt with SHA-256
`f394927d9abcfb7a1e43ef48a30c29b8c70e6facdbda314d7b27c59d8c3ae900`.
The machine-facing configurations pin the tag's immutable commit. A checked-in
equivalence receipt verifies that every model-consumed value, image byte, and
row position matches the historical training input. The optional
`trace_supervision_mode` column is advisory and is ignored by these profiles.
Environment setup, input preparation, smoke runs, full runs, checkpoint merge,
and receipt validation are documented in the
[`rlvr` guide](https://github.com/maveryn/trace/blob/rlvr/rlvr/README.md).

## Verification Boundaries

Native Trace rewards score typed answers and optional annotations against the
metadata contract emitted by the task generator. Those APIs are documented in
[Verifier contracts](../VERIFIERS.md) and covered by the package tests.

External benchmarks retain their official prompt, parsing, and scoring
contracts from pinned evaluator revisions. The `trace_eval_v1` workflow records
every narrowly scoped adapter or post-run reproducibility patch. It does not
substitute a native Trace verifier for an official benchmark scorer.

## Canonical External Evaluation

The evaluation covers 24 benchmarks for:

- Qwen2.5-VL-3B base and TRACE;
- Qwen2.5-VL-7B base, TRACE, VERO, Game-RL, Sphinx, and PCGRPO;
- decoding seeds `42`, `43`, and `44`.

The generated
[`results.json`](https://github.com/maveryn/trace/blob/rlvr/rlvr/evaluation/trace_eval/results.json)
is the machine-readable result authority. It retains all 576
model/seed/benchmark scores and binds each aggregate to immutable run artifacts.
The adjacent
[`suite manifest`](https://github.com/maveryn/trace/blob/rlvr/rlvr/evaluation/trace_eval/suite.v1.json),
[`benchmark provenance`](https://github.com/maveryn/trace/blob/rlvr/rlvr/evaluation/trace_eval/benchmark_provenance.v1.json),
and
[`evaluation guide`](https://github.com/maveryn/trace/blob/rlvr/rlvr/evaluation/trace_eval/README.md)
define benchmark sources, revisions, splits, licenses, generation settings,
answer extraction, scoring, retry behavior, aggregation, and artifact checks.

Validate the frozen training inputs and evaluation metadata with:

```bash
python rlvr/train.py check --config trace-qwen2.5-vl-3b
python rlvr/train.py check --config trace-qwen2.5-vl-7b
python rlvr/evaluation/scripts/validate_release_inputs.py
```

Use immutable dataset, model, evaluator, judge, prompt, and result revisions in
every run receipt. Do not replace them with mutable branch references or
machine-specific paths.

## TRACE Validation

The paper also evaluates the same eight checkpoints on two unseen instances
from each of TRACE's 1,000 task programs. This 2,000-instance, seed-42 campaign
uses its own frozen suite, preparation, generation, extraction, scoring, and
verification interfaces under
[`rlvr/evaluation/trace_validation`](https://github.com/maveryn/trace/tree/rlvr/rlvr/evaluation/trace_validation).
It measures new realizations within the TRACE task distributions and is not a
held-out-task or external-transfer benchmark.

The checked-in result source contains the eight aggregate scores. The
corresponding response and score artifacts are pinned at
[`maveryn/trace-eval-runs@cf0d14aed86db2661d397ce8b68b36171873478d`](https://huggingface.co/datasets/maveryn/trace-eval-runs/tree/cf0d14aed86db2661d397ce8b68b36171873478d/runs/trace-iid-validation-2000-answer-seed42-8models-v1).

## Test Boundary

The `main` branch tests task generation, schemas, verifiers, installation, CLI
behavior, manifests, and repository hygiene. The `rlvr` branch adds focused
training-profile and evaluator validation. Model-specific compatibility checks
belong beside the affected launcher when a reported result depends on them.
