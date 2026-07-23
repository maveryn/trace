# Canonical TRACE paper release results

[`results.json`](results.json) is the single machine-readable result source for
the paper's RLVR comparison. It contains all three decoding seeds
for exactly these models:

- Qwen2.5-VL-3B base and TRACE;
- Qwen2.5-VL-7B base, TRACE, VERO, Game-RL, Sphinx, and PCGRPO.

The file is generated, not hand-maintained. It binds every input result file
to its immutable Hugging Face repository revision and SHA-256, retains all
576 model/seed/benchmark scores, and recomputes benchmark, category, overall,
and paired-delta summaries. The overall score is the unweighted macro mean of
the 24 benchmark scores for the same model and decoding seed. Reported
standard deviations are sample standard deviations across seeds 42, 43, and
44.

## Immutable inputs

The 3B base/TRACE and 7B base/TRACE/VERO inputs come from the immutable
[`4178a839…` dataset revision](https://huggingface.co/datasets/maveryn/trace-eval-runs/tree/4178a839b689babe16f8ac36f0de7b1b2c5ef36c):

```text
maveryn/trace-eval-runs@4178a839b689babe16f8ac36f0de7b1b2c5ef36c
```

The Game-RL/Sphinx/PCGRPO inputs come from the immutable
[`4ca25af7…` dataset revision](https://huggingface.co/datasets/maveryn/trace-eval-runs/tree/4ca25af7a4d7daa644e6f35e070dbed1af078321):

```text
maveryn/trace-eval-runs@4ca25af7a4d7daa644e6f35e070dbed1af078321
```

The exact run ids and result/manifest hashes are recorded under
`source_artifacts` in `results.json`. The builder verifies every metadata file
listed by each source manifest before it reads a score.

## Rebuilding and validation

Place immutable snapshots in the input layout documented by
[`build_release_results.py`](../scripts/build_release_results.py),
then run:

```bash
python rlvr/evaluation/scripts/build_release_results.py --inputs-root <snapshot-root>
python rlvr/evaluation/scripts/build_release_results.py \
  --inputs-root <snapshot-root> --check
python rlvr/evaluation/scripts/validate_release_inputs.py
```

The final command validates the result identities and aggregates against the
suite, validates all 24 provenance rows, verifies the source map, and confirms
that only the suite-defined result sources are inputs.
