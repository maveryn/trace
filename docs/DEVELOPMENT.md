# Development

## Source Layout

- `src/trace_tasks/core/`: generation, identity, validation, trace, reward, and export infrastructure
- `src/trace_tasks/tasks/<domain>/<scene_id>/`: public task implementations
- `src/trace_tasks/tasks/<domain>/shared/`: domain-level reusable helpers
- `src/trace_tasks/resources/configs/domains/`: generation and rendering defaults
- `src/trace_tasks/resources/prompts/`: external prompt bundles
- `src/trace_tasks/resources/assets/`: deterministic runtime resources
- `tests/`: public contract and task regression coverage

## Branches

- `main` contains task generation, contracts, runtime resources, verifiers, and
  export functionality.
- `dev` integrates public contributor tooling, local review workflows, and
  repo-local Codex skills before reviewed promotion to `main`.
- `rlvr` contains the Qwen2.5-VL 3B/7B training recipes and the 24-benchmark
  `trace_eval_v1` evaluation workflow.

Develop general public contributions on `dev`. Promote reviewed stable changes
to `main`, then deliberately incorporate relevant generation changes into
`rlvr`.

The release boundary is defined in
[RLVR release scope](workflows/RLVR_RELEASE_SCOPE.md). Runnable paper training
and evaluation documentation lives on the
[RLVR branch](https://github.com/maveryn/trace/tree/rlvr/rlvr).

## Reproducible Environment

Project metadata supports CPython 3.10-3.14. Committed dataset and gallery
outputs use the exact Python package versions from `constraints/release.txt` on
CPython 3.10-3.12:

```bash
python -m pip install -c constraints/release.txt -e ".[test]"
```

CPython 3.14 package and CLI compatibility uses separate constraints because
the release NumPy and SciPy versions do not publish Python 3.14 wheels:

```bash
python -m pip install -c constraints/compat-py314.txt -e ".[test]"
```

Update a pin only after the source and package checks pass on every affected
Python version. Each file constrains direct and determinism-critical packages;
neither is a complete transitive lock. CI uses Ubuntu 24.04 as its host
baseline, and native Cairo and other system libraries are not locked by these
files. Optional export dependencies are constrained when the `export` extra is
installed; they are not installed for base runtime or source-smoke jobs.

The Python 3.14 constraints validate runtime compatibility only. Regenerate
committed dataset and gallery artifacts on CPython 3.10-3.12 with
`constraints/release.txt`.

## Required Properties

- Fixed seeds, parameters, and versions produce deterministic outputs.
- Answers and annotations come from the same execution trace.
- Final answers are unique by construction.
- Verifiers use metadata contracts rather than pixels as ground truth.
- Prompt text lives in external prompt assets.
- Task ids follow the public taxonomy contract.
- Shared helpers live at the narrowest layer that genuinely reuses them.

## Validation

Run focused tests for the changed task or shared module. Before a release,
validate the registry, generate representative tasks from every domain, build
and export a dataset, scan the repository for secrets and machine paths, and
verify task/resource behavior from a clean clone.

CI separates runtime compatibility from artifact regeneration:

- focused runtime tests cover CPython 3.10 through 3.13;
- CPython 3.14 uses its compatibility constraints and an installed-package/CLI
  check;
- source, catalog, gallery, and representative-generation checks run once on
  CPython 3.12;
- the CPython 3.12 package/CLI check runs in parallel with source checks and uses
  `--package-only`, so it does not regenerate the same release artifacts.

Safe prose-only changes to the top-level guides, development/workflow guides,
and README files run the lightweight documentation-hygiene workflow instead.
Generated catalogs, task documentation, gallery assets, code, configs, and
packaged resources continue to run the full public CI workflow.

Together the CPython 3.12 source-only and package-only jobs cover the same
release surfaces as `python scripts/check_public_release.py`. Run that combined
command locally when a single end-to-end release gate is useful.

Detailed source ownership and code-review rules are in
[Code review guidelines](workflows/CODE_REVIEW_GUIDELINES.md).
