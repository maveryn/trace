# Contributing To Trace

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c constraints/release.txt -e ".[test,export,review]"
```

Use `constraints/compat-py314.txt` instead when developing on CPython 3.14.
That compatibility environment updates NumPy and SciPy to versions with 3.14
wheels. Use `constraints/release.txt` on CPython 3.10-3.12 when regenerating
committed dataset or gallery artifacts. Optional export dependencies are pinned
by both files; native system libraries are not.

## Branch Ownership

- `main` owns public task generation, contracts, packaged resources, verifiers,
  and dataset export.
- `dev` integrates public contributor tooling, local review workflows, and
  repo-local Codex skills before reviewed promotion to `main`.
- `rlvr` owns the Qwen2.5-VL 3B/7B training recipes and the 24-benchmark
  `trace_eval_v1` evaluation workflow.

Develop general public contributions on `dev`. Promote reviewed stable changes
to `main`, then incorporate relevant generation changes into `rlvr`
when they affect training or evaluation.

Documentation source and `mkdocs.yml` follow the same promotion path. GitHub
Pages deployments are restricted to `main`; documentation checks on `dev` and
`rlvr` never publish a site.

## Source Ownership

- Task code: `src/trace_tasks/tasks/<domain>/<scene_id>/<objective_contract>.py`
- Prompt bundles: `src/trace_tasks/resources/prompts/<domain>/...`
- Domain and scene defaults: `src/trace_tasks/resources/configs/domains/<domain>/...`
- Task documentation: `docs/tasks/<domain>/<scene_id>/...`
- Shared infrastructure: `src/trace_tasks/core/` and the narrowest applicable
  `shared/` package

Keep task generation deterministic for fixed seeds and versions. Prompts must
remain external assets, answers and annotations must share one execution trace,
and verifiers must use metadata rather than pixels as ground truth.

## Checks

Run the focused public smoke tests first:

```bash
pytest -q tests/test_public_smoke.py
```

Run the complete public release check before proposing a release:

```bash
python scripts/check_public_release.py
```

Use `--source-only` for the registry, representative generation, manifest, and
repository-hygiene checks without rebuilding and installing the package.

Then run tests covering the modules changed by the contribution. Changes to
task behavior, contracts, prompts, or shared helpers must update the
corresponding documentation in the same commit.

For task-facing changes, materialize a focused slice of the canonical review
recipe and inspect it locally:

```bash
trace-review materialize \
  --recipe docs/review/recipes/trace-review-recipe-v1 \
  --task <task_id> \
  --output review/task-reviews
trace-review verify \
  --recipe docs/review/recipes/trace-review-recipe-v1 \
  --task <task_id>
trace-review audit \
  --recipe docs/review/recipes/trace-review-recipe-v1 \
  --task <task_id> \
  --output review/task-reviews
trace-review serve --review-root review/task-reviews
```

The recipe is committed; generated images, sidecars, workbooks, calibration
responses, and feedback databases are not. See
[Contributor review](docs/review/README.md) and
[Codex skills](docs/workflows/CODEX_SKILLS.md).

## Documentation Site

Install the pinned site dependencies, run the local preview, and require a
strict production build before proposing documentation changes:

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve --strict
mkdocs build --strict --site-dir site
```

The generated `site/` directory is ignored local state and must not be
committed. Pull requests targeting `main` run the same strict build; pushes to
`main` publish the resulting artifact to GitHub Pages after enforcing the
100 MiB site-size limit. The workflow can also be run manually on `main` for
the first deployment after Pages is enabled.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the complete source and
validation workflow.
