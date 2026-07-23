# Documentation Maintenance

Documentation changes must accompany changes to public contracts, source
layout, task behavior, prompts, resources, or validation. Keep links relative
when their targets also live under `docs/`; use canonical public URLs for
repository files outside that tree. Avoid machine-specific paths, and ensure
the active task inventory remains generated from the current registry.

## Site Ownership

- Markdown and site assets live under `docs/`; navigation, theme, and build
  policy live in the root `mkdocs.yml`.
- Develop documentation changes on `dev` and promote reviewed content to
  `main`. Only `main` may deploy GitHub Pages; the workflow also supports a
  manual `main` run for the first deployment after Pages is enabled.
- `site/` is generated, ignored local output. Do not commit it or copy it into
  the source distribution.
- `docs/research/README.md` owns the research artifact, released-model, result,
  licensing, and citation hub. Keep its marker-delimited
  24-benchmark Base/TRACE table byte-identical in `README.md`,
  `docs/README.md`, and the research page. `tests/test_docs_site.py` checks all
  three presentations and, on `rlvr`, checks every displayed value against the
  complete machine-readable result source.
- `docs/research/REPRODUCING_RESULTS.md` owns the public training-smoke and
  external-evaluation walkthrough. Keep its commands synchronized with the
  released `rlvr` interfaces and verify every example in configuration-only or
  `--help` mode when possible.
- Use branch-stable `main` or `rlvr` URLs for files in `maveryn/trace`; do not
  publish commit-pinned Trace URLs. Keep Hugging Face dataset and model links
  pinned to full immutable revisions, and preserve reproducibility claims with
  recorded producer identities, receipts, and content hashes.
- `paper.pdf` is the repository manuscript artifact. When replacing it, update
  the expected hash and size in `tests/test_public_paper.py` and keep the README
  and site links current.
- Global social preview metadata lives in `docs/overrides/main.html`. It uses
  each canonical page's URL and the 1512 x 780 paper-domain montage; keep the
  Open Graph and Twitter image metadata synchronized if that asset changes.
- Do not add placeholder publication links or present exploratory model
  ablations as released research artifacts.

## Local Site Validation

Install the pinned documentation environment and use strict mode for both the
preview and production build:

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve --strict
mkdocs build --strict --site-dir site
```

The strict build must succeed from the repository root. The generated site
must remain smaller than 100 MiB so the CI artifact and Pages deployment stay
bounded.
