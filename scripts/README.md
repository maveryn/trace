# Repository Scripts

This directory contains source-checkout maintenance and release tools. Public
dataset commands that must work after installing `trace-tasks` live in the
package and are exposed as `trace-list`, `trace-generate`, `trace-validate`,
and `trace-export`.

Run these scripts from the repository root after installing the project. Use
`constraints/release.txt` when regenerating committed catalog or gallery
outputs:

```bash
python -m pip install -c constraints/release.txt -e ".[test,export]"
```

The separate `constraints/compat-py314.txt` file tests package and CLI support
on Python 3.14. Committed dataset and gallery outputs use
`constraints/release.txt`.

The tools are:

- `audit_text_legibility.py` audits rendered text against the public
  legibility policy.
- `build_release_dataset.py` builds or verifies the 64,000-row training and
  2,000-row IID validation dataset from the current source tree. It writes
  locally and supports planning and verification modes.
- `check_public_release.py` runs the source, package, installation, CLI, and
  repository-hygiene release gates. Its `--package-only` mode tests another
  supported Python runtime without regenerating committed artifacts.
- `check_review_recipe.py` validates the 1,000-task, 25,000-request review
  recipe and detects changes to its generator inputs.
- `generate_active_task_inventory.py` regenerates the registry-derived active
  task inventory.
- `generate_paper_domain_montage.py` validates the committed domain examples
  and Trace mark, then regenerates the 4-by-3 README montage and manifest.
- `generate_task_catalog.py` regenerates the public task catalog and its
  machine-readable manifest under `docs/task_catalog/`.
- `generate_release_gallery.py` regenerates the deterministic 11-domain
  gallery under `docs/assets/gallery/`, its hero montage, and the manifest at
  `docs/gallery/manifest.v1.json`.
- `check_skill_consistency.py` validates repo-local Codex skill metadata,
  cross-skill routing, documentation references, and public-path hygiene.

Contributor review commands are exposed through `trace-review`.

Scripts added here must use the `trace_tasks` namespace, accept
repository-relative or user-supplied paths, avoid credentials and uploads, and
provide a non-destructive check or dry-run mode where practical.
