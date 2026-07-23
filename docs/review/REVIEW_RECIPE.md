# Deterministic Review Recipe

`trace-review-recipe-v1` is the checked-in contributor-review input. It records
enough information to reconstruct review samples from a clean source checkout
while materialized outputs stay outside Git.

## Stored recipe

The recipe consists of one manifest and domain-sharded JSONL rows. It contains
exactly 25 query-stratified requests for each of the 1,000 public tasks: 25,000
rows total. For each task, every query variant is selected once before the
remaining positions are filled in stable round-robin order.

The accepted 25,000-row tree is about 14 MB and belongs in ordinary Git. Do not
move the recipe to Git LFS or external artifact storage. Only its
multi-gigabyte materialized replay is generated beneath the ignored local
`review/` workspace.

Each row records:

- `domain`, `scene_id`, `task_id`, `query_id`, and stable ordinal;
- accepted seed and sample cursor;
- exact public generation parameters;
- retry decision and maximum attempts;
- semantic, raw-pixel, and encoded-PNG hashes.

The manifest binds the rows to the source revision, task catalog, prompt and
resource hashes, generator contract, constraints profile, Python runtime, and
relevant native rendering-library versions. It must contain no credentials,
absolute paths, reviewer identity, feedback, or model-endpoint state.

The manifest retains a byte-exact generator-tree hash from capture.
Checkout freshness normally requires that exact hash. When it differs,
`check_review_recipe.py` compares the current generator files with the recorded
producer revision and accepts only Python formatting, comments, and docstrings whose
normalized executable AST is unchanged. Added or removed files, non-Python
changes, and any executable Python change still require a complete recipe
replacement. The producer revision must therefore be available in local Git
history for that comparison. If it is unavailable, replace the recipe from the
current source instead of bypassing the check.
Generator code must not derive outputs from `__doc__`, source text, source line
numbers, or Python symlink targets; the checker rejects changed symlinks and
treats the other surfaces as non-runtime contributor documentation.

## Determine the affected slice

Follow runtime dependencies, not only the task named in the change. A shared
helper or resource may expand a task change to a scene or domain. In particular,
prompt metadata records the content hash of the complete prompt-bundle file.
Changing one slot or template therefore changes the semantic identity of every
task that consumes that bundle, even when the rendered wording changes for only
one task. Resolve bundle consumers from domain/scene configs and confirm them in
generated prompt metadata before selecting `--task`, `--scene`, or `--domain`.

Use the resulting dependency-expanded slice for focused capture, materialize,
verify, audit, and manual inspection. Record why any apparently unchanged task
is included.

## Capture prerequisites

Capture only from a clean, committed `dev` checkout:

- commit the exact source and resource changes under review;
- require `git status --short` to be empty, including untracked files;
- use the same commit, constraints, and runtime for repeated captures.

Normal clones contain the history needed for freshness checks. In a shallow
clone, run `git fetch --unshallow` before `check_review_recipe.py`.

Do not capture a dirty working tree or treat an uncommitted capture as an
accepted recipe.

### Draft affected-slice capture

During development, capture the dependency-expanded slice twice beneath the
ignored `review/` workspace. For example, for a scene-scoped change:

```bash
trace-review capture \
  --recipe review/recipe-drafts/<change>-a \
  --requests-per-task 25 \
  --scene <domain>/<scene_id>
trace-review capture \
  --recipe review/recipe-drafts/<change>-b \
  --requests-per-task 25 \
  --scene <domain>/<scene_id>
```

Use task or domain filters when those match the dependency scope more precisely.
Accept a draft for review only when row ordering, seeds, retry decisions, hashes,
and manifest inputs agree between the two independent roots. A draft may contain
fewer than 1,000 tasks and is local review evidence only. Do not commit it or
splice its rows into an accepted recipe.

### Replace the complete accepted recipe

An accepted source or resource fingerprint change requires a complete recipe
replacement. The schema id remains `trace-review-recipe-v1`, and the path
remains `docs/review/recipes/trace-review-recipe-v1`. The manifest digest and
source revision identify the accepted state.

From the same clean, committed checkout, run two complete captures in
independent roots:

```bash
trace-review capture \
  --recipe review/canonical-candidates/replacement-a \
  --requests-per-task 25 \
  --all
trace-review capture \
  --recipe review/canonical-candidates/replacement-b \
  --requests-per-task 25 \
  --all
```

After the two roots and the reviewed affected slice agree, replace the complete
checked-in recipe with one candidate without changing its captured
contents. The promoted manifest must still cover all 1,000 tasks and 25,000
rows. Never hand-edit rows, hashes, or manifests, and never promote a filtered
draft as the accepted recipe.

## Materialize and verify

Generated state belongs under the ignored `review/` workspace. Materialize a
dependency-expanded slice during normal development, using either the matching
draft recipe or an accepted canonical recipe whose semantic contract is
unchanged:

```bash
trace-review materialize \
  --recipe <recipe-root> \
  --scene <domain>/<scene_id> \
  --output review/task-reviews
trace-review verify \
  --recipe <recipe-root> \
  --scene <domain>/<scene_id>
trace-review audit \
  --recipe <recipe-root> \
  --scene <domain>/<scene_id> \
  --output review/task-reviews
```

Capture and materialization filters select whole tasks by domain, scene, or
task. Verification and audit may additionally select a query. Materialization
publishes one complete task atomically so a later broader task/scene replay can
resume it safely; query-only partial task directories are deliberately not
supported. A full replay may occupy roughly 5.15 GiB locally, so it is an
offline curation gate rather than routine CI. Replay resumes verified task
directories and refuses to overwrite artifacts from another recipe. Move or
remove the old ignored output directory explicitly before materializing a
replacement.

## Intentional semantic changes

Prompt text and prompt metadata are part of the semantic payload. A reviewed
prompt wording change must therefore mismatch the older recipe; this means a
new digest and source revision are required. Capture the changed behavior into
a fresh draft, verify that draft, and promote it only through the complete
replacement process above.

For a prompt-only change, compare old and new rows explicitly. Prompt text,
prompt slots, bundle hashes, and semantic hashes may change within the expanded
consumer slice. Seeds, retry decisions, answers, annotations, verifier-relevant
execution data, raw pixels, and encoded PNGs should remain stable. Investigate
any broader delta before promotion.

## Rendering reproducibility

Semantic hashes cover contract-relevant generated values and are portable:
their mismatch is a verification failure. Pixel and PNG hashes capture a
reviewed rendering environment but can drift with Cairo, font, and native
library changes. Default verification reports this drift as an environment
warning when the semantic hash still agrees. Use `--strict-rendering` only in
an environment intended to reproduce the recorded native stack:

```bash
trace-review verify \
  --recipe docs/review/recipes/trace-review-recipe-v1 \
  --strict-rendering
```

Never add materialized images, JSON sidecars, workbooks, feedback SQLite files,
or calibration responses to Git or Git LFS.
