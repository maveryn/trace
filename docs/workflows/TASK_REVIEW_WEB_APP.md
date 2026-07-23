# Local Task Review Application

The task review application is an optional local contributor tool for browsing
materialized recipe artifacts and recording issues. It is not a production
multi-user service and does not generate samples from browser requests.

## Setup

```bash
python -m pip install -c constraints/release.txt -e ".[test,review]"
trace-review materialize \
  --recipe docs/review/recipes/trace-review-recipe-v1 \
  --task <task_id> \
  --output review/task-reviews
trace-review serve --review-root review/task-reviews
```

Open the printed loopback URL. The index discovers only public tasks and local
artifacts under the selected root. Missing rows appear as `not materialized`
with the corresponding CLI command; the app never launches generation or
calibration subprocesses.

## Review surfaces

Domain, scene, task, and query pages expose prompt variants, images, typed
answers, annotation overlays, sanitized trace/verifier summaries, provenance,
and any contributor-supplied distribution diagnostics. Separate resource
pages cover public illustration objects and 3D object profiles; font and icon
legibility is reviewed through the task images that use them.

Use issue fields for prompt, rendering, answer, annotation, verifier/trace,
distribution, code/docs, or taxonomy findings. Keep sample-specific issues on
the stable recipe identity and systematic findings at task or scene level.
Repair notes state what changed and which validation ran; a human reviewer
retains responsibility for resolving an issue unless explicit instructions say
otherwise.

Live state defaults to `review/feedback/review_feedback.sqlite`; exports and
workbooks also stay under `review/`. None of this state is a source-of-truth
contract or a file to commit.

## Network safety

The default bind is loopback. A non-loopback bind requires
`TRACE_REVIEW_APP_TOKEN`, supplied through the environment and entered through
the login form. Also pass each hostname or IP that reviewers will use as an
explicit `--trusted-host`; wildcard hosts are not enabled. For example:

```bash
export TRACE_REVIEW_APP_TOKEN="<review credential>"
trace-review serve \
  --review-root review/task-reviews \
  --host 0.0.0.0 \
  --trusted-host review-host.example.org
```

Do not place tokens in a URL or command argument. Media routes require the same
session, serve only indexed files beneath the resolved review root, and reject
traversal and symlink escapes.

Do not expose the development server directly to the internet. Use an
authenticated, TLS-terminating proxy if a team needs remote access, and keep
the feedback database on a trusted filesystem.

## Refresh and export

Use the app's reload action after materializing or replacing artifacts. Restart
after Python, template, static-resource, or database-schema changes.

```bash
trace-review export \
  --review-root review/task-reviews \
  --output review/exports/task-review.jsonl \
  --format jsonl
```

Exports are portable review reports, not approval gates. Optional calibration
appears as informational data when present and as `not run` otherwise.
