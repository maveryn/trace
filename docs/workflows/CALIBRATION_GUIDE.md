# Portable Endpoint Calibration

Calibration is an optional contributor probe of materialized Trace prompts
against a caller-managed OpenAI-compatible HTTP endpoint. It reports model
behavior alongside deterministic and manual review.

## Run

Start and secure the endpoint outside Trace, then provide its public model name
and URL explicitly. Put credentials in `TRACE_REVIEW_API_KEY`; never place a
secret in a command argument or recipe.

```bash
export TRACE_REVIEW_API_KEY="<endpoint credential>"
trace-review calibrate \
  --review-root review/task-reviews \
  --endpoint http://127.0.0.1:8000/v1 \
  --model <model-id> \
  --task <task_id> \
  --limit 25 \
  --rollouts 1 \
  --max-retries 2 \
  --max-tokens 512 \
  --output review/calibration/<run-name>.json
```

The runner uses answer prompts, extracts the public typed answer, and
scores it with the same public Trace answer contract used by verification.
Configure sample count, rollout count, response cap, timeout, retry budget, and
optional diagnostic thresholds explicitly for each run. Missing thresholds are
reported as unspecified.

## Provenance and safety

The runner requires the selected materialized rows to carry one canonical
recipe id and digest before it contacts the endpoint. It records that recipe
identity, selected rows, model string, endpoint origin without credentials,
generation parameters, response cap, retries, scorer version, and aggregate
results. It does not revalidate the current source checkout; run
`trace-review verify` first. Never record request authorization headers or
environment values. Raw responses and derived summaries must stay under the
ignored `review/` workspace.

An absent API key is valid for an endpoint that permits anonymous or local
access; a required but missing key surfaces as the endpoint's authorization
failure. Network failure, redirects, malformed responses, unsupported media
input, and truncation are explicit per-request or run outcomes. A calibration
command creates one new atomic report and does not resume an older report.

Calibration results are model- and endpoint-dependent diagnostics. A result
may motivate prompt or rendering investigation, but it cannot approve a task,
change verifier ground truth, or define a repository-wide solve-rate threshold.
