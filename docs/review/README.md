# Contributor Review

The contributor review workflow turns checked-in recipes into local,
inspectable outputs. Git stores the recipe and review policy; generated images,
workbooks, endpoint responses, and feedback databases stay under the ignored
`review/` workspace.

## Review sequence

1. [Capture or replay the canonical recipe](REVIEW_RECIPE.md).
2. [Review prompts](PROMPT_REVIEW.md).
3. [Review annotations](ANNOTATION_REVIEW.md).
4. [Review rendering](RENDERER_REVIEW.md).
5. [Review reusable assets](REUSABLE_ASSET_REVIEW.md) when shared visual
   resources changed.
6. [Review the domain as a coherent surface](DOMAIN_QUALITY_REVIEW.md).
7. Use the [local review application](../workflows/TASK_REVIEW_WEB_APP.md) to
   record issues and export a report.
8. Optionally run [portable endpoint calibration](../workflows/CALIBRATION_GUIDE.md)
   as an informational model probe.

The normative answer, annotation, verifier, prompt, taxonomy, and source-layout
rules remain under `docs/contracts/`. Review guides explain how to inspect those
contracts; they do not redefine them.
