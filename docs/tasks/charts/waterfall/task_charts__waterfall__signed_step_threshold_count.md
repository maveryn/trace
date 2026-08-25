# `task_charts__waterfall__signed_step_threshold_count`

## Contract
1. Domain: `charts`
2. Scene id: `waterfall`
3. Source implementation: `src/trace_tasks/tasks/charts/waterfall/signed_step_threshold_count.py`
4. Public query ids: `decrease_step_threshold_count`, `increase_step_threshold_count`
5. Prompt query key: `signed_step_threshold_count`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.waterfall.signed_step_threshold_count.ChartsWaterfallSignedStepThresholdCountTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/waterfall/charts_waterfall_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. The annotation marks each full contribution bar whose signed delta matches the requested direction and whose magnitude exceeds the prompt-bound threshold.
4. The start and summary final bars are not candidates because they carry no signed contribution.

## Program Contract

Program: `count(filter(contribution_steps, sign(delta) = direction and abs(delta) > threshold)); direction={negative,positive}; output=integer_count; annotation=bbox_set(matching_contribution_bars); scene=waterfall; scope=signed_step_threshold_count`

Candidate set: the visible signed contribution bars inside the `signed_step_threshold_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(matching_contribution_bars)`. The annotation marks each qualifying contribution bar. The start and summary final bars are not candidates because they carry no signed contribution.
Query ids: `decrease_step_threshold_count`, `increase_step_threshold_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program contract | Answer schema | Annotation schema |
|---|---|---|---|
| `decrease_step_threshold_count` | `count(filter(contribution_steps, delta < 0 and abs(delta) > threshold)); output=integer_count; annotation=bbox_set(matching_contribution_bars); scene=waterfall; scope=signed_step_threshold_count` | `integer_count` | `bbox_set` |
| `increase_step_threshold_count` | `count(filter(contribution_steps, delta > 0 and delta > threshold)); output=integer_count; annotation=bbox_set(matching_contribution_bars); scene=waterfall; scope=signed_step_threshold_count` | `integer_count` | `bbox_set` |

## Generation Notes
1. The target answer is drawn from a fixed support before the step count, so every
   answer stays equally likely; the step count is then widened to fit the drawn
   target.
2. Every instance carries at least one same-signed contribution *below* the
   threshold and at least one opposite-signed contribution. Bar fill colour
   encodes only the sign, so colour alone never answers the question and the
   printed contribution values must be read.
3. The sign and magnitude plan is constructed first and verified against the
   rendered sequence; an infeasible running-total window raises and resamples
   rather than relaxing bounds.
