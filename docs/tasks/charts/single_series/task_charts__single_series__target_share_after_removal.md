# `task_charts__single_series__target_share_after_removal`

## Contract
1. Domain: `charts`
2. Scene id: `single_series`
3. Source implementation: `src/trace_tasks/tasks/charts/single_series/target_share_after_removal.py`
4. Public task id: `task_charts__single_series__target_share_after_removal`
5. Supported `query_id` values: `single`
6. Query ids are internal replay metadata; scene style, label pool, mark count, and context mode are generation metadata.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.single_series.target_share_after_removal.ChartsHypotheticalTargetShareAfterRemovalPublicTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/single_series/charts_hypothetical_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `percent_share(value(target_label), sum(values(marks excluding removed_labels))); output=integer_value; annotation=point_set(retained_marks); scene=single_series; scope=target_share_after_removal`

Candidate set: the visible marks in the ordered single-series chart inside the `target_share_after_removal` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `percent_share` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_set` witnesses bound by `point_set(retained_marks)`. Annotation marks every retained visible mark used in the denominator, including the target mark. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `state_update`, `formula_evaluation`

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_set`.
3. Annotation marks every retained visible mark used in the denominator, including the target mark.
4. Axes, legend, titles, captions, decorative context, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `percent_share.target_value_after_named_removal` | `integer_value` | `point_set` |
