# `task_charts__dashboard__source_rank_target_value`

## Contract
1. Domain: `charts`
2. Scene id: `dashboard`
3. Source implementation domain/group: `charts/dashboard`
4. Query ids: `largest_source_rank_target_value`, `smallest_source_rank_target_value`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dashboard.source_rank_target_value.ChartsDashboardSourceRankTargetValueTask`
2. Prompt lookup domain/group: `charts/dashboard`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_map`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `value(category_from_source_rank,target_panel); output=integer_value; annotation=point_map(source_panel, target_panel); scene=dashboard; scope=source_rank_target_value`

Candidate set: the visible dashboard panels, linked marks, and panel labels inside the `source_rank_target_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `value` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_map` witnesses bound by `point_map(source_panel, target_panel)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_source_rank_target_value`, `smallest_source_rank_target_value`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_source_rank_target_value` | `numeric.source_rank_target_lookup` | `integer_value` | `point_map` |
| `smallest_source_rank_target_value` | `numeric.source_rank_target_lookup` | `integer_value` | `point_map` |
