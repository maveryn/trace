# `task_charts__composition_panels__composition_shift_l1_distance`

## Contract
1. Domain: `charts`
2. Scene id: `composition_panels`
3. Public task id: `task_charts__composition_panels__composition_shift_l1_distance`
4. Query id: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.composition_panels.composition_shift_l1_distance.ChartsCompositionPanelsCompositionShiftL1DistanceTask`
2. Prompt bundle: `charts_composition_panels_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox_set`.
3. Annotation marks the two full compared panel boxes.

## Program Contract

Program: `sum(abs(share(end_panel,segment)-share(start_panel,segment)) for segment in segments); output=integer_value; annotation=bbox_set(compared_panels); scene=composition_panels; scope=composition_shift_l1_distance`

Candidate set: the visible panel segments and panel/category labels inside the `composition_shift_l1_distance` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(compared_panels)`. Annotation marks the two full compared panel boxes.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `sum(abs(share(end_panel,segment)-share(start_panel,segment)) for segment in segments)` | `integer_value` | `bbox_set` |
