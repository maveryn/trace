# `task_charts__composition_panels__segment_count_extremum_panel_label`

## Contract
1. Domain: `charts`
2. Scene id: `composition_panels`
3. Public task id: `task_charts__composition_panels__segment_count_extremum_panel_label`
4. Query ids: `largest_count`, `smallest_count`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.composition_panels.segment_count_extremum_panel_label.ChartsCompositionPanelsSegmentCountExtremumPanelLabelTask`
2. Prompt bundle: `charts_composition_panels_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point_map`.
3. Annotation maps `segment_percent` to the selected panel's target-segment percentage label and `panel_total` to the selected panel's total-count text.

## Program Contract

Program: `select(panel where count(panel,target_segment) is extremum(direction)); output=string_label; annotation=point_map(segment_percent,panel_total); scene=composition_panels; scope=segment_count_extremum_panel_label`

Candidate set: the visible panel segments and panel/category labels inside the `segment_count_extremum_panel_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point_map` witnesses bound by `point_map(segment_percent,panel_total)`. Annotation maps `segment_percent` to the selected panel's target-segment percentage label and `panel_total` to the selected panel's total-count text.
Query ids: `largest_count`, `smallest_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_count` | `select(panel where count(panel,target_segment) is maximum)` | `string_label` | `point_map` |
| `smallest_count` | `select(panel where count(panel,target_segment) is minimum)` | `string_label` | `point_map` |
