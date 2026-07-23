# `task_charts__composition_panels__segment_count_nearest_target_panel_label`

## Contract
1. Domain: `charts`
2. Scene id: `composition_panels`
3. Public task id: `task_charts__composition_panels__segment_count_nearest_target_panel_label`
4. Query ids: `single`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.composition_panels.segment_count_nearest_target_panel_label.ChartsCompositionPanelsSegmentCountNearestTargetPanelLabelTask`
2. Prompt bundle: `charts_composition_panels_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks the full answer panel.

## Program Contract

Program: `select(panel where abs(count(panel,target_segment)-target_count) is minimum); output=string_label; annotation=bbox(answer_panel); scene=composition_panels; scope=segment_count_nearest_target_panel_label`

Candidate set: the visible panel segments and panel/category labels inside the `segment_count_nearest_target_panel_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `select` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_panel)`. Annotation marks the full answer panel.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `select(panel where abs(count(panel,target_segment)-target_count) is minimum)` | `string_label` | `bbox` |
