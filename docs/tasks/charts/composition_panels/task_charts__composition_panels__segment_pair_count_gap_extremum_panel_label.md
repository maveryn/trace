# `task_charts__composition_panels__segment_pair_count_gap_extremum_panel_label`

## Contract
1. Domain: `charts`
2. Scene id: `composition_panels`
3. Public task id: `task_charts__composition_panels__segment_pair_count_gap_extremum_panel_label`
4. Query ids: `largest_count_gap`, `smallest_count_gap`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.composition_panels.segment_pair_count_gap_extremum_panel_label.ChartsCompositionPanelsSegmentPairCountGapExtremumPanelLabelTask`
2. Prompt bundle: `charts_composition_panels_v1`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation marks the full answer panel.

## Program Contract

Program: `select(panel where abs(count(panel,segment_a)-count(panel,segment_b)) is extremum(direction)); output=string_label; annotation=bbox(answer_panel); scene=composition_panels; scope=segment_pair_count_gap_extremum_panel_label`

Candidate set: the visible panel segments and panel/category labels inside the `segment_pair_count_gap_extremum_panel_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_panel)`. Annotation marks the full answer panel.
Query ids: `largest_count_gap`, `smallest_count_gap`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_count_gap` | `select(panel where abs(count(panel,segment_a)-count(panel,segment_b)) is maximum)` | `string_label` | `bbox` |
| `smallest_count_gap` | `select(panel where abs(count(panel,segment_a)-count(panel,segment_b)) is minimum)` | `string_label` | `bbox` |
