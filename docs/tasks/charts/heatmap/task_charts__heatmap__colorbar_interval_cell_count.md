# `task_charts__heatmap__colorbar_interval_cell_count`

## Contract
1. Domain: `charts`
2. Scene id: `heatmap`
3. Source implementation: `src/trace_tasks/tasks/charts/heatmap/colorbar_interval_cell_count.py`
4. Query id: `single`
5. Interval bounds are sampled generation parameters, not public query branches.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.heatmap.colorbar_interval_cell_count.ChartsHeatmapColorbarIntervalCellCountTask`
2. Prompt lookup: `src/trace_tasks/resources/prompts/charts/heatmap/charts_heatmap_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `count(filter(cells, lower_bound <= colorbar_value(cell) <= upper_bound)); output=integer_count; annotation=bbox_set(counted_cells); scene=heatmap; scope=colorbar_interval_cell_count`

Candidate set: the visible heatmap cells with row and column labels inside the `colorbar_interval_cell_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(counted_cells)`. Annotation marks rendered bboxes for exactly the counted cells in row-major order. Colorbar, axes, labels, titles, and distractor text are context unless the task explicitly asks for them as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. Annotation marks rendered bboxes for exactly the counted cells in row-major order.
4. Colorbar, axes, labels, titles, and distractor text are context unless the task explicitly asks for them as annotation.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count.cells.colorbar_interval` | `integer_count` | `bbox_set` |
