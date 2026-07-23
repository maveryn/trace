# `task_charts__radial_progress__extremum_remaining_label`

## Contract
1. Domain: `charts`
2. Scene id: `radial_progress`
3. Task id: `task_charts__radial_progress__extremum_remaining_label`
4. Objective contract: `extremum_remaining_label`
5. Supported `query_id` values: `highest_remaining_label`, `lowest_remaining_label`

## Program Contract

Program: `select_label(arg_extreme(radial_progress_widgets, 100 - progress_value(widget), direction=max|min)); scene=radial_progress; scope=extremum_remaining_label`

Candidate set: the visible radial progress rings, arcs, and labels inside the `extremum_remaining_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `select_label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `see_annotation_contract`. Annotation marks the single answer widget card bbox. Titles, tick marks, card decorations, and non-answer widgets are context, not annotation.
Query ids: `highest_remaining_label`, `lowest_remaining_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.radial_progress.extremum_remaining_label.ChartsRadialProgressExtremumRemainingLabelTask`
2. Prompt bundle: `charts_radial_progress_v1`
3. Scene key: `radial_progress_scene`
4. Task key: `radial_progress_remaining_extremum_query`
5. Query keys: `highest_remaining_label`, `lowest_remaining_label`

## Annotation Contract
1. Answer schema: `string_label`
2. Annotation schema: `bbox`
3. Annotation marks the single answer widget card bbox.
4. Titles, tick marks, card decorations, and non-answer widgets are context, not annotation.

## Query Details

| Query id | Program argument | Answer schema | Annotation schema |
|---|---|---|---|
| `highest_remaining_label` | `direction=max_remaining` | `string_label` | `bbox` |
| `lowest_remaining_label` | `direction=min_remaining` | `string_label` | `bbox` |
