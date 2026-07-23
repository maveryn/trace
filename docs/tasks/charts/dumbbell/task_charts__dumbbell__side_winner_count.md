# `task_charts__dumbbell__side_winner_count`

## Contract
1. Domain: `charts`
2. Scene id: `dumbbell`
3. Source implementation domain/group: `charts/dumbbell`
4. Query ids: `series_a_greater_threshold_count`, `series_b_greater_threshold_count`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dumbbell.side_winner_count.ChartsDumbbellSideWinnerCountTask`
2. Prompt lookup domain/group: `charts/dumbbell`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `segment_set`.
3. Annotation is a `segment_set`; each segment is two `[x, y]` pixel points formatted `[[x0, y0], [x1, y1]]` and connects the two colored dot centers for one matching dumbbell row.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(row where compare(value(winning_series,row), value(other_series,row), relation=greater_than)); output=integer_count; annotation=segment_set(matching_row_dot_centers); scene=dumbbell; scope=side_winner_count`

Candidate set: the visible paired endpoint markers, connectors, and category labels inside the `side_winner_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `segment_set` witnesses bound by `segment_set(matching_row_dot_centers)`. Annotation is a `segment_set`; each segment is two `[x, y]` pixel points formatted `[[x0, y0], [x1, y1]]` and connects the two colored dot centers for one matching dumbbell row. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `series_a_greater_threshold_count`, `series_b_greater_threshold_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `series_a_greater_threshold_count` | `count.side_winner_rows` | `integer_count` | `segment_set` |
| `series_b_greater_threshold_count` | `count.side_winner_rows` | `integer_count` | `segment_set` |
