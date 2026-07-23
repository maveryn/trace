# `task_charts__dumbbell__gap_rank_row_label`

## Contract
1. Domain: `charts`
2. Scene id: `dumbbell`
3. Source implementation domain/group: `charts/dumbbell`
4. Query ids: `largest_gap_rank_row_label`, `smallest_gap_rank_row_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dumbbell.gap_rank_row_label.ChartsDumbbellGapRankRowLabelTask`
2. Prompt lookup domain/group: `charts/dumbbell`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `segment`.
3. Annotation is one `segment`: two `[x, y]` pixel points formatted `[[x0, y0], [x1, y1]]`, connecting the two colored dot centers for the selected dumbbell row.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(row_label, abs(value(series_a,row_label)-value(series_b,row_label)), direction={largest,smallest}, rank={1,2}); output=string_label; annotation=segment(answer_row_dot_centers); scene=dumbbell; scope=gap_rank_row_label`

Candidate set: the visible paired endpoint markers, connectors, and category labels inside the `gap_rank_row_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `segment` witnesses bound by `segment(answer_row_dot_centers)`. Annotation is one `segment`: two `[x, y]` pixel points formatted `[[x0, y0], [x1, y1]]`, connecting the two colored dot centers for the selected dumbbell row. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_gap_rank_row_label`, `smallest_gap_rank_row_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_gap_rank_row_label` | `selection.gap_rank_row_label` | `string_label` | `segment` |
| `smallest_gap_rank_row_label` | `selection.gap_rank_row_label` | `string_label` | `segment` |

Rank position is task-internal metadata and is sampled only from `1` and `2`, yielding largest/second-largest or smallest/second-smallest prompts.
