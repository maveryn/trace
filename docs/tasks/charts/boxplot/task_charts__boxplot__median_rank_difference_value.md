# `task_charts__boxplot__median_rank_difference_value`

## Contract
1. Domain: `charts`
2. Scene id: `boxplot`
3. Source implementation scene: `charts/boxplot`
4. Query ids: `median_top_second_difference_value`, `median_top_third_difference_value`, `median_top_bottom_difference_value`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.boxplot.median_rank_difference_value.ChartsDistributionBoxplotMedianRankDifferenceValueTask`
2. Prompt lookup source scene: `charts/boxplot`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_map`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `absolute_difference(median(group_at_rank=top), median(group_at_rank={second,third,bottom})); output=integer_value; annotation=point_map(upper_median, lower_median); scene=boxplot; scope=median_rank_difference_value`

Candidate set: the visible boxplot glyphs and their group labels inside the `median_rank_difference_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `absolute_difference` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_map` witnesses bound by `point_map(upper_median, lower_median)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `median_top_second_difference_value`, `median_top_third_difference_value`, `median_top_bottom_difference_value`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `median_top_second_difference_value` | `numeric.ranked_median_difference` | `integer_value` | `point_map` |
| `median_top_third_difference_value` | `numeric.ranked_median_difference` | `integer_value` | `point_map` |
| `median_top_bottom_difference_value` | `numeric.ranked_median_difference` | `integer_value` | `point_map` |
