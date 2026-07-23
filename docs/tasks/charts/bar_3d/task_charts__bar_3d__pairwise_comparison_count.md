# `task_charts__bar_3d__pairwise_comparison_count`

## Contract
1. Domain: `charts`
2. Scene id: `bar_3d`
3. Source implementation scene: `charts/bar_3d`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.bar_3d.pairwise_comparison_count.ChartsThreeDBarPairwiseComparisonCountTask`
2. Prompt lookup scene: `charts/bar_3d`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `segment_set`.
3. Annotation is a `segment_set`; each segment is two `[x, y]` pixel points formatted `[[x0, y0], [x1, y1]]` and connects the top-center points of the two compared bars for one counted category.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(category where compare(value(category,series_a), value(category,series_b), relation=greater_than)); output=integer_count; annotation=segment_set(comparison_bar_top_center_pairs); scene=bar_3d; scope=pairwise_comparison_count`

Candidate set: the visible 3D bars grouped by category and series inside the `pairwise_comparison_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `segment_set` witnesses bound by `segment_set(comparison_bar_top_center_pairs)`. Annotation is a `segment_set`; each segment is two `[x, y]` pixel points formatted `[[x0, y0], [x1, y1]]` and connects the top-center points of the two compared bars for one counted category. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count.pairwise_series_comparison` | `integer_count` | `segment_set` |
