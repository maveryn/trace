# `task_charts__histogram__cumulative_rank_bin_label`

## Contract
1. Domain: `charts`
2. Scene id: `histogram`
3. Source implementation scene: `charts/histogram`
4. Query id: `single`
5. Semantic query details are recorded in trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.histogram.cumulative_rank_bin_label.ChartsDistributionHistogramCumulativeRankLabelTask`
2. Prompt lookup domain/scene: `charts/histogram`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox`.
3. Annotation marks the single histogram bar containing the requested cumulative item rank.
4. Renderer context such as axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `label(bin containing cumulative_rank(target_rank, counts_left_to_right)); output=integer_value; annotation=bbox(answer_bin); scene=histogram; scope=cumulative_rank_bin_label`

Candidate set: the visible histogram bins and axis/value labels inside the `cumulative_rank_bin_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `label` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_bin)`. Annotation marks the single histogram bar containing the requested cumulative item rank. Renderer context such as axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `aggregation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selection.cumulative_rank_bin_label` | `integer_value` | `bbox` |
