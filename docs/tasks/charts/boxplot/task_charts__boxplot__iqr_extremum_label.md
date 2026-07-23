# `task_charts__boxplot__iqr_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `boxplot`
3. Source implementation scene: `charts/boxplot`
4. Query ids: `largest_iqr_label`, `smallest_iqr_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.boxplot.iqr_extremum_label.ChartsDistributionBoxplotIqrExtremumLabelTask`
2. Prompt lookup source scene: `charts/boxplot`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `bbox`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(group_label, q3(group_label)-q1(group_label), direction={largest,smallest}); output=string_label; annotation=bbox(answer_iqr_box); scene=boxplot; scope=iqr_extremum_label`

Candidate set: the visible boxplot glyphs and their group labels inside the `iqr_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `bbox` witnesses bound by `bbox(answer_iqr_box)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_iqr_label`, `smallest_iqr_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_iqr_label` | `selection.boxplot_iqr_extremum_label` | `string_label` | `bbox` |
| `smallest_iqr_label` | `selection.boxplot_iqr_extremum_label` | `string_label` | `bbox` |
