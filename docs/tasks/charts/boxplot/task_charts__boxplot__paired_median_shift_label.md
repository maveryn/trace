# `task_charts__boxplot__paired_median_shift_label`

## Contract
1. Domain: `charts`
2. Scene id: `boxplot`
3. Source implementation scene: `charts/boxplot`
4. Query ids: `paired_median_greatest_increase_label`, `paired_median_greatest_decrease_label`, `paired_median_greatest_absolute_change_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.boxplot.paired_median_shift_label.ChartsDistributionBoxplotPairedMedianShiftLabelTask`
2. Prompt lookup source scene: `charts/boxplot`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point_map`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(pair_label, median(after_boxplot)-median(before_boxplot), mode={increase,decrease,absolute_change}); output=string_label; annotation=point_map(before_boxplot, after_boxplot); scene=boxplot; scope=paired_median_shift_label`

Candidate set: the visible boxplot glyphs and their group labels inside the `paired_median_shift_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point_map` witnesses bound by `point_map(before_boxplot, after_boxplot)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `paired_median_greatest_increase_label`, `paired_median_greatest_decrease_label`, `paired_median_greatest_absolute_change_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `paired_median_greatest_increase_label` | `selection.paired_median_shift_extremum_label` | `string_label` | `point_map` |
| `paired_median_greatest_decrease_label` | `selection.paired_median_shift_extremum_label` | `string_label` | `point_map` |
| `paired_median_greatest_absolute_change_label` | `selection.paired_median_shift_extremum_label` | `string_label` | `point_map` |
