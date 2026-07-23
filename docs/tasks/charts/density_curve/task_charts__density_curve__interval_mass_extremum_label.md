# `task_charts__density_curve__interval_mass_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `density_curve`
3. Source implementation domain/scene: `charts/density_curve`
4. Query ids: `greatest_interval_mass_label`, `least_interval_mass_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.density_curve.interval_mass_extremum_label.ChartsDistributionDensityCurveIntervalMassExtremumLabelTask`
2. Prompt lookup domain/scene: `charts/density_curve`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation should mark a point on the answer curve inside the marked interval, not the legend label, title, interval label, or full interval region.
4. Renderer context such as legends, axes, interval guides, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(curve_label, integral(density(curve_label), selected_interval), direction={greatest,least}); output=string_label; annotation=point(answer_interval_mass_curve_point); scene=density_curve; scope=interval_mass_extremum_label`

Candidate set: the visible density curves, shaded regions, and axis labels inside the `interval_mass_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(answer_interval_mass_curve_point)`. Annotation should mark a point on the answer curve inside the marked interval, not the legend label, title, interval label, or full interval region. Renderer context such as legends, axes, interval guides, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `greatest_interval_mass_label`, `least_interval_mass_label`.

## Reasoning Operations

Families: `ranking`, `aggregation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `greatest_interval_mass_label` | `selection.interval_mass_extremum_label` | `string_label` | `point` |
| `least_interval_mass_label` | `selection.interval_mass_extremum_label` | `string_label` | `point` |
