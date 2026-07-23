# `task_charts__histogram__interval_mass`

## Contract
1. Domain: `charts`
2. Scene id: `histogram`
3. Source implementation scene: `charts/histogram`
4. Query ids: `single`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.histogram.interval_mass.ChartsDistributionHistogramIntervalMassTask`
2. Prompt lookup domain/scene: `charts/histogram`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox_set`.
3. Annotation marks the histogram bars included in the requested inside interval total.
4. Renderer context such as axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `sum(count(bin) for bin where bin is inside query_interval); output=integer_value; annotation=bbox_set(included_bins); scene=histogram; scope=interval_mass`

Candidate set: the visible histogram bins and axis/value labels inside the `interval_mass` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(included_bins)`. Annotation marks the histogram bars included in the requested inside interval total. Renderer context such as axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `numeric.interval_mass(relation=inside)` | `integer_value` | `bbox_set` |
