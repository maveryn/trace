# `task_charts__curve_panels__earliest_maximum_panel_label`

## Contract
1. Domain: `charts`
2. Scene id: `curve_panels`
3. Source implementation domain/group: `charts/curve_panels`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.curve_panels.earliest_maximum_panel_label.ChartsScientificEarliestMaximumPanelLabelTask`
2. Prompt lookup domain/group: `charts/curve_panels`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `argmin(panel_label, x_at_max(value(method,x,panel_label))); output=string_label; annotation=point(answer_peak_mark); scene=curve_panels; scope=earliest_maximum_panel_label`

Candidate set: the visible curve panels, curve traces, points, and panel labels inside the `earliest_maximum_panel_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `argmin` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(answer_peak_mark)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selection.earliest_maximum_panel_label` | `string_label` | `point` |
