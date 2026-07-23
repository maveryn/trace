# `task_charts__curve_panels__endpoint_rank_panel_label`

## Contract
1. Domain: `charts`
2. Scene id: `curve_panels`
3. Source implementation domain/group: `charts/curve_panels`
4. Query ids: `start_highest_panel_label`, `start_lowest_panel_label`, `end_highest_panel_label`, `end_lowest_panel_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.curve_panels.endpoint_rank_panel_label.ChartsScientificEndpointRankPanelLabelTask`
2. Prompt lookup domain/group: `charts/curve_panels`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `point`.
3. Annotation should mark the selected endpoint marker in the answer subplot.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(panel_label, value(method,endpoint={start,end},panel_label), direction={highest,lowest}); output=string_label; annotation=point(answer_endpoint_mark); scene=curve_panels; scope=endpoint_rank_panel_label`

Candidate set: the visible curve panels, curve traces, points, and panel labels inside the `endpoint_rank_panel_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `point` witnesses bound by `point(answer_endpoint_mark)`. Annotation should mark the selected endpoint marker in the answer subplot. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `start_highest_panel_label`, `start_lowest_panel_label`, `end_highest_panel_label`, `end_lowest_panel_label`.

## Reasoning Operations

Families: `ranking`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `start_highest_panel_label` | `selection.endpoint_rank_panel_label` | `string_label` | `point` |
| `start_lowest_panel_label` | `selection.endpoint_rank_panel_label` | `string_label` | `point` |
| `end_highest_panel_label` | `selection.endpoint_rank_panel_label` | `string_label` | `point` |
| `end_lowest_panel_label` | `selection.endpoint_rank_panel_label` | `string_label` | `point` |
