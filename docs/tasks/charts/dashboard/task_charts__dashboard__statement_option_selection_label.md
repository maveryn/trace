# `task_charts__dashboard__statement_option_selection_label`

## Contract
1. Domain: `charts`
2. Scene id: `dashboard`
3. Source implementation domain/group: `charts/dashboard`
4. Query id: `single`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.dashboard.statement_option_selection_label.ChartsDashboardStatementOptionSelectionLabelTask`
2. Prompt lookup domain/group: `charts/dashboard`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `option_letter`.
2. Annotation schema: `point_set`.
3. The rendered statement-option panel uses either `4` options (`A..D`) or `6` options (`A..F`) by construction.
4. Annotation should mark the two chart marks that verify the selected rendered statement option, not the option text itself.
5. Renderer context such as legends, axes, decorative labels, titles, distractor text, and statement-option text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `select(option_letter where truth(statement(option_letter), dashboard_values)==target_truth); output=option_letter; annotation=point_set(verifying_marks); scene=dashboard; scope=statement_option_selection_label`

Candidate set: the visible dashboard panels, linked marks, and panel labels inside the `statement_option_selection_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `select` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `option_letter` value bound by `option_letter`.
Annotation witnesses: `point_set` witnesses bound by `point_set(verifying_marks)`. The rendered statement-option panel uses either `4` options (`A..D`) or `6` options (`A..F`) by construction. Annotation should mark the two chart marks that verify the selected rendered statement option, not the option text itself. Renderer context such as legends, axes, decorative labels, titles, distractor text, and statement-option text is metadata unless the task explicitly asks for it as annotation.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `selection.statement_option_label` | `option_letter` | `point_set` |
