# `task_charts__candlestick__counterfactual_close_value`

## Contract
1. Domain: `charts`
2. Scene id: `candlestick`
3. Source implementation scene: `charts/candlestick`
4. Query ids: `close_after_body_increase_value`, `close_after_body_decrease_value`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.candlestick.counterfactual_close_value.ChartsCandlestickCounterfactualCloseValueTask`
2. Prompt lookup domain/scene: `charts/candlestick`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `counterfactual_close(candle_label, body_change_direction={increase,decrease}); output=integer_value; annotation=point(target_body_center); scene=candlestick; scope=counterfactual_close_value`

Candidate set: the visible candlestick glyphs and period/category labels inside the `counterfactual_close_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `counterfactual_close` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point` witnesses bound by `point(target_body_center)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `close_after_body_increase_value`, `close_after_body_decrease_value`.

## Reasoning Operations

Families: `state_update`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `close_after_body_increase_value` | `numeric.counterfactual_candle_close` | `integer_value` | `point` |
| `close_after_body_decrease_value` | `numeric.counterfactual_candle_close` | `integer_value` | `point` |
