# `task_charts__candlestick__range_extremum_label`

## Contract
1. Domain: `charts`
2. Scene id: `candlestick`
3. Source implementation scene: `charts/candlestick`
4. Query ids: `largest_wick_range_label`, `smallest_wick_range_label`, `largest_body_range_label`, `smallest_body_range_label`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.candlestick.range_extremum_label.ChartsCandlestickRangeExtremumLabelTask`
2. Prompt lookup domain/scene: `charts/candlestick`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `string_label`.
2. Annotation schema: `segment`.
3. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `arg_extreme(candle_label, range(candle_label, range_kind={wick,body}), direction={largest,smallest}); output=string_label; annotation=segment(answer_candle_range_mark); scene=candlestick; scope=range_extremum_label`

Candidate set: the visible candlestick glyphs and period/category labels inside the `range_extremum_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `string_label` value bound by `string_label`.
Annotation witnesses: `segment` witnesses bound by `segment(answer_candle_range_mark)`. Annotation should mark the minimal visual witnesses required by the task, following the cross-domain annotation policy. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `largest_wick_range_label`, `smallest_wick_range_label`, `largest_body_range_label`, `smallest_body_range_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `largest_wick_range_label` | `selection.candle_range_extremum_label` | `string_label` | `segment` |
| `smallest_wick_range_label` | `selection.candle_range_extremum_label` | `string_label` | `segment` |
| `largest_body_range_label` | `selection.candle_range_extremum_label` | `string_label` | `segment` |
| `smallest_body_range_label` | `selection.candle_range_extremum_label` | `string_label` | `segment` |
