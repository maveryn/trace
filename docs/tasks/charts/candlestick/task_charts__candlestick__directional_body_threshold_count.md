# `task_charts__candlestick__directional_body_threshold_count`

## Contract
1. Domain: `charts`
2. Scene id: `candlestick`
3. Source implementation scene: `charts/candlestick`
4. Query ids: `up_body_threshold_count`, `down_body_threshold_count`
5. Semantic query details are recorded in `query_id` and trace params.

## Implementation
1. Registered class: `trace_tasks.tasks.charts.candlestick.directional_body_threshold_count.ChartsCandlestickDirectionalBodyThresholdCountTask`
2. Prompt lookup domain/scene: `charts/candlestick`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. The annotation marks the open-close body of every candle that closes on the requested side of its open by more than the prompt-bound threshold.
4. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.

## Program Contract

Program: `count(filter(candles, direction(candle) = direction and body_size(candle) > threshold)); direction={up,down}; output=integer_count; annotation=bbox_set(matching_candle_bodies); scene=candlestick; scope=directional_body_threshold_count`

Candidate set: the visible candlestick glyphs and period/category labels inside the `directional_body_threshold_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(matching_candle_bodies)`. The annotation marks each qualifying candle body. Renderer context such as legends, axes, decorative labels, titles, and distractor text is metadata unless the task explicitly asks for it as annotation.
Query ids: `up_body_threshold_count`, `down_body_threshold_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `up_body_threshold_count` | `selection.directional_body_threshold_count` | `integer_count` | `bbox_set` |
| `down_body_threshold_count` | `selection.directional_body_threshold_count` | `integer_count` | `bbox_set` |

## Generation Notes
1. The target answer is drawn from a fixed support before the candle count, so every
   answer stays equally likely; the candle count is then widened to fit the drawn
   target.
2. Every instance carries at least one same-direction candle whose body is at or
   below the threshold, plus at least two candles in each direction. Candle fill
   colour encodes only direction, so colour alone never answers the question and
   the printed open and close values must be read.
3. Body and wick sizes keep the scene's uniqueness constraints; an infeasible
   open/close window raises and resamples rather than relaxing bounds.
