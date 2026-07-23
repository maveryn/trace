# `task_charts__part_whole__sector_share_to_angle`

## Contract
1. Domain: `charts`
2. Scene id: `part_whole`
3. Supported `query_id`: `clockwise_sector_angle`, `counterclockwise_sector_angle`
4. Answer schema: `integer_value`
5. Annotation schema: `point_map`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.part_whole.sector_share_to_angle.ChartsCompositionChartSectorShareToAngleTask`
2. Prompt lookup domain/scene: `charts/part_whole`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `degrees(sum(value(category) for category in contiguous_circular_span(start_category, end_category, direction))); scene=part_whole; scope=sector_share_to_angle`

Candidate set: the visible part-whole segments, slices, and category labels inside the `sector_share_to_angle` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `degrees` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `unspecified`.
Annotation witnesses: `unspecified` witnesses bound by `see_annotation_contract`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `clockwise_sector_angle`, `counterclockwise_sector_angle`.

## Reasoning Operations

Families: `aggregation`, `topology`, `formula_evaluation`

## Annotation Contract
Annotation maps each included category label to the `[x,y]` pixel point at the center of its chart segment.

## Query Details

| Query id | Direction | Answer schema | Annotation schema |
|---|---|---|---|
| `clockwise_sector_angle` | clockwise | `integer_value` | `point_map` |
| `counterclockwise_sector_angle` | counterclockwise | `integer_value` | `point_map` |
