# `task_charts__region_map__named_region_set_total_value`

## Contract
1. Domain: `charts`
2. Scene id: `region_map`
3. Source implementation path: `src/trace_tasks/tasks/charts/region_map/named_region_set_total_value.py`
4. Supported `query_id`: `single`
5. Public task id: `task_charts__region_map__named_region_set_total_value`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.region_map.named_region_set_total_value.ChartsMapNamedRegionSetTotalValueTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/region_map/charts_region_map_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `bbox_set`.
3. Annotation contains pixel boxes around every map region included in the total; legend, title, and context text are not annotation targets.
4. This task uses the synthetic region-map variant only; geographic map variants are intentionally unsupported for this objective.

## Program Contract

Program: `sum(value(region) for region in named_region_set); output=integer_value; annotation=bbox_set(named_region_set); scene=region_map; scope=named_region_set_total_value`

Candidate set: the visible map regions, region labels, and legend/value encodings inside the `named_region_set_total_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `sum` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(named_region_set)`. Annotation contains pixel boxes around every map region included in the total; legend, title, and context text are not annotation targets. This task uses the synthetic region-map variant only; geographic map variants are intentionally unsupported for this objective.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `sum(value(region) for region in named_region_set)` | `integer_value` | `bbox_set` |
