# `task_charts__region_map__categorical_region_count`

## Contract
1. Domain: `charts`
2. Scene id: `region_map`
3. Source implementation path: `src/trace_tasks/tasks/charts/region_map/categorical_region_count.py`
4. Supported `query_id`: `single`
5. Public task id: `task_charts__region_map__categorical_region_count`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.region_map.categorical_region_count.ChartsMapCategoricalRegionCountTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/region_map/charts_region_map_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `point_set`.
3. Annotation contains center points for every counted map region; legend, title, and context text are not annotation targets.
4. Geographic variants use only selected regions whose largest projected connected component is at least `400 px^2`; only that largest component is colored and annotated.

## Program Contract

Program: `count(filter(regions, category == target_category)); output=integer_count; annotation=point_set(matching_regions); scene=region_map; scope=categorical_region_count`

Candidate set: the visible map regions, region labels, and legend/value encodings inside the `categorical_region_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `point_set` witnesses bound by `point_set(matching_regions)`. Annotation contains center points for every counted map region; legend, title, and context text are not annotation targets. Geographic variants use only selected regions whose largest projected connected component is at least `400 px^2`; only that largest component is colored and annotated.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count(regions where category == target_category)` | `integer_count` | `point_set` |
