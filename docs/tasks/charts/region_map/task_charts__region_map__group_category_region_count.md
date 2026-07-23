# `task_charts__region_map__group_category_region_count`

## Contract
1. Domain: `charts`
2. Scene id: `region_map`
3. Source implementation path: `src/trace_tasks/tasks/charts/region_map/group_category_region_count.py`
4. Supported `query_id`: `single`
5. Public task id: `task_charts__region_map__group_category_region_count`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.region_map.group_category_region_count.ChartsMapGroupCategoryRegionCountTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/region_map/charts_region_map_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_value`.
2. Annotation schema: `point_set`.
3. Annotation contains center points for every visible country that is both inside the named continent and in the target category; legend, title, and context text are not annotation targets.
4. This task is geography-only and uses the `world_countries` geographic map variant. Selected countries must pass the region-map geographic component visibility constraints.
5. Answer values are constructed in the range `1..5`.

## Program Contract

Program: `count(region for region in visible_regions if group(region) == target_continent and category(region) == target_category); output=integer_value; annotation=point_set(filtered_regions); scene=region_map; scope=group_category_region_count`

Candidate set: the visible map regions, region labels, and legend/value encodings inside the `group_category_region_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_value` value bound by `integer_value`.
Annotation witnesses: `point_set` witnesses bound by `point_set(filtered_regions)`. Annotation contains center points for every visible country that is both inside the named continent and in the target category; legend, title, and context text are not annotation targets. This task is geography-only and uses the `world_countries` geographic map variant. Selected countries must pass the region-map geographic component visibility constraints. Answer values are constructed in the range `1..5`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count(region for region in visible_regions if group(region) == target_continent and category(region) == target_category)` | `integer_value` | `point_set` |
