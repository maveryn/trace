# `task_charts__region_map__adjacent_same_category_count`

## Contract
1. Domain: `charts`
2. Scene id: `region_map`
3. Source implementation path: `src/trace_tasks/tasks/charts/region_map/adjacent_same_category_count.py`
4. Supported `query_id`: `single`
5. Public task id: `task_charts__region_map__adjacent_same_category_count`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.region_map.adjacent_same_category_count.ChartsMapAdjacentSameCategoryCountTask`
2. Prompt bundle: `src/trace_tasks/resources/prompts/charts/region_map/charts_region_map_v1.json`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Annotation Contract
1. Answer schema: `integer_count`.
2. Annotation schema: `bbox_set`.
3. Annotation marks the counted or summed map-region boxes only; legend, title, and context text are not annotation targets.
4. Adjacent-region tasks identify the reference region by its short in-map marker label: a letter or compact alphanumeric code capped at 4 characters; pure-number marker labels are not used.
5. Annotation marks matching neighbors and excludes the labeled reference region.

## Program Contract

Program: `count(filter(adjacent(reference_region_label), category == category(reference_region_label))); output=integer_count; annotation=bbox_set(matching_neighbor_regions); scene=region_map; scope=adjacent_same_category_count`

Candidate set: the visible map regions, region labels, and legend/value encodings inside the `adjacent_same_category_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `integer_count` value bound by `integer_count`.
Annotation witnesses: `bbox_set` witnesses bound by `bbox_set(matching_neighbor_regions)`. Annotation marks the counted or summed map-region boxes only; legend, title, and context text are not annotation targets. Adjacent-region tasks identify the reference region by its short in-map marker label: a letter or compact alphanumeric code capped at 4 characters; pure-number marker labels are not used. Annotation marks matching neighbors and excludes the labeled reference region.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count(adjacent(reference_region_label) where category == category(reference_region_label))` | `integer_count` | `bbox_set` |
