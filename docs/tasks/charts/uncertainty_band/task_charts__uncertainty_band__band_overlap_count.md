# `task_charts__uncertainty_band__band_overlap_count`

## Public Contract

1. Domain: `charts`
2. Scene: `uncertainty_band`
3. Source file: `src/trace_tasks/tasks/charts/uncertainty_band/band_overlap_count.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/uncertainty_band/charts_uncertainty_band_v1.json`
5. Supported sampled `query_id`: `single`

## Program Contract

Program: `count(x_label where intersects(vertical_interval(series_a_band_at_x), vertical_interval(series_b_band_at_x))); output=integer_value; annotation=point_set(overlap_region_centers); scene=uncertainty_band; scope=band_overlap_count`

Candidate set: the visible uncertainty bands, central series marks, and x labels inside the `band_overlap_count` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `count` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `integer_value`.
Annotation witnesses: `unspecified` witnesses bound by `point_set(overlap_region_centers)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Contract Notes

This task uses the current source layout. The public task file owns overlap construction, answer binding, annotation binding, query metadata, and prompt slots; scene-local shared code only provides uncertainty-band data structures, rendering, prompt, and projection primitives.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `count(x_label where intersects(vertical_interval(series_a_band_at_x), vertical_interval(series_b_band_at_x)))` | `integer_value` | `point_set` |
