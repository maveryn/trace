# `task_pages__hero_callout_infographic__callout_composite_metric_extremum_label`

## Identity
1. Domain: `pages`
2. Scene id: `hero_callout_infographic`
3. Source path: `src/trace_tasks/tasks/pages/hero_callout_infographic/callout_composite_metric_extremum_label.py`
4. Task id: `task_pages__hero_callout_infographic__callout_composite_metric_extremum_label`

## Program Contract
1. Program schema: `rank.callout_composite_metric_extremum(fields=[Score, Count], operation=sum, rank_direction=highest|lowest); scene=hero_callout_infographic; scope=one poster-style hero infographic`
2. Contract: compare every callout card by adding the visible numeric values for the two named fields, then return the callout title with the unique highest or lowest combined value.
3. Public query contract: two semantic query branches select the unresolved extremum direction.
4. answer schema: `string`
5. Annotation schema: `bbox_map` with `winning_callout_card` and compared `candidate_N_first_field_row` / `candidate_N_second_field_row` boxes.
6. Supported `query_id`: `highest_composite_metric_callout_label`, `lowest_composite_metric_callout_label`
7. Prompt query key: `callout_composite_metric_extremum_label`
8. scalar_annotation_checked=true

## Reasoning Operations

Families: `ranking`, `aggregation`

## Prompt + Trace
1. Prompt bundle: `pages_hero_callout_infographic_v1`
2. Scene key: `hero_callout_infographic`
3. Task key: `hero_callout_infographic_query`
4. Trace records the selected query id, compared callout values, parsed numeric values, per-callout composite values, selected winner, sampled visual assets, sampled style metadata, final card/field-row bboxes, and layout geometry.
5. Generation guarantees that every compared callout contains both fields and that the composite extremum is unique.

## Rendering Notes
1. This task reuses the hero-callout infographic renderer, layout variants, and pages-owned visual asset pool.
2. Annotation stays on compared field-row witnesses and the winning callout card; decorative assets, badges, and connectors are visual context only.
