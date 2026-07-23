# `task_pages__hero_callout_infographic__callout_metric_extremum_label`

## Identity
1. Domain: `pages`
2. Scene id: `hero_callout_infographic`
3. Source path: `src/trace_tasks/tasks/pages/hero_callout_infographic/callout_metric_extremum_label.py`
4. Task id: `task_pages__hero_callout_infographic__callout_metric_extremum_label`

## Program Contract
1. Program schema: `rank.callout_metric_extremum(field=resolved_field, rank_direction=highest|lowest); scene=hero_callout_infographic; scope=one poster-style hero infographic`
2. Contract: compare every visible value for one named field across callout cards and return the callout title with the unique highest or lowest value.
3. Public query contract: two semantic query branches select the unresolved extremum direction; target field remains a sampled operand recorded in trace metadata.
4. answer schema: `string`
5. Annotation schema: `bbox_map` with `winning_callout_card` and compared `candidate_N_field_row` boxes.
6. Supported `query_id`: `highest_field_value_callout_label`, `lowest_field_value_callout_label`
7. Prompt query key: `callout_metric_extremum_label`
8. scalar_annotation_checked=true

## Reasoning Operations

Families: `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_hero_callout_infographic_v1`
2. Scene key: `hero_callout_infographic`
3. Task key: `hero_callout_infographic_query`
4. Trace records the selected query id, compared callout values, parsed numeric values, query-derived rank direction, selected winner, sampled visual assets, sampled style metadata, final card/field-row bboxes, and layout geometry.
5. Generation guarantees that the selected field has at least three visible candidates and a unique extremum.

## Rendering Notes
1. This task reuses the hero-callout infographic renderer, layout variants, and pages-owned visual asset pool.
2. Annotation stays on the compared field-row witnesses and the winning callout card; decorative assets, badges, and connectors are visual context only.
