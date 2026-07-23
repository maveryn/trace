# `task_pages__hero_callout_infographic__callout_condition_count`

## Identity
1. Domain: `pages`
2. Scene id: `hero_callout_infographic`
3. Source path: `src/trace_tasks/tasks/pages/hero_callout_infographic/callout_condition_count.py`
4. Task id: `task_pages__hero_callout_infographic__callout_condition_count`

## Program Contract
1. Program schema: `count.callout_metric_threshold(field=resolved_field, predicate=above|below); scene=hero_callout_infographic; scope=one poster-style hero infographic`
2. Contract: count callout cards whose visible value for one named field satisfies one threshold predicate.
3. Public query contract: two semantic predicate branches; target field and threshold are query arguments.
4. answer schema: `integer`
5. Annotation schema: `bbox_set` containing the matching field-row boxes.
6. Supported `query_id`: `field_value_above_threshold_count`, `field_value_below_threshold_count`
7. Prompt query key: `callout_condition_count`
8. scalar_annotation_checked=true

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Prompt + Trace
1. Prompt bundle: `pages_hero_callout_infographic_v1`
2. Scene key: `hero_callout_infographic`
3. Task key: `hero_callout_infographic_query`
4. Trace records the selected predicate query id, threshold, operator, candidate callout values, matching values, sampled visual assets, sampled style metadata, final field-row bboxes, and layout geometry.
5. Threshold sampling keeps the answer nonzero and not all visible callouts for the selected field.

## Rendering Notes
1. This task reuses the hero-callout infographic renderer, layout variants, and pages-owned visual asset pool.
2. Decorative assets and connector lines are not annotation; only matching field rows are prompt-facing annotation.
