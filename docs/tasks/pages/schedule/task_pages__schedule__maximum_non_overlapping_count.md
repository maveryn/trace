# `task_pages__schedule__maximum_non_overlapping_count`

## Identity
1. Domain: `pages`
2. Scene id: `schedule`
3. Source scene: `schedule`
4. Task id: `task_pages__schedule__maximum_non_overlapping_count`

## Program Contract
1. Program schema: `schedule_maximum_non_overlapping_count(event_set) -> event_count; scene=schedule; scope=maximum_non_overlapping_count`
2. Scene: `schedule`
3. Scope: one rendered single-day schedule with time labels and scheduled event blocks.
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Answer support: integers `2` through `5`.
7. Annotation schema: `bbox_set`
8. Annotation roles: unordered event-block boxes for the unique maximum-cardinality non-overlapping event set.
9. Query arguments: fixed maximum compatible subset objective; old source branch is recorded as `source_query_id=maximum_non_overlapping_count`.
10. Render arguments: day label, event intervals, lane assignments, scene variant, style variant, accent color, render dimensions, and post-render noise.

Generation brute-force checks the sampled event intervals and accepts only
instances with exactly one maximum-cardinality non-overlapping event set. Lane
assignment is separately randomized so the answer set cannot be read off from a
single schedule column.

## Reasoning Operations

Families: `counting`, `ranking`, `spatial_relations`

## Prompt + Trace
1. Prompt bundle: `pages_schedule_v1`
2. Scene key: `day_schedule`
3. Task key: `schedule_day_query`
4. Prompt query key: `maximum_non_overlapping_count`
5. Trace records selected public `query_id`, source query id, event intervals, lane assignments, answer event ids, event-block bboxes, style metadata, and projected annotation.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
