# `task_pages__schedule__overlap_count`

## Identity
1. Domain: `pages`
2. Scene id: `schedule`
3. Source scene: `schedule`
4. Task id: `task_pages__schedule__overlap_count`

## Program Contract
1. Program schema: `schedule_overlap_count(reference_event) -> event_count; scene=schedule; scope=overlap_count`
2. Scene: `schedule`
3. Scope: one rendered single-day schedule with time labels, scheduled event blocks, and one highlighted reference event.
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Annotation roles: unordered event-block boxes for every non-reference event overlapping the highlighted reference event.
8. Query arguments: fixed overlap predicate; old source branch is recorded as `source_query_id=overlap_count`.
9. Render arguments: day label, event intervals, lane assignments, scene variant, style variant, accent color, render dimensions, and post-render noise.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Prompt + Trace
1. Prompt bundle: `pages_schedule_v1`
2. Scene key: `day_schedule`
3. Task key: `schedule_day_query`
4. Prompt query key: `overlap_count`
5. Trace records selected public `query_id`, source query id, event intervals, lane assignments, reference event id, answer event ids, event-block bboxes, style metadata, and projected annotation.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
