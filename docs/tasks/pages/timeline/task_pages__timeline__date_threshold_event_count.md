# `task_pages__timeline__date_threshold_event_count`

## Identity
1. Domain: `pages`
2. Scene id: `timeline`
3. Source scene: `timeline`
4. Task id: `task_pages__timeline__date_threshold_event_count`

## Program Contract
1. Program schema: `timeline_date_threshold_event_count(threshold_date, side) -> event_count; scene=timeline; scope=date_threshold_event_count`
2. Scene: `timeline`
3. Scope: one rendered milestone timeline with dated event cards and one prompt-specified threshold date.
4. Supported `query_id` values: `before_threshold_date_count`, `after_threshold_date_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Annotation roles: unordered event-card boxes for every counted event card.
8. Query arguments: `side=before|after`; counted cards must be strictly before or strictly after the prompt date.
9. Render arguments: month/year, event count, scene variant, style variant, accent color, render dimensions, and post-render noise.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Prompt + Trace
1. Prompt bundle: `pages_timeline_v1`
2. Scene key: `milestone_timeline`
3. Task key: `timeline_milestone_query`
4. Prompt query keys: `before_threshold_date_count`, `after_threshold_date_count`
5. Trace records selected `query_id`, threshold relation, threshold date, event order, dates, answer event ids, event-card bboxes, style metadata, and projected annotation.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
