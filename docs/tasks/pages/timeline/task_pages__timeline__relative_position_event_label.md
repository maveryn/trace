# `task_pages__timeline__relative_position_event_label`

## Identity
1. Domain: `pages`
2. Scene id: `timeline`
3. Source scene: `timeline`
4. Task id: `task_pages__timeline__relative_position_event_label`

## Program Contract
1. Program schema: `timeline_relative_position_event_label(reference_date, direction, offset) -> event_label; scene=timeline; scope=relative_position_event_label`
2. Scene: `timeline`
3. Scope: one rendered milestone timeline with dated event cards and one prompt-specified reference date belonging to a visible event card.
4. Supported `query_id` values: `event_before_dated_event_label`, `event_after_dated_event_label`
5. Answer schema: `string`
6. Annotation schema: `bbox`
7. Annotation roles: bounding box of the target event card at the requested timeline offset.
8. Query arguments: `direction=before|after`; `offset=1..4` is sampled as a prompt parameter, not a query id.
9. Render arguments: month/year, event count, scene variant, style variant, accent color, render dimensions, and post-render noise.

## Reasoning Operations

Families: `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_timeline_v1`
2. Scene key: `milestone_timeline`
3. Task key: `timeline_milestone_query`
4. Prompt query keys: `event_before_dated_event_label`, `event_after_dated_event_label`
5. Trace records selected `query_id`, reference date, reference event id, offset, direction, target event id, event order, dates, event-card bboxes, style metadata, and projected annotation.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
