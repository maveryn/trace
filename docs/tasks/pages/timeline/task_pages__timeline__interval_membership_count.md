# `task_pages__timeline__interval_membership_count`

## Identity
1. Domain: `pages`
2. Scene id: `timeline`
3. Source scene: `timeline`
4. Task id: `task_pages__timeline__interval_membership_count`

## Program Contract
1. Program schema: `timeline_interval_membership_count(reference_event_pair, interval_relation) -> event_count; scene=timeline; scope=interval_membership_count`
2. Scene: `timeline`
3. Scope: one rendered milestone timeline with dated event cards and two highlighted reference events.
4. Supported `query_id` values: `between_reference_events_count`, `outside_reference_interval_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Annotation roles: unordered event-card boxes for every counted event; empty set is valid for zero-count samples.
8. Query arguments: `interval_relation=between|outside`.
9. Render arguments: month/year, event count, scene variant, style variant, accent color, render dimensions, and post-render noise.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `logical_composition`

## Prompt + Trace
1. Prompt bundle: `pages_timeline_v1`
2. Scene key: `milestone_timeline`
3. Task key: `timeline_milestone_query`
4. Prompt query keys: `between_reference_events_count`, `outside_reference_interval_count`
5. Trace records selected `query_id`, `interval_relation`, event order, dates, reference event ids, answer event ids, event-card bboxes, style metadata, and projected annotation.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
