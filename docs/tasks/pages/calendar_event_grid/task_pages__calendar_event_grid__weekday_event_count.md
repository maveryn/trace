# `task_pages__calendar_event_grid__weekday_event_count`

## Identity
1. Domain: `pages`
2. Scene id: `calendar_event_grid`
3. Source scene: `calendar_event_grid`
4. Task id: `task_pages__calendar_event_grid__weekday_event_count`

## Contract
1. Objective: count all event chips shown in a requested weekday column.
2. Public task contract: `weekday_event_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: event-chip boxes in the requested weekday column.
7. Query argument axes: sampled weekday label and matching visible weekday-header abbreviation.

## Program Contract
- `calendar_event_grid_weekday_event_count(weekday_label); output=integer_value; annotation=bbox_set(weekday_event_chips); scene=calendar_event_grid; scope=one month calendar with Top/Mid/Bottom event slots`

## Reasoning Operations

Families: `counting`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_event_grid_v1`
2. Scene key: `calendar_event_grid`
3. Task key: `calendar_event_grid_query`
4. Prompt query key: `weekday_event_count`
5. Trace records the month/year, requested weekday, visible weekday header, matching chip ids, final date-cell bboxes, and final event-chip bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar event-grid metadata.
