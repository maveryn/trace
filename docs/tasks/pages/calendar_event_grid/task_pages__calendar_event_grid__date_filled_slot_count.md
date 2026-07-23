# `task_pages__calendar_event_grid__date_filled_slot_count`

## Identity
1. Domain: `pages`
2. Scene id: `calendar_event_grid`
3. Source scene: `calendar_event_grid`
4. Task id: `task_pages__calendar_event_grid__date_filled_slot_count`

## Contract
1. Objective: count how many event slots on a requested calendar date contain an event chip.
2. Public task contract: `date_filled_slot_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: the event-chip boxes on the requested date; an empty set is allowed when no slots are filled.
7. Query argument axes: sampled date number.

## Program Contract
- `calendar_event_grid_date_filled_slot_count(date_number); output=integer_value; annotation=bbox_set(date_event_chips); scene=calendar_event_grid; scope=one month calendar with Top/Mid/Bottom event slots`

## Reasoning Operations

Families: `counting`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_event_grid_v1`
2. Scene key: `calendar_event_grid`
3. Task key: `calendar_event_grid_query`
4. Prompt query key: `date_filled_slot_count`
5. Trace records the month/year, requested date, matching chip ids, final date-cell bboxes, and final event-chip bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar event-grid metadata.
