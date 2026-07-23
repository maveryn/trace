# `task_pages__calendar_event_grid__busiest_date_label`

## Identity
1. Domain: `pages`
2. Scene id: `calendar_event_grid`
3. Source scene: `calendar_event_grid`
4. Task id: `task_pages__calendar_event_grid__busiest_date_label`

## Contract
1. Objective: find the unique date with the most event chips.
2. Public task contract: `busiest_date_label`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: event-chip boxes on the busiest date.
7. Query argument axes: sampled month/year and busiest date.

## Program Contract
- `calendar_event_grid_busiest_date_label(); output=integer_date_number; annotation=bbox_set(event_chips_on_unique_busiest_date); scene=calendar_event_grid; scope=one month calendar with Top/Mid/Bottom event slots`

## Reasoning Operations

Families: `filtering`, `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_event_grid_v1`
2. Scene key: `calendar_event_grid`
3. Task key: `calendar_event_grid_query`
4. Prompt query key: `busiest_date_label`
5. Trace records the month/year, busiest date, per-date event counts, matching chip ids, final date-cell bboxes, and final event-chip bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar event-grid metadata.
