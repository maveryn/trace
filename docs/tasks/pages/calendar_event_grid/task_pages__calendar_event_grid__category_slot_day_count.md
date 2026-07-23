# `task_pages__calendar_event_grid__category_slot_day_count`

## Identity
1. Domain: `pages`
2. Scene id: `calendar_event_grid`
3. Source scene: `calendar_event_grid`
4. Task id: `task_pages__calendar_event_grid__category_slot_day_count`

## Contract
1. Objective: count dates whose requested event slot contains the requested category label.
2. Public task contract: `category_slot_day_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: matching event-chip boxes; an empty set is allowed only if no chips match.
7. Query argument axes: sampled category label and event slot label.

## Program Contract
- `calendar_event_grid_category_slot_day_count(category_label, slot_id); output=integer_value; annotation=bbox_set(matching_event_chips); scene=calendar_event_grid; scope=one month calendar with Top/Mid/Bottom event slots`

## Reasoning Operations

Families: `filtering`, `counting`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_event_grid_v1`
2. Scene key: `calendar_event_grid`
3. Task key: `calendar_event_grid_query`
4. Prompt query key: `category_slot_day_count`
5. Trace records the month/year, requested slot/category, matching chip ids, final date-cell bboxes, and final event-chip bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar event-grid metadata.
