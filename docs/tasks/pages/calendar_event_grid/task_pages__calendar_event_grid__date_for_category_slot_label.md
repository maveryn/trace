# `task_pages__calendar_event_grid__date_for_category_slot_label`

## Identity
1. Domain: `pages`
2. Scene id: `calendar_event_grid`
3. Source scene: `calendar_event_grid`
4. Task id: `task_pages__calendar_event_grid__date_for_category_slot_label`

## Contract
1. Objective: find the unique date whose requested event slot contains the requested category label.
2. Public task contract: `date_for_category_slot_label`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox`
6. Annotation witness: the matching event-chip box.
7. Query argument axes: sampled category label and event slot label.

## Program Contract
- `calendar_event_grid_date_for_category_slot_lookup(category_label, slot_id); output=integer_date_number; annotation=bbox(matching_event_chip); scene=calendar_event_grid; scope=one month calendar with Top/Mid/Bottom event slots`

## Reasoning Operations

Families: `direct_retrieval`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_event_grid_v1`
2. Scene key: `calendar_event_grid`
3. Task key: `calendar_event_grid_query`
4. Prompt query key: `date_for_category_slot_label`
5. Trace records the month/year, requested slot/category, matching chip id, final date-cell bboxes, and final event-chip bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar event-grid metadata.
