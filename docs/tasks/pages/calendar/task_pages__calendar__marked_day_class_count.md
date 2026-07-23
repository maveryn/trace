# `task_pages__calendar__marked_day_class_count`

## Identity
1. Domain: `pages`
2. Scene id: `calendar`
3. Source scene: `calendar`
4. Task id: `task_pages__calendar__marked_day_class_count`

## Contract
1. Objective: count how many marked dates in one month-view calendar fall on the requested day class.
2. Public task contract: `marked_day_class_count`
3. Supported `query_id`: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witnesses: all marked date cells that match the requested day class; use an empty set when the answer is 0.
7. Query argument axes: sampled marked day class `weekday|weekend`.

## Program Contract
- `calendar_month_marked_day_class_count(marked_day_class={weekday,weekend}); output=integer_value; annotation=bbox_set(matching_marked_date_cells); scene=calendar; scope=one Gregorian month-view calendar`

## Reasoning Operations

Families: `filtering`, `counting`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_v1`
2. Scene key: `month_calendar`
3. Task key: `calendar_marked_day_class_count_query`
4. Prompt query key: `marked_day_class_count`
5. Trace records the month/year, displayed week start, marked dates, annotation dates, day-class operand, and date-cell bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar metadata.
7. Displayed week start is a scene/rendering axis (`monday|sunday`) and is not a public `query_id`.
