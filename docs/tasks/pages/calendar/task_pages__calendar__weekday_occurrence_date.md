# `task_pages__calendar__weekday_occurrence_date`

## Identity
1. Domain: `pages`
2. Scene id: `calendar`
3. Source scene: `calendar`
4. Task id: `task_pages__calendar__weekday_occurrence_date`

## Contract
1. Objective: find the date number of an nth weekday in one Gregorian month-view calendar.
2. Public task contract: `weekday_occurrence_date`
3. Supported `query_id`: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox`
6. Annotation witness: the target date cell.
7. Query argument axes: sampled weekday name and occurrence ordinal.

## Program Contract
- `calendar_month_weekday_occurrence_lookup(weekday_index, occurrence); output=integer_value; annotation=bbox(target_date_cell); scene=calendar; scope=one Gregorian month-view calendar`

## Reasoning Operations

Families: `filtering`, `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_v1`
2. Scene key: `month_calendar`
3. Task key: `calendar_weekday_occurrence_query`
4. Prompt query key: `weekday_occurrence_date`
5. Trace records the month/year, displayed week start, weekday/ordinal operands, answer date, and date-cell bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar metadata.
7. Displayed week start is a scene/rendering axis (`monday|sunday`) and is not a public `query_id`.
