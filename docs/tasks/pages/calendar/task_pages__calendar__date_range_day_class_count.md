# `task_pages__calendar__date_range_day_class_count`

## Identity
1. Domain: `pages`
2. Scene id: `calendar`
3. Source scene: `calendar`
4. Task id: `task_pages__calendar__date_range_day_class_count`

## Contract
1. Objective: count how many weekday or weekend dates lie in the inclusive date range bounded by the two marked dates in one month-view calendar.
2. Public task contract: `date_range_day_class_count`
3. Supported `query_id`: `weekday_range_count`, `weekend_range_count`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witnesses: all date cells inside the inclusive marked range that match the requested day class.
7. Query argument axes: requested day class `weekday|weekend`.

## Program Contract
- `calendar_month_date_range_day_class_count(day_class={weekday,weekend}, inclusive_boundary_dates); output=integer_value; annotation=bbox_set(matching_date_cells_in_range); scene=calendar; scope=one Gregorian month-view calendar`

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_v1`
2. Scene key: `month_calendar`
3. Task key: `calendar_date_range_day_class_count_query`
4. Prompt query keys: `weekday_range_count` and `weekend_range_count`
5. Trace records the month/year, displayed week start, marked boundary dates, counted dates, requested day class, answer count, and date-cell bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar metadata.
7. Displayed week start is a scene/rendering axis (`monday|sunday`) and is not a public `query_id`.
