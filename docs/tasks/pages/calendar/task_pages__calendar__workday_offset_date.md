# `task_pages__calendar__workday_offset_date`

## Identity
1. Domain: `pages`
2. Scene id: `calendar`
3. Source scene: `calendar`
4. Task id: `task_pages__calendar__workday_offset_date`

## Contract
1. Objective: find the date number reached by moving a requested number of workdays before or after one marked reference date, skipping Saturdays and Sundays.
2. Public task contract: `workday_offset_date`
3. Supported `query_id`: `workday_after_offset_date`, `workday_before_offset_date`
4. Answer type: `integer`
5. Annotation schema: `bbox_map`
6. Annotation witnesses: `reference_date` and `target_date` date cells keyed by role.
7. Query argument axes: workday direction and offset count.

## Program Contract
- `calendar_month_workday_offset(direction={after,before}, offset); output=integer_value; annotation=bbox_map(reference_date,target_date); scene=calendar; scope=one Gregorian month-view calendar`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_v1`
2. Scene key: `month_calendar`
3. Task key: `calendar_workday_offset_query`
4. Prompt query keys: `workday_after_offset_date` and `workday_before_offset_date`
5. Trace records the month/year, displayed week start, reference date, target date, direction, offset, weekend indices, and date-cell bboxes.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar metadata.
7. Displayed week start is a scene/rendering axis (`monday|sunday`) and is not a public `query_id`.
