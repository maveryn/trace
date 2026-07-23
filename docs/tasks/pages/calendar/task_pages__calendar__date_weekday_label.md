# `task_pages__calendar__date_weekday_label`

## Identity
1. Domain: `pages`
2. Scene id: `calendar`
3. Source scene: `calendar`
4. Task id: `task_pages__calendar__date_weekday_label`

## Contract
1. Objective: find the visible weekday header label above a queried date in one month-view calendar.
2. Public task contract: `date_weekday_label`
3. Supported `query_id`: `single`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: the queried date cell.
7. Query argument axes: sampled target date number.

## Program Contract
- `calendar_month_date_weekday_header_lookup(target_date); output=visible_weekday_header_label; annotation=bbox(target_date_cell); scene=calendar; scope=one Gregorian month-view calendar`

## Reasoning Operations

Families: `direct_retrieval`

## Prompt + Trace
1. Prompt bundle: `pages_calendar_v1`
2. Scene key: `month_calendar`
3. Task key: `calendar_date_weekday_label_query`
4. Prompt query key: `date_weekday_label`
5. Trace records the month/year, displayed week start, target date, semantic weekday index, exact answer label, valid answer labels, and date-cell bbox.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized calendar metadata.
7. Displayed week start is a scene/rendering axis (`monday|sunday`) and is not a public `query_id`; the answer string must match the visible three-letter weekday header label (`Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, or `Sun`).
