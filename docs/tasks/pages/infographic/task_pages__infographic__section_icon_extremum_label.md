# `task_pages__infographic__section_icon_extremum_label`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/section_icon_extremum_label.py`
4. Task id: `task_pages__infographic__section_icon_extremum_label`

## Program Contract
1. Program schema: `selection.extreme_metric_label(candidate_set=sections, metric=sum_cards_with_resolved_icon, rank_direction=highest|lowest); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: for each section, sum cards with the requested icon, then return the unique section with the highest or lowest filtered total.
3. Public query id: `single`
4. Answer schema: `string`
5. Annotation schema: scalar `bbox` around the answer section panel.
6. Query argument axes: icon filter and extremum direction.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `filtering`, `ranking`, `aggregation`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Prompt query key: `section_icon_extremum_label`
5. Trace records filtered section totals, icon kind/label, rank direction, answer section, and answer-section panel bbox.
