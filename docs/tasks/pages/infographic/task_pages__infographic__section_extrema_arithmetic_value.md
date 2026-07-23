# `task_pages__infographic__section_extrema_arithmetic_value`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/section_extrema_arithmetic_value.py`
4. Task id: `task_pages__infographic__section_extrema_arithmetic_value`

## Program Contract
1. Program schema: `numeric.ranked_difference(candidate_sets=two_resolved_sections, selected_metrics=section_extrema, operator=sum|absolute_difference); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: find the requested maximum/minimum metric card in each of two named sections, then return the requested arithmetic result from those two printed values.
3. Public query id: `single`
4. Answer schema: `integer`
5. Annotation schema: role-keyed `bbox_map` with keys `section_a_extremum` and `section_b_extremum`.
6. Query argument axes: section pair, extremum kind per section, and arithmetic operator.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `ranking`, `aggregation`, `formula_evaluation`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Prompt query key: `section_extrema_arithmetic`
5. Trace records selected sections, extrema roles, operand labels/values, operator, answer, and supporting card boxes. Public annotation uses semantic operand-role keys rather than visible card-label keys.
