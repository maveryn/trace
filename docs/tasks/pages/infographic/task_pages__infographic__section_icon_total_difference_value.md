# `task_pages__infographic__section_icon_total_difference_value`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/section_icon_total_difference_value.py`
4. Task id: `task_pages__infographic__section_icon_total_difference_value`

## Program Contract
1. Program schema: `numeric.difference_or_change(left=sum_cards_with_resolved_icon_in_section_a, right=sum_cards_with_resolved_icon_in_section_b, operator=absolute_difference); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: filter two named sections to cards with the same requested icon, sum each filtered group, and return the nonnegative difference between the two totals.
3. Public query id: `single`
4. Answer schema: `integer`
5. Annotation schema: `bbox_set_map` with keys `section_a_filtered_icon_cards` and `section_b_filtered_icon_cards`, each containing matching metric-card boxes from that section.
6. Query argument axes: section pair and comparison icon.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Prompt query key: `section_icon_total_difference_value`
5. Trace records compared sections, icon kind/label, filtered groups, group totals, answer, and matching card boxes. Public annotation preserves the two section groups while leaving label-keyed card boxes in trace diagnostics.
