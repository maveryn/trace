# `task_pages__infographic__section_icon_total_value`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/section_icon_total_value.py`
4. Task id: `task_pages__infographic__section_icon_total_value`

## Program Contract
1. Program schema: `numeric.aggregate_sum(candidate_set=cards_in_resolved_section_with_resolved_icon, metric=printed_value); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: filter one named section to cards with the requested icon, then return the sum of their printed values.
3. Public query id: `single`
4. Answer schema: `integer`
5. Annotation schema: `bbox_set` containing the matching icon-filtered metric-card boxes.
6. Query argument axes: target section and icon filter.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `aggregation`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Prompt query key: `section_icon_total_value`
5. Trace records icon kind/label, filtered labels, target values, answer, and matching card boxes. Public annotation is an unordered homogeneous set of matching card boxes; label-keyed card boxes remain trace diagnostics.
