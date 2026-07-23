# `task_pages__infographic__section_total_except_named_value`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/section_total_except_named_value.py`
4. Task id: `task_pages__infographic__section_total_except_named_value`

## Program Contract
1. Program schema: `numeric.aggregate_sum(candidate_set=cards_in_resolved_section_excluding_named_cards, metric=printed_value); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: in the named section, exclude the named metric cards and return the sum of the remaining printed values.
3. Public query id: `single`
4. Answer schema: `integer`
5. Annotation schema: `bbox_set` containing the included metric-card boxes after the named exclusions are removed.
6. Query argument axes: target section and excluded card labels.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `aggregation`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Prompt query key: `section_total_except_named`
5. Trace records included labels, excluded labels, target section, answer expression, and section card boxes. Public annotation marks only the included cards contributing to the final sum; excluded cards remain trace context.
