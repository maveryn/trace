# `task_pages__infographic__sum_named_metrics_value`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/sum_named_metrics_value.py`
4. Task id: `task_pages__infographic__sum_named_metrics_value`

## Program Contract
1. Program schema: `numeric.aggregate_sum(candidate_set=resolved_named_metric_cards, metric=printed_value); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: locate each named metric card in the prompt and return the sum of their printed values.
3. Public query id: `single`
4. Answer schema: `integer`
5. Annotation schema: `bbox_set` containing the supporting metric-card boxes.
6. Query argument axes: named metric-card labels and operand count.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `aggregation`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Prompt query key: `sum_named_metrics`
5. Trace records target labels, target values, arithmetic expression, answer, and supporting card boxes. Public annotation is an unordered homogeneous set of supporting metric-card boxes; label-keyed card boxes remain trace diagnostics.
