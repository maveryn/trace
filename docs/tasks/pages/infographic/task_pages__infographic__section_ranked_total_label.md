# `task_pages__infographic__section_ranked_total_label`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/section_ranked_total_label.py`
4. Task id: `task_pages__infographic__section_ranked_total_label`

## Program Contract
1. Program schema: `selection.ranked_item(candidate_set=sections, metric=sum_printed_values, rank_direction=highest|lowest); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: sum every section's metric-card values, rank sections by total, and return the requested section label.
3. Public query id: `single`
4. Answer schema: `string`
5. Annotation schema: `bbox_set` containing the metric-card boxes in the answer section.
6. Query argument axes: rank direction and rank position.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `ranking`, `aggregation`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Prompt query key: `section_ranked_total_label`
5. Trace records section totals, rank direction, rank position, answer section, and answer-section card boxes. Public annotation is an unordered homogeneous set of selected-section metric-card boxes; label-keyed card boxes remain trace diagnostics.
