# `task_pages__infographic__section_total_extrema_difference_value`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/section_total_extrema_difference_value.py`
4. Task id: `task_pages__infographic__section_total_extrema_difference_value`

## Program Contract
1. Program schema: `numeric.ranked_difference(candidate_sets=all_sections, aggregate=sum_printed_values, selected_sections=highest_total|lowest_total); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: total every section, identify the unique highest-total and lowest-total sections, then return highest total minus lowest total.
3. Public query id: `single`
4. Answer schema: `integer`
5. Annotation schema: `bbox_set_map` with keys `highest_total_section` and `lowest_total_section`, each containing that section's metric-card boxes.
6. Query argument axes: section totals derived from visible card values.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `ranking`, `aggregation`, `formula_evaluation`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Prompt query key: `section_total_extrema_difference`
5. Trace records every section total, selected high/low sections, answer expression, and supporting card boxes. Public annotation preserves the high/low section grouping while leaving label-keyed card boxes in trace diagnostics.
