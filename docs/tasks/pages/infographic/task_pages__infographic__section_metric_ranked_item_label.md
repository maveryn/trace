# `task_pages__infographic__section_metric_ranked_item_label`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/section_metric_ranked_item_label.py`
4. Task id: `task_pages__infographic__section_metric_ranked_item_label`

## Program Contract
1. Program schema: `selection.ranked_item(candidate_set=metric_cards_in_resolved_section, metric=printed_value, rank_direction=highest|lowest); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: restrict attention to the named section, rank that section's metric cards by printed value, then return the label of the card at the requested rank.
3. Public query ids: `nth_highest_metric_in_section_label`, `nth_lowest_metric_in_section_label`
4. Answer schema: `string`
5. Annotation schema: scalar `bbox` around the selected metric card.
6. Query argument axes: rank direction, rank position, and target section.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Trace records the target section, ranked section candidates, rank direction, rank position, selected label/value, the section panel bbox, and selected metric-card bbox. The section bbox is trace context; public annotation is only the selected metric-card bbox.
5. Generation guarantees unique printed values within the ranked section.
